"""GitHub Device Flow (RFC 8628) implementation.

Manages the device authorization flow for CLI login:
1. Client calls initiate() -> gets device_code + user_code
2. User visits github.com/login/device and enters user_code
3. Client polls poll() until user authorizes -> gets GitHub user info

State is stored via AuthStateStore (in-memory by default).
"""

import httpx

from atp.dashboard.auth.state_store import AuthStateStore, get_auth_state_store

GITHUB_DEVICE_VERIFY_URI = "https://github.com/login/device"
GITHUB_DEVICE_AUTH_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API = "https://api.github.com/user"
GITHUB_USER_EMAILS_API = "https://api.github.com/user/emails"

DEFAULT_EXPIRES_IN = 900  # 15 minutes
DEFAULT_INTERVAL = 5  # seconds

# Key prefix for device flow entries in AuthStateStore
_KEY_PREFIX = "device:"


class DeviceFlowError(Exception):
    """Base error for device flow."""


class DeviceCodePendingError(DeviceFlowError):
    """User has not yet authorized the device code."""


class DeviceCodeExpiredError(DeviceFlowError):
    """Device code has expired."""


class DeviceCodeNotFoundError(DeviceFlowError):
    """Device code not found in store."""


class DeviceFlowManager:
    """Manages GitHub Device Flow authorization.

    Uses the unified AuthStateStore for state persistence,
    sharing the same storage backend as SSO and SAML flows.

    Supports the two-phase flow:
    1. initiate() — creates a device_code + user_code pair in the store
    2. poll(device_code) — checks authorization status and returns user info
    """

    def __init__(
        self,
        client_id: str,
        store: AuthStateStore | None = None,
    ) -> None:
        self._client_id = client_id
        self._store = store or get_auth_state_store()

    async def initiate(self) -> dict[str, str | int]:
        """Start a new device flow by requesting codes from GitHub.

        Calls GitHub's device/code endpoint to get a real device_code
        and user_code, then stores them locally for poll tracking.
        """
        # Request device code from GitHub
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                GITHUB_DEVICE_AUTH_URL,
                data={
                    "client_id": self._client_id,
                    "scope": "read:user user:email",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            github_data = resp.json()

        device_code = github_data["device_code"]
        user_code = github_data["user_code"]
        verification_uri = github_data.get("verification_uri", GITHUB_DEVICE_VERIFY_URI)
        expires_in = github_data.get("expires_in", DEFAULT_EXPIRES_IN)
        interval = github_data.get("interval", DEFAULT_INTERVAL)

        # Store for poll tracking
        await self._store.put(
            f"{_KEY_PREFIX}{device_code}",
            {
                "client_id": self._client_id,
                "user_code": user_code,
                "interval": interval,
                "github_access_token": None,
            },
            ttl_seconds=expires_in,
        )

        return {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": verification_uri,
            "expires_in": expires_in,
            "interval": interval,
        }

    async def poll(self, device_code: str) -> dict[str, str | int | None]:
        """Poll for authorization status.

        Returns user info dict on success.
        Raises DeviceCodeNotFoundError, DeviceCodeExpiredError, or
        DeviceCodePendingError depending on state.
        """
        key = f"{_KEY_PREFIX}{device_code}"
        entry = await self._store.get(key)

        if entry is None:
            raise DeviceCodeNotFoundError("Device code not found or expired")

        # Check if already authorized (from a previous poll)
        if entry.get("github_access_token"):
            await self._store.pop(key)
            return await self._fetch_github_user(entry["github_access_token"])

        # Try exchanging the device code for a token
        token = await self._exchange_device_code(device_code=device_code)
        if token is None:
            raise DeviceCodePendingError("Authorization pending")

        # Mark authorized and fetch user info
        entry["github_access_token"] = token
        await self._store.put(key, entry, ttl_seconds=60)
        await self._store.pop(key)
        return await self._fetch_github_user(token)

    async def _exchange_device_code(self, device_code: str) -> str | None:
        """Exchange a device code for an access token via GitHub token endpoint.

        Returns the access token string, or None if still pending.
        Raises DeviceCodeExpiredError or DeviceFlowError on fatal errors.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        error = data.get("error")
        if error in ("authorization_pending", "slow_down"):
            return None
        if error == "expired_token":
            raise DeviceCodeExpiredError("Device code expired at GitHub")
        if error == "access_denied":
            raise DeviceFlowError("User denied access")
        if error:
            raise DeviceFlowError(f"GitHub error: {error}")

        return data.get("access_token")  # type: ignore[return-value]

    async def _fetch_github_user(
        self, access_token: str
    ) -> dict[str, str | int | None]:
        """Fetch GitHub user profile and primary verified email.

        Returns a dict with login, email, name, id, avatar_url.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }
            resp = await client.get(GITHUB_USER_API, headers=headers)
            resp.raise_for_status()
            profile = resp.json()

            email: str | None = profile.get("email")
            if not email:
                resp = await client.get(GITHUB_USER_EMAILS_API, headers=headers)
                resp.raise_for_status()
                for e in resp.json():
                    if e.get("primary") and e.get("verified"):
                        email = e["email"]
                        break

        return {
            "login": profile.get("login"),
            "email": email,
            "name": profile.get("name"),
            "id": profile.get("id"),
            "avatar_url": profile.get("avatar_url"),
        }
