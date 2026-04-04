"""GitHub Device Flow (RFC 8628) implementation.

Manages the device authorization flow for CLI login:
1. Client calls initiate() -> gets device_code + user_code
2. User visits github.com/login/device and enters user_code
3. Client polls poll() until user authorizes -> gets GitHub user info

State is stored in-memory (does not survive server restart).
"""

import secrets
import string
import time
from dataclasses import dataclass, field

import httpx

GITHUB_DEVICE_VERIFY_URI = "https://github.com/login/device"
GITHUB_DEVICE_AUTH_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API = "https://api.github.com/user"
GITHUB_USER_EMAILS_API = "https://api.github.com/user/emails"

DEFAULT_EXPIRES_IN = 900  # 15 minutes
DEFAULT_INTERVAL = 5  # seconds

# Ambiguous characters removed for user-friendliness
_USER_CODE_CHARS = string.ascii_uppercase.replace("I", "").replace(
    "O", ""
) + string.digits.replace("0", "").replace("1", "")


class DeviceFlowError(Exception):
    """Base error for device flow."""


class DeviceCodePendingError(DeviceFlowError):
    """User has not yet authorized the device code."""


class DeviceCodeExpiredError(DeviceFlowError):
    """Device code has expired."""


class DeviceCodeNotFoundError(DeviceFlowError):
    """Device code not found in store."""


@dataclass
class DeviceCodeEntry:
    """In-memory entry for a pending device authorization."""

    device_code: str
    user_code: str
    client_id: str
    expires_at: float
    interval: int
    github_access_token: str | None = field(default=None)

    @property
    def is_expired(self) -> bool:
        """Return True if the device code has expired."""
        return time.monotonic() >= self.expires_at

    @property
    def is_authorized(self) -> bool:
        """Return True if user has already authorized this device code."""
        return self.github_access_token is not None


def _generate_user_code() -> str:
    """Generate an 8-char alphanumeric user code (e.g. XXXX-XXXX display)."""
    return "".join(secrets.choice(_USER_CODE_CHARS) for _ in range(8))


class DeviceFlowStatus:
    """Status constants for device flow entries."""

    FOUND = "found"
    EXPIRED = "expired"
    MISSING = "missing"


class DeviceFlowStore:
    """In-memory store for pending device flow authorizations."""

    def __init__(self) -> None:
        self._entries: dict[str, DeviceCodeEntry] = {}
        self._by_user_code: dict[str, str] = {}

    def create(
        self,
        client_id: str,
        expires_in: int = DEFAULT_EXPIRES_IN,
        interval: int = DEFAULT_INTERVAL,
        device_code: str | None = None,
        user_code: str | None = None,
    ) -> DeviceCodeEntry:
        """Create and store a new device code entry."""
        device_code = device_code or secrets.token_urlsafe(32)
        user_code = user_code or _generate_user_code()
        entry = DeviceCodeEntry(
            device_code=device_code,
            user_code=user_code,
            client_id=client_id,
            expires_at=time.monotonic() + expires_in,
            interval=interval,
        )
        self._entries[device_code] = entry
        self._by_user_code[user_code] = device_code
        return entry

    def lookup(self, device_code: str) -> tuple[DeviceCodeEntry | None, str]:
        """Look up entry by device code.

        Returns:
            Tuple of (entry, status) where status is one of
            DeviceFlowStatus.FOUND / EXPIRED / MISSING.
        """
        entry = self._entries.get(device_code)
        if entry is None:
            return None, DeviceFlowStatus.MISSING
        if entry.is_expired:
            return entry, DeviceFlowStatus.EXPIRED
        return entry, DeviceFlowStatus.FOUND

    def lookup_by_user_code(self, user_code: str) -> tuple[DeviceCodeEntry | None, str]:
        """Look up entry by user code."""
        device_code = self._by_user_code.get(user_code)
        if device_code is None:
            return None, DeviceFlowStatus.MISSING
        return self.lookup(device_code)

    def mark_authorized(self, device_code: str, github_access_token: str) -> None:
        """Store the GitHub access token for an authorized device code."""
        entry = self._entries.get(device_code)
        if entry is not None:
            entry.github_access_token = github_access_token

    def remove(self, device_code: str) -> None:
        """Remove a device code entry from the store."""
        entry = self._entries.pop(device_code, None)
        if entry is not None:
            self._by_user_code.pop(entry.user_code, None)

    def cleanup_expired(self) -> None:
        """Remove all expired entries from the store."""
        expired = [dc for dc, e in self._entries.items() if e.is_expired]
        for dc in expired:
            self.remove(dc)


class DeviceFlowManager:
    """Manages GitHub Device Flow authorization.

    Supports the two-phase flow:
    1. initiate() — creates a device_code + user_code pair in the store
    2. poll(device_code) — checks authorization status and returns user info
    """

    def __init__(self, client_id: str) -> None:
        self._client_id = client_id
        self._store = DeviceFlowStore()

    async def initiate(self) -> dict[str, str | int]:
        """Start a new device flow by requesting codes from GitHub.

        Calls GitHub's device/code endpoint to get a real device_code
        and user_code, then stores them locally for poll tracking.
        """
        self._store.cleanup_expired()

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

        # Store for poll tracking (use public API, not private attrs)
        self._store.create(
            client_id=self._client_id,
            expires_in=expires_in,
            interval=interval,
            device_code=device_code,
            user_code=user_code,
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
        entry, flow_status = self._store.lookup(device_code)
        if flow_status == DeviceFlowStatus.EXPIRED:
            self._store.remove(device_code)
            raise DeviceCodeExpiredError("Device code expired")
        if flow_status == DeviceFlowStatus.MISSING:
            raise DeviceCodeNotFoundError("Device code not found")

        assert entry is not None  # status == "found" guarantees non-None
        if entry.is_authorized:
            user_info = await self._fetch_github_user(
                entry.github_access_token  # type: ignore[arg-type]
            )
            self._store.remove(device_code)
            return user_info

        token = await self._exchange_device_code(device_code=device_code)
        if token is None:
            raise DeviceCodePendingError("Authorization pending")

        self._store.mark_authorized(device_code, token)
        user_info = await self._fetch_github_user(token)
        self._store.remove(device_code)
        return user_info

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
