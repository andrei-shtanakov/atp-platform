# GitHub OAuth + Device Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub as an OIDC provider and implement Device Flow (RFC 8628) for CLI login via atp-sdk.

**Architecture:** GitHub is added as a new `OIDCProvider` enum value with a preset in `ProviderPresets`. Device Flow is implemented as two new endpoints (`POST /auth/device`, `POST /auth/device/poll`) that manage device codes in an in-memory store, exchange user codes with GitHub's OAuth device flow, provision users via JIT, and return ATP JWT tokens. Config is driven by env vars `ATP_GITHUB_CLIENT_ID` / `ATP_GITHUB_CLIENT_SECRET`.

**Tech Stack:** FastAPI, httpx, pydantic, PyJWT, SQLAlchemy (async), pytest + anyio

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `packages/atp-dashboard/atp/dashboard/auth/sso/oidc.py` | Add `GITHUB` to `OIDCProvider`, add `ProviderPresets.github()` |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add `github_client_id`, `github_client_secret` to `DashboardConfig` |
| Create | `packages/atp-dashboard/atp/dashboard/auth/device_flow.py` | Device Flow logic: initiate, poll, in-memory store, GitHub exchange |
| Create | `packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py` | `POST /auth/device` and `POST /auth/device/poll` endpoints |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register `device_auth_router` |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/routes/sso.py` | Add GitHub to provider list |
| Create | `tests/unit/dashboard/test_device_flow.py` | Unit tests for device flow logic |
| Create | `tests/unit/dashboard/test_device_auth_routes.py` | Route-level tests for device auth endpoints |

---

### Task 1: Add GitHub to OIDCProvider enum and ProviderPresets

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/auth/sso/oidc.py:33-40` (enum), `:159-287` (presets)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/sso.py:510-561` (provider list)

- [ ] **Step 1: Add GITHUB to OIDCProvider enum**

In `packages/atp-dashboard/atp/dashboard/auth/sso/oidc.py`, add `GITHUB = "github"` to the `OIDCProvider` enum:

```python
class OIDCProvider(StrEnum):
    """Supported OIDC providers."""

    OKTA = "okta"
    AUTH0 = "auth0"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    GITHUB = "github"
    GENERIC = "generic"
```

- [ ] **Step 2: Add ProviderPresets.github() static method**

Add after the `google()` method in `ProviderPresets`:

```python
@staticmethod
def github(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Get GitHub preset configuration.

    Args:
        client_id: GitHub OAuth App client ID.
        client_secret: GitHub OAuth App client secret.
        redirect_uri: OAuth2 callback URL.

    Returns:
        Configuration dict for SSOConfig.
    """
    return {
        "provider": OIDCProvider.GITHUB,
        "client_id": client_id,
        "client_secret": client_secret,
        "issuer_url": "https://github.com",
        "redirect_uri": redirect_uri,
        "scopes": ["read:user", "user:email"],
        "authorization_endpoint": "https://github.com/login/oauth/authorize",
        "token_endpoint": "https://github.com/login/oauth/access_token",
        "userinfo_endpoint": "https://api.github.com/user",
        "email_claim": "email",
        "name_claim": "name",
        "username_claim": "login",
    }
```

- [ ] **Step 3: Add GitHub to provider list in SSO routes**

In `packages/atp-dashboard/atp/dashboard/v2/routes/sso.py`, add a `ProviderPresetResponse` for GitHub in the `list_providers` endpoint, after Google and before Generic:

```python
ProviderPresetResponse(
    provider=OIDCProvider.GITHUB,
    name="GitHub",
    description="GitHub OAuth for developer authentication",
    required_fields=["client_id", "client_secret", "redirect_uri"],
    optional_fields=[],
    documentation_url="https://docs.github.com/en/apps/oauth-apps/building-oauth-apps",
),
```

Also add a handler in `get_provider_preset` for `OIDCProvider.GITHUB`:

```python
elif provider == OIDCProvider.GITHUB:
    return ProviderPresets.github(
        client_id="<your-client-id>",
        client_secret="<your-client-secret>",
        redirect_uri="<your-redirect-uri>",
    )
```

- [ ] **Step 4: Run tests to verify nothing broke**

Run: `cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform && uv run pytest tests/unit/dashboard/test_auth.py -v`
Expected: All existing tests PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/auth/sso/oidc.py packages/atp-dashboard/atp/dashboard/v2/routes/sso.py
git commit -m "feat(auth): add GitHub as OIDC provider with preset"
```

---

### Task 2: Add GitHub OAuth config to DashboardConfig

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py:59-67`

- [ ] **Step 1: Add github_client_id and github_client_secret fields**

In `DashboardConfig`, add after the `disable_auth` field:

```python
# GitHub OAuth settings (for Device Flow)
github_client_id: str | None = Field(
    default=None,
    description="GitHub OAuth App client ID for device flow",
)
github_client_secret: str | None = Field(
    default=None,
    description="GitHub OAuth App client secret for device flow",
)
```

- [ ] **Step 2: Update to_dict to redact secrets**

In `to_dict()`, add:

```python
"github_client_id": self.github_client_id,
"github_client_secret": "***" if self.github_client_secret else None,
```

- [ ] **Step 3: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/config.py
git commit -m "feat(config): add GitHub OAuth client ID/secret settings"
```

---

### Task 3: Implement Device Flow logic module

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/auth/device_flow.py`
- Test: `tests/unit/dashboard/test_device_flow.py`

- [ ] **Step 1: Write tests for device flow logic**

Create `tests/unit/dashboard/test_device_flow.py`:

```python
"""Tests for GitHub Device Flow logic."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from atp.dashboard.auth.device_flow import (
    DeviceCodeExpiredError,
    DeviceCodeNotFoundError,
    DeviceCodePendingError,
    DeviceFlowManager,
    DeviceFlowStore,
)


class TestDeviceFlowStore:
    """Tests for in-memory device flow store."""

    def test_create_device_code(self) -> None:
        """Test creating a device code entry."""
        store = DeviceFlowStore()
        entry = store.create(
            client_id="test-client",
            expires_in=900,
            interval=5,
        )
        assert entry.device_code is not None
        assert entry.user_code is not None
        assert len(entry.user_code) == 8  # XXXX-XXXX without dash = 8
        assert entry.client_id == "test-client"
        assert entry.interval == 5

    def test_get_by_device_code(self) -> None:
        """Test retrieving entry by device code."""
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        found = store.get_by_device_code(entry.device_code)
        assert found is not None
        assert found.device_code == entry.device_code

    def test_get_by_user_code(self) -> None:
        """Test retrieving entry by user code."""
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        found = store.get_by_user_code(entry.user_code)
        assert found is not None
        assert found.user_code == entry.user_code

    def test_get_expired_entry(self) -> None:
        """Test that expired entries are not returned."""
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=0, interval=5)
        found = store.get_by_device_code(entry.device_code)
        assert found is None

    def test_remove(self) -> None:
        """Test removing an entry."""
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        store.remove(entry.device_code)
        found = store.get_by_device_code(entry.device_code)
        assert found is None

    def test_mark_authorized(self) -> None:
        """Test marking an entry as authorized."""
        store = DeviceFlowStore()
        entry = store.create(client_id="test", expires_in=900, interval=5)
        store.mark_authorized(
            entry.device_code,
            github_access_token="gho_abc123",
        )
        found = store.get_by_device_code(entry.device_code)
        assert found is not None
        assert found.github_access_token == "gho_abc123"

    def test_cleanup_expired(self) -> None:
        """Test cleaning up expired entries."""
        store = DeviceFlowStore()
        store.create(client_id="test", expires_in=0, interval=5)
        store.create(client_id="test2", expires_in=900, interval=5)
        store.cleanup_expired()
        assert len(store._entries) == 1


class TestDeviceFlowManager:
    """Tests for Device Flow manager."""

    @pytest.mark.anyio
    async def test_initiate_creates_entry(self) -> None:
        """Test initiating device flow creates store entry."""
        manager = DeviceFlowManager(
            client_id="test-client",
            client_secret="test-secret",
        )
        result = manager.initiate()
        assert "device_code" in result
        assert "user_code" in result
        assert result["verification_uri"] == (
            "https://github.com/login/device"
        )
        assert result["expires_in"] == 900
        assert result["interval"] == 5

    @pytest.mark.anyio
    async def test_poll_pending(self) -> None:
        """Test polling when authorization is pending."""
        manager = DeviceFlowManager(
            client_id="test-client",
            client_secret="test-secret",
        )
        result = manager.initiate()
        with pytest.raises(DeviceCodePendingError):
            await manager.poll(result["device_code"])

    @pytest.mark.anyio
    async def test_poll_expired(self) -> None:
        """Test polling with expired device code."""
        manager = DeviceFlowManager(
            client_id="test-client",
            client_secret="test-secret",
        )
        # Create with 0 expiry
        entry = manager._store.create(
            client_id="test-client",
            expires_in=0,
            interval=5,
        )
        with pytest.raises(DeviceCodeExpiredError):
            await manager.poll(entry.device_code)

    @pytest.mark.anyio
    async def test_poll_not_found(self) -> None:
        """Test polling with unknown device code."""
        manager = DeviceFlowManager(
            client_id="test-client",
            client_secret="test-secret",
        )
        with pytest.raises(DeviceCodeNotFoundError):
            await manager.poll("nonexistent-code")

    @pytest.mark.anyio
    async def test_poll_authorized(self) -> None:
        """Test polling after user authorized."""
        manager = DeviceFlowManager(
            client_id="test-client",
            client_secret="test-secret",
        )
        result = manager.initiate()

        # Simulate authorization
        manager._store.mark_authorized(
            result["device_code"],
            github_access_token="gho_test123",
        )

        # Mock GitHub user info fetch
        mock_user_info = {
            "login": "testuser",
            "email": "test@example.com",
            "name": "Test User",
            "id": 12345,
            "avatar_url": "https://avatars.githubusercontent.com/u/12345",
        }
        with patch.object(
            manager,
            "_fetch_github_user",
            new_callable=AsyncMock,
            return_value=mock_user_info,
        ):
            user_info = await manager.poll(result["device_code"])
            assert user_info["login"] == "testuser"
            assert user_info["email"] == "test@example.com"
```

- [ ] **Step 2: Run tests to see them fail**

Run: `cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform && uv run pytest tests/unit/dashboard/test_device_flow.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement DeviceFlowStore and DeviceFlowManager**

Create `packages/atp-dashboard/atp/dashboard/auth/device_flow.py`:

```python
"""GitHub Device Flow (RFC 8628) implementation.

Manages the device authorization flow for CLI login:
1. Client calls initiate() → gets device_code + user_code
2. User visits github.com/login/device and enters user_code
3. Client polls poll() until user authorizes → gets GitHub user info

State is stored in-memory (does not survive server restart).
"""

import secrets
import string
import time
from dataclasses import dataclass, field

import httpx

GITHUB_DEVICE_VERIFY_URI = "https://github.com/login/device"
GITHUB_DEVICE_AUTH_URL = "https://github.com/login/oauth/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API = "https://api.github.com/user"
GITHUB_USER_EMAILS_API = "https://api.github.com/user/emails"

DEFAULT_EXPIRES_IN = 900  # 15 minutes
DEFAULT_INTERVAL = 5  # seconds


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
    github_access_token: str | None = None

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    @property
    def is_authorized(self) -> bool:
        return self.github_access_token is not None


def _generate_user_code() -> str:
    """Generate an 8-char alphanumeric user code (XXXX-XXXX display)."""
    chars = string.ascii_uppercase + string.digits
    # Remove ambiguous characters
    chars = chars.replace("O", "").replace("0", "").replace("I", "").replace("1", "")
    return "".join(secrets.choice(chars) for _ in range(8))


class DeviceFlowStore:
    """In-memory store for pending device flow authorizations."""

    def __init__(self) -> None:
        self._entries: dict[str, DeviceCodeEntry] = {}
        self._by_user_code: dict[str, str] = {}  # user_code → device_code

    def create(
        self,
        client_id: str,
        expires_in: int = DEFAULT_EXPIRES_IN,
        interval: int = DEFAULT_INTERVAL,
    ) -> DeviceCodeEntry:
        """Create a new device code entry."""
        device_code = secrets.token_urlsafe(32)
        user_code = _generate_user_code()
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

    def get_by_device_code(
        self, device_code: str
    ) -> DeviceCodeEntry | None:
        entry = self._entries.get(device_code)
        if entry is None or entry.is_expired:
            return None
        return entry

    def get_by_user_code(
        self, user_code: str
    ) -> DeviceCodeEntry | None:
        device_code = self._by_user_code.get(user_code)
        if device_code is None:
            return None
        return self.get_by_device_code(device_code)

    def mark_authorized(
        self,
        device_code: str,
        github_access_token: str,
    ) -> None:
        entry = self._entries.get(device_code)
        if entry is not None:
            entry.github_access_token = github_access_token

    def remove(self, device_code: str) -> None:
        entry = self._entries.pop(device_code, None)
        if entry is not None:
            self._by_user_code.pop(entry.user_code, None)

    def cleanup_expired(self) -> None:
        expired = [
            dc for dc, e in self._entries.items() if e.is_expired
        ]
        for dc in expired:
            self.remove(dc)


class DeviceFlowManager:
    """Manages GitHub Device Flow authorization.

    Two-phase flow:
    1. initiate() — returns device_code, user_code, verification_uri
    2. poll(device_code) — checks if user authorized, returns user info
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._store = DeviceFlowStore()

    def initiate(self) -> dict[str, str | int]:
        """Start a new device flow authorization.

        Returns:
            Dict with device_code, user_code, verification_uri,
            expires_in, interval.
        """
        self._store.cleanup_expired()
        entry = self._store.create(client_id=self._client_id)
        return {
            "device_code": entry.device_code,
            "user_code": entry.user_code,
            "verification_uri": GITHUB_DEVICE_VERIFY_URI,
            "expires_in": DEFAULT_EXPIRES_IN,
            "interval": entry.interval,
        }

    async def poll(
        self, device_code: str
    ) -> dict[str, str | int | None]:
        """Poll for authorization status.

        Args:
            device_code: The device code from initiate().

        Returns:
            GitHub user info dict if authorized.

        Raises:
            DeviceCodeNotFoundError: Unknown device code.
            DeviceCodeExpiredError: Device code expired.
            DeviceCodePendingError: User hasn't authorized yet.
        """
        # Check in-memory store first (for already-authorized)
        entry = self._store._entries.get(device_code)
        if entry is None:
            raise DeviceCodeNotFoundError(
                "Device code not found"
            )
        if entry.is_expired:
            self._store.remove(device_code)
            raise DeviceCodeExpiredError("Device code expired")

        if entry.is_authorized:
            user_info = await self._fetch_github_user(
                entry.github_access_token  # type: ignore[arg-type]
            )
            self._store.remove(device_code)
            return user_info

        # Try exchanging with GitHub
        token = await self._exchange_device_code(
            device_code=device_code,
        )
        if token is None:
            raise DeviceCodePendingError(
                "Authorization pending"
            )

        # Got token — fetch user info
        self._store.mark_authorized(device_code, token)
        user_info = await self._fetch_github_user(token)
        self._store.remove(device_code)
        return user_info

    async def _exchange_device_code(
        self, device_code: str
    ) -> str | None:
        """Exchange device code for access token with GitHub.

        Returns access token string, or None if still pending.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "device_code": device_code,
                    "grant_type": (
                        "urn:ietf:params:oauth:grant-type:device_code"
                    ),
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        error = data.get("error")
        if error == "authorization_pending":
            return None
        if error == "slow_down":
            return None
        if error == "expired_token":
            raise DeviceCodeExpiredError("Device code expired at GitHub")
        if error == "access_denied":
            raise DeviceFlowError("User denied access")
        if error:
            raise DeviceFlowError(f"GitHub error: {error}")

        return data.get("access_token")

    async def _fetch_github_user(
        self, access_token: str
    ) -> dict[str, str | int | None]:
        """Fetch GitHub user profile and primary email."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }
            # Fetch profile
            resp = await client.get(GITHUB_USER_API, headers=headers)
            resp.raise_for_status()
            profile = resp.json()

            # Fetch primary email if not in profile
            email = profile.get("email")
            if not email:
                resp = await client.get(
                    GITHUB_USER_EMAILS_API, headers=headers
                )
                resp.raise_for_status()
                emails = resp.json()
                for e in emails:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/dashboard/test_device_flow.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/auth/device_flow.py tests/unit/dashboard/test_device_flow.py
git commit -m "feat(auth): implement GitHub Device Flow logic with in-memory store"
```

---

### Task 4: Create Device Auth route endpoints

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Test: `tests/unit/dashboard/test_device_auth_routes.py`

- [ ] **Step 1: Write route-level tests**

Create `tests/unit/dashboard/test_device_auth_routes.py`:

```python
"""Tests for device auth route endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def app():
    """Create test app with device auth routes."""
    with patch(
        "atp.dashboard.v2.routes.device_auth.get_config"
    ) as mock_config:
        cfg = MagicMock()
        cfg.github_client_id = "test-client-id"
        cfg.github_client_secret = "test-client-secret"
        mock_config.return_value = cfg
        yield create_test_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c


class TestDeviceAuthInitiate:
    """Tests for POST /auth/device."""

    @pytest.mark.anyio
    async def test_initiate_success(self, client) -> None:
        """Test successful device flow initiation."""
        with patch(
            "atp.dashboard.v2.routes.device_auth.get_config"
        ) as mock_config:
            cfg = MagicMock()
            cfg.github_client_id = "test-client-id"
            cfg.github_client_secret = "test-client-secret"
            mock_config.return_value = cfg

            resp = await client.post("/api/auth/device")
            assert resp.status_code == 200
            data = resp.json()
            assert "device_code" in data
            assert "user_code" in data
            assert "verification_uri" in data
            assert data["verification_uri"] == (
                "https://github.com/login/device"
            )
            assert "expires_in" in data
            assert "interval" in data

    @pytest.mark.anyio
    async def test_initiate_not_configured(self, client) -> None:
        """Test initiation when GitHub OAuth is not configured."""
        with patch(
            "atp.dashboard.v2.routes.device_auth.get_config"
        ) as mock_config:
            cfg = MagicMock()
            cfg.github_client_id = None
            cfg.github_client_secret = None
            mock_config.return_value = cfg

            resp = await client.post("/api/auth/device")
            assert resp.status_code == 501


class TestDeviceAuthPoll:
    """Tests for POST /auth/device/poll."""

    @pytest.mark.anyio
    async def test_poll_unknown_code(self, client) -> None:
        """Test polling with unknown device code."""
        with patch(
            "atp.dashboard.v2.routes.device_auth.get_config"
        ) as mock_config:
            cfg = MagicMock()
            cfg.github_client_id = "test-client-id"
            cfg.github_client_secret = "test-client-secret"
            mock_config.return_value = cfg

            resp = await client.post(
                "/api/auth/device/poll",
                json={"device_code": "nonexistent"},
            )
            assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to see them fail**

Run: `uv run pytest tests/unit/dashboard/test_device_auth_routes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement device auth routes**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py`:

```python
"""Device Flow authentication routes (RFC 8628).

Provides endpoints for CLI login via GitHub Device Flow:
- POST /auth/device     — initiate device flow, get user_code
- POST /auth/device/poll — poll for authorization result
"""

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_user_by_email,
)
from atp.dashboard.auth.device_flow import (
    DeviceCodeExpiredError,
    DeviceCodeNotFoundError,
    DeviceCodePendingError,
    DeviceFlowError,
    DeviceFlowManager,
)
from atp.dashboard.auth.sso.oidc import (
    SSOUserInfo,
    provision_sso_user,
)
from atp.dashboard.schemas import Token
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/auth", tags=["auth"])

# Module-level manager, lazily initialized
_manager: DeviceFlowManager | None = None


def _get_manager() -> DeviceFlowManager:
    """Get or create the DeviceFlowManager singleton."""
    global _manager
    config = get_config()
    if not config.github_client_id or not config.github_client_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="GitHub OAuth is not configured. "
            "Set ATP_GITHUB_CLIENT_ID and "
            "ATP_GITHUB_CLIENT_SECRET.",
        )
    if _manager is None:
        _manager = DeviceFlowManager(
            client_id=config.github_client_id,
            client_secret=config.github_client_secret,
        )
    return _manager


class DeviceInitResponse(BaseModel):
    """Response from POST /auth/device."""

    device_code: str = Field(
        ..., description="Code for polling"
    )
    user_code: str = Field(
        ..., description="Code user enters at verification_uri"
    )
    verification_uri: str = Field(
        ..., description="URL where user enters user_code"
    )
    expires_in: int = Field(
        ..., description="Seconds until device_code expires"
    )
    interval: int = Field(
        ..., description="Minimum polling interval in seconds"
    )


class DevicePollRequest(BaseModel):
    """Request body for POST /auth/device/poll."""

    device_code: str = Field(
        ..., description="Device code from initiation"
    )


@router.post("/device", response_model=DeviceInitResponse)
async def initiate_device_flow() -> DeviceInitResponse:
    """Initiate GitHub Device Flow for CLI login.

    Returns a device_code and user_code. The user should visit
    verification_uri and enter the user_code to authorize.
    """
    manager = _get_manager()
    result = manager.initiate()
    return DeviceInitResponse(
        device_code=str(result["device_code"]),
        user_code=str(result["user_code"]),
        verification_uri=str(result["verification_uri"]),
        expires_in=int(result["expires_in"]),
        interval=int(result["interval"]),
    )


@router.post("/device/poll")
async def poll_device_flow(
    body: DevicePollRequest,
    session: DBSession,
) -> Token:
    """Poll for device flow authorization result.

    Returns an ATP access token once the user has authorized
    the device code on GitHub.

    Raises:
        HTTPException 428: Authorization still pending (retry).
        HTTPException 410: Device code expired.
        HTTPException 404: Unknown device code.
    """
    manager = _get_manager()
    try:
        github_user = await manager.poll(body.device_code)
    except DeviceCodePendingError:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="authorization_pending",
        )
    except DeviceCodeExpiredError:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="expired_token",
        )
    except DeviceCodeNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device code not found",
        )
    except DeviceFlowError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Provision or find user
    email = github_user.get("email") or ""
    login = github_user.get("login") or ""
    name = str(github_user.get("name") or login)

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub account has no verified email",
        )

    user_info = SSOUserInfo(
        sub=f"github:{github_user.get('id', '')}",
        email=email,
        email_verified=True,
        name=name,
        preferred_username=login,
    )

    user = await provision_sso_user(
        session=session,
        user_info=user_info,
    )
    await session.commit()

    # Issue ATP JWT
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    )
    return Token(access_token=access_token)
```

- [ ] **Step 4: Register route in routes/__init__.py**

In `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`, add the import and include:

```python
from atp.dashboard.v2.routes.device_auth import (
    router as device_auth_router,
)
```

Add after `router.include_router(auth_router)`:

```python
router.include_router(device_auth_router)
```

Add `"device_auth_router"` to `__all__`.

- [ ] **Step 5: Run route tests**

Run: `uv run pytest tests/unit/dashboard/test_device_auth_routes.py -v`
Expected: PASS

- [ ] **Step 6: Run full auth test suite**

Run: `uv run pytest tests/unit/dashboard/test_auth.py tests/unit/dashboard/test_device_flow.py tests/unit/dashboard/test_device_auth_routes.py -v`
Expected: All PASS

- [ ] **Step 7: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py tests/unit/dashboard/test_device_auth_routes.py
git commit -m "feat(auth): add device flow endpoints POST /auth/device and /auth/device/poll"
```

---

### Task 5: Integration verification and full test run

**Files:**
- All files from Tasks 1-4

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform && uv run pytest tests/ -v --timeout=60 -x -q`
Expected: No regressions

- [ ] **Step 2: Run ruff format + check + pyrefly on the whole project**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`
Expected: Clean

- [ ] **Step 3: Verify endpoints are registered**

Run: `cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform && uv run python -c "from atp.dashboard.v2.routes import router; routes = [r.path for r in router.routes]; print([r for r in routes if 'device' in r])"`
Expected: `['/auth/device', '/auth/device/poll']`

- [ ] **Step 4: Final commit if any formatting changes**

```bash
git add -u
git diff --cached --stat
# Only commit if there are changes
git commit -m "style: format and lint fixes for device flow"
```
