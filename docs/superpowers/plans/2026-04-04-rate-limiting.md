# Rate Limiting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HTTP rate limiting to the ATP Dashboard API to protect against brute-force and abuse.

**Architecture:** slowapi middleware with per-endpoint decorators. Key function resolves user_id from JWT for authenticated requests, falls back to client IP. Configuration via DashboardConfig env vars.

**Tech Stack:** slowapi 0.1.9+, limits library (transitive), FastAPI 0.128.0

---

### Task 1: Add slowapi dependency and config fields

**Files:**
- Modify: `packages/atp-dashboard/pyproject.toml`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py`

- [ ] **Step 1: Add slowapi to dependencies**

In `packages/atp-dashboard/pyproject.toml`, add to the `dependencies` list:

```toml
"slowapi>=0.1.9",
```

- [ ] **Step 2: Add rate limit config fields**

In `packages/atp-dashboard/atp/dashboard/v2/config.py`, add these fields to `DashboardConfig` class after `upload_max_size_mb`:

```python
# Rate limiting settings
rate_limit_enabled: bool = Field(
    default=True,
    description="Enable HTTP rate limiting",
)
rate_limit_default: str = Field(
    default="60/minute",
    description="Default rate limit for undecorated endpoints",
)
rate_limit_auth: str = Field(
    default="5/minute",
    description="Rate limit for auth endpoints (brute-force protection)",
)
rate_limit_api: str = Field(
    default="120/minute",
    description="Rate limit for benchmark API endpoints",
)
rate_limit_upload: str = Field(
    default="10/minute",
    description="Rate limit for file upload endpoints",
)
rate_limit_storage: str = Field(
    default="memory://",
    description="Rate limit storage URI (memory:// or redis://host:port)",
)
```

- [ ] **Step 3: Sync dependencies**

Run: `uv sync --group dev`
Expected: resolves slowapi, no errors

- [ ] **Step 4: Verify import works**

Run: `uv run python -c "from slowapi import Limiter; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/pyproject.toml packages/atp-dashboard/atp/dashboard/v2/config.py uv.lock
git commit -m "feat(dashboard): add slowapi dependency and rate limit config fields"
```

---

### Task 2: Create rate_limit module with key function and 429 handler

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py`
- Create: `tests/unit/dashboard/test_rate_limit.py`

- [ ] **Step 1: Write failing tests for key function and 429 handler**

Create `tests/unit/dashboard/test_rate_limit.py`:

```python
"""Tests for rate limiting module."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.rate_limit import (
    create_limiter,
    get_rate_limit_key,
    limiter,
    rate_limit_exceeded_handler,
)


class TestGetRateLimitKey:
    """Tests for rate limit key function."""

    def test_returns_user_id_when_jwt_present(self) -> None:
        """Authenticated request uses user_id as key."""
        request = MagicMock()
        request.state.user_id = "42"
        key = get_rate_limit_key(request)
        assert key == "user:42"

    def test_returns_ip_when_no_jwt(self) -> None:
        """Unauthenticated request falls back to IP."""
        request = MagicMock()
        request.state = MagicMock(spec=[])  # no user_id attr
        request.client.host = "192.168.1.1"
        key = get_rate_limit_key(request)
        assert key == "ip:192.168.1.1"

    def test_returns_ip_when_user_id_is_none(self) -> None:
        """Request with user_id=None falls back to IP."""
        request = MagicMock()
        request.state.user_id = None
        request.client.host = "10.0.0.1"
        key = get_rate_limit_key(request)
        assert key == "ip:10.0.0.1"

    def test_uses_x_forwarded_for_header(self) -> None:
        """Request behind proxy uses X-Forwarded-For."""
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.client.host = "127.0.0.1"
        request.headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18"}
        key = get_rate_limit_key(request)
        assert key == "ip:203.0.113.50"


class TestCreateLimiter:
    """Tests for limiter factory."""

    def test_creates_limiter_with_default_config(self) -> None:
        """Limiter is created with memory storage."""
        config = DashboardConfig(debug=True)
        limiter = create_limiter(config)
        assert limiter is not None

    def test_disabled_limiter(self) -> None:
        """Disabled limiter still creates instance (noop)."""
        config = DashboardConfig(debug=True, rate_limit_enabled=False)
        limiter = create_limiter(config)
        assert limiter is not None


class TestRateLimitExceededHandler:
    """Tests for 429 response handler."""

    @pytest.mark.anyio
    async def test_returns_429_with_json(self) -> None:
        """Handler returns proper 429 JSON response."""
        request = MagicMock()
        exc = RateLimitExceeded(detail="5 per 1 minute")
        response = await rate_limit_exceeded_handler(request, exc)
        assert response.status_code == 429
        import json

        body = json.loads(response.body)
        assert body["error"] == "rate_limit_exceeded"
        assert "5 per 1 minute" in body["detail"]
        assert "retry_after" in body


class TestRateLimitIntegration:
    """Integration tests with FastAPI test client."""

    def test_rate_limit_returns_429(self) -> None:
        """Exceeding rate limit returns 429."""
        from slowapi import Limiter
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address

        app = FastAPI()
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["2/minute"],
            storage_uri="memory://",
        )
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)
        app.add_exception_handler(
            RateLimitExceeded, rate_limit_exceeded_handler
        )

        @app.get("/test")
        @limiter.limit("2/minute")
        async def test_endpoint(request: MagicMock):
            return {"ok": True}

        client = TestClient(app)
        # First two requests succeed
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 200
        # Third request is rate limited
        resp = client.get("/test")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        body = resp.json()
        assert body["error"] == "rate_limit_exceeded"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/dashboard/test_rate_limit.py -v`
Expected: ImportError — `rate_limit` module doesn't exist yet

- [ ] **Step 3: Implement rate_limit module**

Create `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py`:

```python
"""HTTP rate limiting for ATP Dashboard.

Uses slowapi (built on the limits library) for per-endpoint rate limiting.
Key function resolves user_id from JWT for authenticated requests,
falls back to client IP for anonymous requests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

if TYPE_CHECKING:
    from atp.dashboard.v2.config import DashboardConfig

logger = logging.getLogger("atp.dashboard.rate_limit")

# Module-level limiter — populated by create_limiter(), imported by route files
limiter: Limiter | None = None


def get_rate_limit_key(request: Request) -> str:
    """Extract rate limit key from request.

    Uses user_id from JWT if available, otherwise client IP.
    Respects X-Forwarded-For header for proxy deployments.
    """
    # Try user_id from JWT (set by auth middleware/dependency)
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return f"user:{user_id}"

    # Fall back to IP, respecting X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # First IP in chain is the original client
        ip = forwarded.split(",")[0].strip()
    else:
        ip = get_remote_address(request)
    return f"ip:{ip}"


def create_limiter(config: DashboardConfig) -> Limiter:
    """Create a slowapi Limiter instance from config."""
    global limiter
    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=[config.rate_limit_default],
        storage_uri=config.rate_limit_storage,
        enabled=config.rate_limit_enabled,
    )
    return limiter


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Custom 429 response with JSON body and Retry-After header."""
    # Parse retry window from exception detail
    retry_after = getattr(exc, "retry_after", 60)

    response = JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": retry_after,
        },
    )
    response.headers["Retry-After"] = str(retry_after)
    return response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/dashboard/test_rate_limit.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/rate_limit.py tests/unit/dashboard/test_rate_limit.py
git commit -m "feat(dashboard): add rate limit module with key function and 429 handler"
```

---

### Task 3: Wire rate limiter into FastAPI app

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`

- [ ] **Step 1: Add rate limiter to factory.py**

In `packages/atp-dashboard/atp/dashboard/v2/factory.py`, add imports at the top:

```python
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from atp.dashboard.v2.rate_limit import create_limiter, rate_limit_exceeded_handler
```

Then after `app.state.config = config` (line 100), add:

```python
# Set up rate limiting
limiter = create_limiter(config)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `uv run python -m pytest tests/unit/dashboard/ -q --tb=short --ignore=tests/unit/dashboard/v2/test_templates.py --ignore=tests/unit/dashboard/test_ui_routes.py -x`
Expected: all pass (rate limiting defaults to enabled with 60/min — won't affect test speed)

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/factory.py
git commit -m "feat(dashboard): wire rate limiter middleware into FastAPI app"
```

---

### Task 4: Add rate limit decorators to auth endpoints

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/sso.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/saml.py`

- [ ] **Step 1: Add limiter import helper**

slowapi needs `request.app.state.limiter` to find the limiter. Each route file needs:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
```

And each decorated endpoint needs a `request: Request` parameter if it doesn't already have one.

- [ ] **Step 2: Decorate auth.py endpoints**

In `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py`:

Add import at top:
```python
from fastapi import APIRouter, Depends, HTTPException, Request, status
from atp.dashboard.v2.rate_limit import get_rate_limit_key
```

Add limiter access helper after router definition:
```python
from slowapi import Limiter

def _limiter(request: Request) -> Limiter:
    return request.app.state.limiter
```

Add `@_limiter(request).limit(...)` — actually, slowapi uses a different pattern. The limiter instance must be accessed from app state. The standard pattern is:

```python
# At module level — will be set when app starts
from atp.dashboard.v2.rate_limit import get_rate_limit_key

# On each endpoint:
@router.post("/token", response_model=Token)
async def login(
    request: Request,  # ADD this parameter
    session: DBSession,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
```

slowapi's `@limiter.limit()` decorator needs the limiter at import time. Since our limiter is created in factory.py, we use `shared_limiter` approach — create a module-level Limiter reference in rate_limit.py that factory.py populates.

Better approach: use `request.app.state.limiter.limit()` at decoration time via a helper. Actually the standard slowapi pattern is to have a module-level limiter. Let's create it in rate_limit.py and import it:

Update `rate_limit.py` — add a module-level limiter that `create_limiter` populates:

```python
# Module-level limiter instance — configured by create_limiter()
limiter: Limiter | None = None


def create_limiter(config: DashboardConfig) -> Limiter:
    """Create a slowapi Limiter instance from config."""
    global limiter
    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=[config.rate_limit_default],
        storage_uri=config.rate_limit_storage,
        enabled=config.rate_limit_enabled,
    )
    return limiter
```

Then in route files:
```python
from atp.dashboard.v2.rate_limit import limiter
```

And decorate:
```python
@router.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login(
    request: Request,
    session: DBSession,
    ...
```

- [ ] **Step 3: Decorate auth.py**

Add `Request` import and `limiter` import. Decorate `/token` and `/register` with `5/minute`:

```python
from fastapi import APIRouter, Depends, HTTPException, Request, status
from atp.dashboard.v2.rate_limit import limiter

@router.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login(
    request: Request,
    session: DBSession,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    ...

@router.post("/register", ...)
@limiter.limit("5/minute")
async def register(
    request: Request,
    session: DBSession,
    ...
```

- [ ] **Step 4: Decorate device_auth.py**

Add `Request` import and `limiter` import. Decorate `/device` and `/device/poll` with `5/minute`:

```python
from atp.dashboard.v2.rate_limit import limiter

@router.post("/device", response_model=DeviceInitResponse)
@limiter.limit("5/minute")
async def initiate_device_flow(request: Request) -> DeviceInitResponse:
    ...

@router.post("/device/poll")
@limiter.limit("5/minute")
async def poll_device_flow(request: Request, body: DevicePollRequest, session: DBSession) -> Token:
    ...
```

- [ ] **Step 5: Decorate sso.py**

Add `limiter` import. `/init` and `/callback` get `10/minute`:

```python
from atp.dashboard.v2.rate_limit import limiter

@router.post("/init", response_model=SSOInitResponse)
@limiter.limit("10/minute")
async def initiate_sso(request: Request, session: DBSession, ...) -> SSOInitResponse:
    ...

@router.get("/callback")
@limiter.limit("10/minute")
async def sso_callback(request: Request, session: DBSession, ...) -> Token:
    ...
```

Note: sso.py callback already takes `session: DBSession` but not `request: Request` — add it.

- [ ] **Step 6: Decorate saml.py**

Add `limiter` import. `/init` and `/acs` get `10/minute`:

```python
from atp.dashboard.v2.rate_limit import limiter

@router.post("/init", response_model=SAMLInitResponse)
@limiter.limit("10/minute")
async def initiate_saml(request: Request, session: DBSession, ...) -> SAMLInitResponse:
    ...

@router.post("/acs")
@limiter.limit("10/minute")
async def assertion_consumer_service(request: Request, session: DBSession, ...) -> Token:
    ...
```

Note: saml.py already has `request: Request` parameter on both endpoints.

- [ ] **Step 7: Run auth tests**

Run: `uv run python -m pytest tests/unit/dashboard/auth/ -q --tb=short`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/rate_limit.py packages/atp-dashboard/atp/dashboard/v2/routes/auth.py packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py packages/atp-dashboard/atp/dashboard/v2/routes/sso.py packages/atp-dashboard/atp/dashboard/v2/routes/saml.py
git commit -m "feat(dashboard): add rate limit decorators to auth endpoints (5-10/min)"
```

---

### Task 5: Add rate limit decorators to API and UI endpoints

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`

- [ ] **Step 1: Decorate benchmark_api.py**

Add import and decorate all endpoints with `120/minute`. The benchmark_api router has many endpoints — apply limiter at the router level or individually. Since slowapi doesn't support router-level limits, decorate each endpoint.

Key endpoints to decorate:
- `POST /v1/benchmarks` (create)
- `GET /v1/benchmarks` (list)
- `POST /v1/benchmarks/{id}/start` (start run)
- `GET /v1/runs/{id}/next-task` (poll)
- `POST /v1/runs/{id}/submit` (submit result)
- `GET /v1/runs/{id}` (status)
- `POST /v1/runs/{id}/cancel` (cancel)
- `GET /v1/benchmarks/{id}/leaderboard` (leaderboard)

```python
from atp.dashboard.v2.rate_limit import limiter

# On each endpoint:
@router.post("/benchmarks", ...)
@limiter.limit("120/minute")
async def create_benchmark(request: Request, ...):
    ...
```

Add `request: Request` parameter to any endpoint that doesn't already have it.

- [ ] **Step 2: Decorate upload.py**

```python
from atp.dashboard.v2.rate_limit import limiter

@router.post("/suite-definitions/upload", ...)
@limiter.limit("10/minute")
async def upload_suite(request: Request, ...):
    ...
```

- [ ] **Step 3: Decorate ui.py**

UI routes already have `request: Request`. Add `120/minute` to page endpoints:

```python
from atp.dashboard.v2.rate_limit import limiter

@router.get("/login", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def login_page(request: Request, ...):
    ...
```

Apply to all `@router.get` page endpoints in ui.py.

- [ ] **Step 4: Run all dashboard tests**

Run: `uv run python -m pytest tests/unit/dashboard/ -q --tb=short --ignore=tests/unit/dashboard/v2/test_templates.py --ignore=tests/unit/dashboard/test_ui_routes.py -x`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py packages/atp-dashboard/atp/dashboard/v2/routes/upload.py packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git commit -m "feat(dashboard): add rate limit decorators to API (120/min) and UI (120/min) endpoints"
```

---

### Task 6: Final integration test and cleanup

**Files:**
- Modify: `tests/unit/dashboard/test_rate_limit.py`

- [ ] **Step 1: Add integration test with real app factory**

Add to `tests/unit/dashboard/test_rate_limit.py`:

```python
class TestRateLimitWithApp:
    """Test rate limiting with the real app factory."""

    def test_app_has_limiter_in_state(self) -> None:
        """App factory sets up limiter in app.state."""
        from atp.dashboard.v2.factory import create_test_app

        app = create_test_app()
        assert hasattr(app.state, "limiter")
        assert app.state.limiter is not None

    def test_rate_limit_disabled(self) -> None:
        """Rate limiting can be disabled via config."""
        config = DashboardConfig(
            debug=True,
            rate_limit_enabled=False,
        )
        from atp.dashboard.v2.factory import create_app

        app = create_app(config=config)
        client = TestClient(app)
        # Should never get 429 even with many requests
        for _ in range(20):
            resp = client.get("/ui/login")
            assert resp.status_code != 429
```

- [ ] **Step 2: Run full test suite**

Run: `uv run python -m pytest tests/unit/dashboard/test_rate_limit.py -v`
Expected: all pass

- [ ] **Step 3: Run ruff**

Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/rate_limit.py packages/atp-dashboard/atp/dashboard/v2/factory.py packages/atp-dashboard/atp/dashboard/v2/config.py packages/atp-dashboard/atp/dashboard/v2/routes/auth.py packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py packages/atp-dashboard/atp/dashboard/v2/routes/sso.py packages/atp-dashboard/atp/dashboard/v2/routes/saml.py packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py packages/atp-dashboard/atp/dashboard/v2/routes/upload.py packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
Expected: `All checks passed!`

- [ ] **Step 4: Final commit**

```bash
git add tests/unit/dashboard/test_rate_limit.py
git commit -m "test(dashboard): add integration tests for rate limiting with app factory"
```
