# Token Self-Service & Agent Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to self-manage agents and API tokens through REST API and dashboard UI, with invite-based registration.

**Architecture:** New ORM models (`APIToken`, `Invite`) + modifications to existing `Agent` and `Participant`. New route modules (`token_api.py`, `invite_api.py`) following existing FastAPI router pattern. Auth middleware extended with prefix-based token routing (`atp_u_`/`atp_a_` → DB lookup, else JWT). Dashboard UI pages via HTMX + Pico CSS templates.

**Tech Stack:** SQLAlchemy async ORM, FastAPI, Pydantic v2, HTMX, Pico CSS, Alembic, pytest + anyio

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `packages/atp-dashboard/atp/dashboard/tokens.py` | APIToken + Invite ORM models, token generation helpers (`generate_api_token`, `hash_token`, `verify_token`) |
| `packages/atp-dashboard/atp/dashboard/v2/routes/token_api.py` | REST endpoints: POST/GET/DELETE `/api/v1/tokens` |
| `packages/atp-dashboard/atp/dashboard/v2/routes/invite_api.py` | REST endpoints: POST/GET/DELETE `/api/v1/invites` + registration mode check |
| `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py` | REST endpoints: POST/GET/PATCH/DELETE `/api/v1/agents` (ownership-aware CRUD) |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html` | My Agents list page |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agent_detail.html` | Agent detail + tokens + tournament history |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tokens.html` | My Tokens page |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/invites.html` | Admin invite management page |
| `migrations/dashboard/versions/<hash>_agent_ownership_tokens_invites.py` | Alembic migration |
| `tests/integration/dashboard/test_token_api.py` | Token CRUD + auth middleware integration tests |
| `tests/integration/dashboard/test_invite_api.py` | Invite + registration mode tests |
| `tests/integration/dashboard/test_agent_management_api.py` | Agent ownership CRUD tests |
| `tests/unit/dashboard/test_token_helpers.py` | Unit tests for token generation/hashing |

### Modified Files
| File | Changes |
|------|---------|
| `packages/atp-dashboard/atp/dashboard/models.py` | Agent: add `owner_id`, `version`, `deleted_at`; update `__table_args__` |
| `packages/atp-dashboard/atp/dashboard/tournament/models.py` | Participant: add `agent_id` FK |
| `packages/atp-dashboard/atp/dashboard/schemas.py` | Add Pydantic schemas for tokens, invites, agent ownership |
| `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add `registration_mode`, `max_agents_per_user`, `max_tokens_per_agent`, `max_user_tokens`, `default_token_days`, `max_token_days` |
| `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py` | Extend `JWTUserStateMiddleware` with API token resolution |
| `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register new routers |
| `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py` | Add invite_code validation to register endpoint |
| `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` | Add agent/token/invite UI routes |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html` | Add sidebar nav items |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html` | Add invite code field in registration form |
| `packages/atp-dashboard/atp/dashboard/database.py` | Import new models so `create_all()` picks them up |
| `tests/integration/dashboard/conftest.py` | Add `regular_user`, `regular_token` fixtures |

---

## Task 1: Config — Add New Environment Variables

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/test_config_new_fields.py`:

```python
"""Test new config fields for token self-service."""

import os

import pytest

from atp.dashboard.v2.config import DashboardConfig


class TestTokenSelfServiceConfig:
    def test_defaults(self) -> None:
        config = DashboardConfig(debug=True)
        assert config.registration_mode == "invite"
        assert config.max_agents_per_user == 10
        assert config.max_tokens_per_agent == 3
        assert config.max_user_tokens == 5
        assert config.default_token_days == 30
        assert config.max_token_days == 365

    def test_registration_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATP_REGISTRATION_MODE", "open")
        config = DashboardConfig(debug=True)
        assert config.registration_mode == "open"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_config_new_fields.py -v`
Expected: FAIL with `ValidationError` — fields don't exist yet

- [ ] **Step 3: Add fields to DashboardConfig**

In `packages/atp-dashboard/atp/dashboard/v2/config.py`, add after the `rate_limit_storage` field (around line 125):

```python
    # Token self-service settings
    registration_mode: str = Field(
        default="invite",
        description="Registration mode: 'invite' (code required) or 'open'",
    )
    max_agents_per_user: int = Field(
        default=10,
        ge=1,
        description="Maximum agents per user",
    )
    max_tokens_per_agent: int = Field(
        default=3,
        ge=1,
        description="Maximum active API tokens per agent",
    )
    max_user_tokens: int = Field(
        default=5,
        ge=1,
        description="Maximum user-level API tokens",
    )
    default_token_days: int = Field(
        default=30,
        ge=1,
        description="Default token expiry in days",
    )
    max_token_days: int = Field(
        default=365,
        ge=0,
        description="Maximum token expiry in days (0 = allow 'never')",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/dashboard/test_config_new_fields.py -v`
Expected: PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/config.py tests/unit/dashboard/test_config_new_fields.py
git commit -m "feat: add token self-service config fields"
```

---

## Task 2: ORM Models — APIToken, Invite, Agent Changes

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tokens.py`
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- Modify: `packages/atp-dashboard/atp/dashboard/database.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/test_token_helpers.py`:

```python
"""Test token generation and hashing helpers."""

from atp.dashboard.tokens import generate_api_token, hash_token


class TestTokenGeneration:
    def test_user_token_prefix(self) -> None:
        token = generate_api_token(agent_scoped=False)
        assert token.startswith("atp_u_")
        assert len(token) == 38  # "atp_u_" + 32 hex chars

    def test_agent_token_prefix(self) -> None:
        token = generate_api_token(agent_scoped=True)
        assert token.startswith("atp_a_")
        assert len(token) == 38

    def test_tokens_are_unique(self) -> None:
        t1 = generate_api_token(agent_scoped=False)
        t2 = generate_api_token(agent_scoped=False)
        assert t1 != t2

    def test_hash_is_deterministic(self) -> None:
        token = "atp_u_abcdef1234567890abcdef1234567890"
        assert hash_token(token) == hash_token(token)

    def test_hash_differs_for_different_tokens(self) -> None:
        t1 = generate_api_token(agent_scoped=False)
        t2 = generate_api_token(agent_scoped=False)
        assert hash_token(t1) != hash_token(t2)

    def test_token_prefix_extraction(self) -> None:
        token = generate_api_token(agent_scoped=False)
        prefix = token[:12]
        assert prefix.startswith("atp_u_")
        assert len(prefix) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_token_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.dashboard.tokens'`

- [ ] **Step 3: Create tokens.py with helpers and ORM models**

Create `packages/atp-dashboard/atp/dashboard/tokens.py`:

```python
"""API token and invite ORM models + token generation helpers."""

import hashlib
import secrets
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from atp.dashboard.models import DEFAULT_TENANT_ID, Base


def generate_api_token(*, agent_scoped: bool) -> str:
    """Generate a new API token with appropriate prefix.

    Args:
        agent_scoped: If True, generates an agent-scoped token (atp_a_).
            If False, generates a user-level token (atp_u_).

    Returns:
        Token string: "atp_u_<32hex>" or "atp_a_<32hex>" (38 chars total).
    """
    prefix = "atp_a_" if agent_scoped else "atp_u_"
    return prefix + secrets.token_hex(16)


def hash_token(token: str) -> str:
    """Compute SHA-256 hash of a token for storage.

    Args:
        token: Raw API token string.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def generate_invite_code() -> str:
    """Generate a new invite code.

    Returns:
        Invite code string: "atp_inv_<16hex>" (24 chars total).
    """
    return "atp_inv_" + secrets.token_hex(8)


class APIToken(Base):
    """API token for programmatic access."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default=DEFAULT_TENANT_ID
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    agent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=True
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    token_prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    scopes: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["*"])
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_api_token_hash"),
        Index("idx_api_token_user", "user_id"),
        Index("idx_api_token_hash", "token_hash"),
    )

    def __repr__(self) -> str:
        return f"APIToken(id={self.id}, name={self.name!r}, prefix={self.token_prefix!r})"


class Invite(Base):
    """Invite code for registration."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    created_by_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )

    used_by_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    max_uses: Mapped[int] = mapped_column(Integer, default=1)
    use_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_invite_code", "code"),
    )

    def __repr__(self) -> str:
        return f"Invite(id={self.id}, code={self.code[:16]}...)"
```

- [ ] **Step 4: Run token helper tests**

Run: `uv run pytest tests/unit/dashboard/test_token_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Add new fields to Agent model**

In `packages/atp-dashboard/atp/dashboard/models.py`, modify the `Agent` class:

```python
class Agent(Base):
    """Agent configuration stored in the database."""

    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default=DEFAULT_TENANT_ID, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Ownership fields (Scope #2)
    owner_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="latest"
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    suite_executions: Mapped[list["SuiteExecution"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )

    # Indexes — owner_id + name + version unique within tenant
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "owner_id", "name", "version",
            name="uq_agent_tenant_owner_name_version",
        ),
        Index("idx_agent_name", "name"),
        Index("idx_agent_tenant", "tenant_id"),
        Index("idx_agent_owner", "owner_id"),
    )

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, name={self.name!r}, type={self.agent_type!r})"
```

- [ ] **Step 6: Add agent_id to Participant**

In `packages/atp-dashboard/atp/dashboard/tournament/models.py`, add to the `Participant` class after the `released_at` field:

```python
    # Scope #2 — link to Agent record (nullable for old participants)
    agent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=True
    )
```

- [ ] **Step 7: Import tokens module in database.py**

In `packages/atp-dashboard/atp/dashboard/database.py`, add import so `create_all()` sees the new tables. Add near the top imports:

```python
import atp.dashboard.tokens as _tokens_models  # noqa: F401  — register ORM models
```

- [ ] **Step 8: Run full test suite to check for breakage**

Run: `uv run pytest tests/ -v -x --timeout=60 -m "not slow"`
Expected: All existing tests pass (new nullable columns don't break anything)

- [ ] **Step 9: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 10: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tokens.py \
       packages/atp-dashboard/atp/dashboard/models.py \
       packages/atp-dashboard/atp/dashboard/tournament/models.py \
       packages/atp-dashboard/atp/dashboard/database.py \
       tests/unit/dashboard/test_token_helpers.py
git commit -m "feat: add APIToken, Invite models + Agent ownership fields"
```

---

## Task 3: Pydantic Schemas for Tokens, Invites, Agent Ownership

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/schemas.py`

- [ ] **Step 1: Add schemas to schemas.py**

Append to `packages/atp-dashboard/atp/dashboard/schemas.py`:

```python
# --- Token Self-Service Schemas (Scope #2) ---


class APITokenCreate(BaseModel):
    """Request body for creating an API token."""

    name: str = Field(..., min_length=1, max_length=100)
    agent_id: int | None = None
    expires_in_days: int | None = 30  # None = never

    model_config = {"json_schema_extra": {"examples": [{"name": "ci-runner", "expires_in_days": 90}]}}


class APITokenResponse(BaseModel):
    """API token in list responses (no secret)."""

    id: int
    name: str
    token_prefix: str
    agent_id: int | None
    scopes: list[str]
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class APITokenCreated(APITokenResponse):
    """Response when a token is first created (includes the raw token ONCE)."""

    token: str


class InviteCreate(BaseModel):
    """Request body for creating an invite code."""

    expires_in_days: int | None = 7  # None = never


class InviteResponse(BaseModel):
    """Invite code in list responses."""

    id: int
    code: str
    created_by_id: int
    used_by_id: int | None
    used_at: datetime | None
    expires_at: datetime | None
    max_uses: int
    use_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentOwnerCreate(BaseModel):
    """Request body for creating an owned agent."""

    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(default="latest", min_length=1, max_length=50)
    agent_type: str = Field(..., min_length=1, max_length=50)
    config: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class AgentOwnerUpdate(BaseModel):
    """Request body for updating an owned agent."""

    version: str | None = Field(default=None, min_length=1, max_length=50)
    config: dict[str, Any] | None = None
    description: str | None = None


class AgentOwnerResponse(BaseModel):
    """Agent in ownership API responses."""

    id: int
    name: str
    version: str
    agent_type: str
    config: dict[str, Any]
    description: str | None
    owner_id: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
```

Note: `datetime` and `Any` imports should already be available in schemas.py from existing code. If not, add `from datetime import datetime` and `from typing import Any` at the top.

- [ ] **Step 2: Run ruff + pyrefly to verify**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`
Expected: Clean

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/schemas.py
git commit -m "feat: add Pydantic schemas for tokens, invites, agent ownership"
```

---

## Task 4: Auth Middleware — API Token Resolution

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/test_api_token_auth.py`:

```python
"""Test API token authentication through middleware."""

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User
from atp.dashboard.tokens import APIToken, generate_api_token, hash_token
from atp.dashboard.v2.factory import create_app
from atp.dashboard.v2.config import DashboardConfig, get_config


@pytest.fixture
async def app_with_db():
    """Create app with in-memory DB and seed a user + token."""
    import os
    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "false"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    # Seed user and API token
    async with db.session() as session:
        user = User(
            username="testuser",
            email="test@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.flush()

        raw_token = generate_api_token(agent_scoped=False)
        api_token = APIToken(
            user_id=user.id,
            name="test-token",
            token_prefix=raw_token[:12],
            token_hash=hash_token(raw_token),
        )
        session.add(api_token)
        await session.commit()

    yield app, raw_token, db
    await db.close()
    set_database(None)  # type: ignore
    get_config.cache_clear()


class TestAPITokenMiddleware:
    @pytest.mark.anyio
    async def test_api_token_sets_user_id(self, app_with_db) -> None:
        app, raw_token, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {raw_token}"},
            )
            assert resp.status_code == 200
            assert resp.json()["username"] == "testuser"

    @pytest.mark.anyio
    async def test_revoked_token_rejected(self, app_with_db) -> None:
        app, raw_token, db = app_with_db
        # Revoke the token
        from datetime import datetime
        async with db.session() as session:
            from sqlalchemy import update
            await session.execute(
                update(APIToken)
                .where(APIToken.token_hash == hash_token(raw_token))
                .values(revoked_at=datetime.now())
            )
            await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {raw_token}"},
            )
            assert resp.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/dashboard/test_api_token_auth.py -v`
Expected: FAIL — middleware doesn't handle `atp_u_` prefix yet

- [ ] **Step 3: Extend JWTUserStateMiddleware**

In `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py`, modify the `JWTUserStateMiddleware.dispatch` method to add API token resolution before JWT decode. The modified class:

```python
class JWTUserStateMiddleware(BaseHTTPMiddleware):
    """Best-effort middleware that extracts user_id from Bearer token or
    cookie and sets request.state.user_id.

    Handles both JWT session tokens and API tokens (atp_u_/atp_a_ prefix).
    Runs BEFORE SlowAPIMiddleware so that the rate-limit key function can
    see the authenticated identity.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request.state.user_id = None
        request.state.agent_id = None
        request.state.token_type = None

        token = self._extract_token(request)
        if token:
            if token.startswith(("atp_u_", "atp_a_")):
                await self._resolve_api_token(request, token)
            else:
                self._resolve_jwt(request, token)

        return await call_next(request)

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return request.cookies.get("atp_token")

    @staticmethod
    def _resolve_jwt(request: Request, token: str) -> None:
        try:
            from atp.dashboard.auth import ALGORITHM, SECRET_KEY
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user_id = payload.get("user_id")
            request.state.token_type = "session"
        except InvalidTokenError:
            pass

    @staticmethod
    async def _resolve_api_token(request: Request, token: str) -> None:
        from atp.dashboard.tokens import APIToken, hash_token
        from atp.dashboard.database import get_database
        from datetime import datetime
        from sqlalchemy import select, update

        token_hash = hash_token(token)
        try:
            db = get_database()
        except Exception:
            return

        async with db.session() as session:
            result = await session.execute(
                select(APIToken).where(
                    APIToken.token_hash == token_hash,
                    APIToken.revoked_at.is_(None),
                )
            )
            api_token = result.scalar_one_or_none()
            if api_token is None:
                return

            # Check expiry
            if api_token.expires_at and api_token.expires_at < datetime.now():
                return

            request.state.user_id = api_token.user_id
            request.state.agent_id = api_token.agent_id
            request.state.token_type = "api"

            # Atomic last_used_at update (debounced: max once per 60s)
            await session.execute(
                update(APIToken)
                .where(
                    APIToken.id == api_token.id,
                    (
                        APIToken.last_used_at.is_(None)
                        | (APIToken.last_used_at < datetime.now())
                    ),
                )
                .values(last_used_at=datetime.now())
            )
            await session.commit()
```

Keep the existing `import jwt` and `from jwt.exceptions import InvalidTokenError` at the top of the file. Add `from starlette.requests import Request` and `from starlette.responses import Response` if not already imported.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_api_token_auth.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v -x --timeout=60 -m "not slow"`
Expected: All pass

- [ ] **Step 6: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/rate_limit.py \
       tests/integration/dashboard/test_api_token_auth.py
git commit -m "feat: extend middleware to resolve API tokens (atp_u_/atp_a_ prefix)"
```

---

## Task 5: Token Management API

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/token_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/test_token_api.py`:

```python
"""Test token management API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User
from atp.dashboard.tokens import APIToken, hash_token
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.factory import create_app, DashboardConfig


@pytest.fixture
async def setup():
    """Create app with test DB, user, and agent."""
    import os
    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "false"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    async with db.session() as session:
        user = User(
            username="alice",
            email="alice@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.flush()
        user_id = user.id

        agent = Agent(
            name="tit-for-tat",
            agent_type="mcp",
            owner_id=user.id,
            version="v1",
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)

    jwt_token = create_access_token(data={"sub": "alice", "user_id": user_id})
    headers = {"Authorization": f"Bearer {jwt_token}"}

    yield app, headers, user_id, agent.id, db
    await db.close()
    set_database(None)  # type: ignore
    get_config.cache_clear()


class TestTokenAPI:
    @pytest.mark.anyio
    async def test_create_user_token(self, setup) -> None:
        app, headers, user_id, agent_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/tokens",
                json={"name": "my-token", "expires_in_days": 30},
                headers=headers,
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["name"] == "my-token"
            assert data["token"].startswith("atp_u_")
            assert len(data["token"]) == 38
            assert data["agent_id"] is None

    @pytest.mark.anyio
    async def test_create_agent_token(self, setup) -> None:
        app, headers, user_id, agent_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/tokens",
                json={"name": "bot-token", "agent_id": agent_id, "expires_in_days": None},
                headers=headers,
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["token"].startswith("atp_a_")
            assert data["agent_id"] == agent_id
            assert data["expires_at"] is None

    @pytest.mark.anyio
    async def test_list_tokens(self, setup) -> None:
        app, headers, user_id, agent_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a token first
            await client.post(
                "/api/v1/tokens",
                json={"name": "list-test"},
                headers=headers,
            )
            resp = await client.get("/api/v1/tokens", headers=headers)
            assert resp.status_code == 200
            tokens = resp.json()
            assert len(tokens) >= 1
            # No raw token in list response
            assert "token" not in tokens[0]

    @pytest.mark.anyio
    async def test_revoke_token(self, setup) -> None:
        app, headers, user_id, agent_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/v1/tokens",
                json={"name": "revoke-test"},
                headers=headers,
            )
            token_id = create_resp.json()["id"]
            resp = await client.delete(f"/api/v1/tokens/{token_id}", headers=headers)
            assert resp.status_code == 200
            assert resp.json()["revoked_at"] is not None

    @pytest.mark.anyio
    async def test_user_token_limit(self, setup) -> None:
        app, headers, user_id, agent_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create max_user_tokens (default 5)
            for i in range(5):
                resp = await client.post(
                    "/api/v1/tokens",
                    json={"name": f"tok-{i}"},
                    headers=headers,
                )
                assert resp.status_code == 201
            # 6th should fail
            resp = await client.post(
                "/api/v1/tokens",
                json={"name": "tok-overflow"},
                headers=headers,
            )
            assert resp.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/dashboard/test_token_api.py -v`
Expected: FAIL — route doesn't exist yet

- [ ] **Step 3: Create token_api.py**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/token_api.py`:

```python
"""Token management API endpoints."""

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select

from atp.dashboard.models import Agent
from atp.dashboard.schemas import APITokenCreate, APITokenCreated, APITokenResponse
from atp.dashboard.tokens import APIToken, generate_api_token, hash_token
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession, RequiredUser
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(prefix="/v1/tokens", tags=["tokens"])


@router.post("", response_model=APITokenCreated, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_token(
    request: Request,
    session: DBSession,
    user: RequiredUser,
    body: APITokenCreate,
) -> APITokenCreated:
    """Create a new API token."""
    config = get_config()

    if body.agent_id is not None:
        # Agent-scoped token: verify ownership
        agent = await session.get(Agent, body.agent_id)
        if agent is None or agent.owner_id != user.id or agent.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't own this agent",
            )
        # Check per-agent token limit
        count_result = await session.execute(
            select(func.count(APIToken.id)).where(
                APIToken.agent_id == body.agent_id,
                APIToken.revoked_at.is_(None),
            )
        )
        if count_result.scalar_one() >= config.max_tokens_per_agent:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Token limit reached for this agent (max {config.max_tokens_per_agent})",
            )
    else:
        # User-level token: check limit
        count_result = await session.execute(
            select(func.count(APIToken.id)).where(
                APIToken.user_id == user.id,
                APIToken.agent_id.is_(None),
                APIToken.revoked_at.is_(None),
            )
        )
        if count_result.scalar_one() >= config.max_user_tokens:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Token limit reached (max {config.max_user_tokens})",
            )

    # Validate expiry
    expires_at = None
    if body.expires_in_days is not None:
        if config.max_token_days > 0 and body.expires_in_days > config.max_token_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Token expiry exceeds maximum ({config.max_token_days} days)",
            )
        expires_at = datetime.now() + timedelta(days=body.expires_in_days)
    elif config.max_token_days > 0:
        # "never" requested but max_token_days is set (non-zero means "never" disallowed)
        pass  # 0 means allow "never", positive means max — but None is "never"

    raw_token = generate_api_token(agent_scoped=body.agent_id is not None)
    db_token = APIToken(
        user_id=user.id,
        agent_id=body.agent_id,
        name=body.name,
        token_prefix=raw_token[:12],
        token_hash=hash_token(raw_token),
        expires_at=expires_at,
    )
    session.add(db_token)
    await session.flush()
    await session.refresh(db_token)

    return APITokenCreated(
        id=db_token.id,
        name=db_token.name,
        token_prefix=db_token.token_prefix,
        agent_id=db_token.agent_id,
        scopes=db_token.scopes,
        expires_at=db_token.expires_at,
        last_used_at=db_token.last_used_at,
        revoked_at=db_token.revoked_at,
        created_at=db_token.created_at,
        token=raw_token,
    )


@router.get("", response_model=list[APITokenResponse])
async def list_tokens(
    session: DBSession,
    user: RequiredUser,
) -> list[APITokenResponse]:
    """List all tokens for the current user."""
    result = await session.execute(
        select(APIToken)
        .where(APIToken.user_id == user.id)
        .order_by(APIToken.created_at.desc())
    )
    tokens = result.scalars().all()
    return [APITokenResponse.model_validate(t) for t in tokens]


@router.delete("/{token_id}", response_model=APITokenResponse)
async def revoke_token(
    session: DBSession,
    user: RequiredUser,
    token_id: int,
) -> APITokenResponse:
    """Revoke an API token."""
    token = await session.get(APIToken, token_id)
    if token is None or (token.user_id != user.id and not user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found",
        )
    if token.revoked_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Token already revoked",
        )
    token.revoked_at = datetime.now()
    await session.flush()
    await session.refresh(token)
    return APITokenResponse.model_validate(token)
```

- [ ] **Step 4: Register router**

In `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`, add:

```python
from atp.dashboard.v2.routes.token_api import router as token_api_router
```

And include it:

```python
router.include_router(token_api_router)
```

Add `"token_api_router"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_token_api.py -v`
Expected: PASS

- [ ] **Step 6: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/token_api.py \
       packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py \
       tests/integration/dashboard/test_token_api.py
git commit -m "feat: token management API (create, list, revoke)"
```

---

## Task 6: Agent Management API (Ownership-Aware)

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/test_agent_management_api.py`:

```python
"""Test agent ownership management API."""

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.factory import create_app, DashboardConfig


@pytest.fixture
async def setup():
    """Create app with test DB and user."""
    import os
    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "false"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    async with db.session() as session:
        user = User(
            username="alice",
            email="alice@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    jwt_token = create_access_token(data={"sub": "alice", "user_id": user.id})
    headers = {"Authorization": f"Bearer {jwt_token}"}

    yield app, headers, user.id, db
    await db.close()
    set_database(None)  # type: ignore
    get_config.cache_clear()


class TestAgentManagementAPI:
    @pytest.mark.anyio
    async def test_create_agent(self, setup) -> None:
        app, headers, user_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/agents",
                json={
                    "name": "tit-for-tat",
                    "version": "v1",
                    "agent_type": "mcp",
                    "description": "Classic TFT",
                },
                headers=headers,
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["name"] == "tit-for-tat"
            assert data["version"] == "v1"
            assert data["owner_id"] == user_id

    @pytest.mark.anyio
    async def test_list_my_agents(self, setup) -> None:
        app, headers, user_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/agents",
                json={"name": "bot-a", "agent_type": "http"},
                headers=headers,
            )
            resp = await client.get("/api/v1/agents", headers=headers)
            assert resp.status_code == 200
            assert len(resp.json()) >= 1

    @pytest.mark.anyio
    async def test_duplicate_name_version_rejected(self, setup) -> None:
        app, headers, user_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/agents",
                json={"name": "dup", "version": "v1", "agent_type": "http"},
                headers=headers,
            )
            resp = await client.post(
                "/api/v1/agents",
                json={"name": "dup", "version": "v1", "agent_type": "http"},
                headers=headers,
            )
            assert resp.status_code == 409

    @pytest.mark.anyio
    async def test_soft_delete(self, setup) -> None:
        app, headers, user_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/v1/agents",
                json={"name": "to-delete", "agent_type": "http"},
                headers=headers,
            )
            agent_id = create_resp.json()["id"]
            resp = await client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
            assert resp.status_code == 200

            # Should not appear in list
            list_resp = await client.get("/api/v1/agents", headers=headers)
            ids = [a["id"] for a in list_resp.json()]
            assert agent_id not in ids

    @pytest.mark.anyio
    async def test_agent_limit(self, setup) -> None:
        app, headers, user_id, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(10):
                resp = await client.post(
                    "/api/v1/agents",
                    json={"name": f"agent-{i}", "agent_type": "http"},
                    headers=headers,
                )
                assert resp.status_code == 201
            # 11th should fail
            resp = await client.post(
                "/api/v1/agents",
                json={"name": "agent-overflow", "agent_type": "http"},
                headers=headers,
            )
            assert resp.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/dashboard/test_agent_management_api.py -v`
Expected: FAIL — route doesn't exist

- [ ] **Step 3: Create agent_management_api.py**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`:

```python
"""Agent ownership management API endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select, update

from atp.dashboard.models import Agent
from atp.dashboard.schemas import AgentOwnerCreate, AgentOwnerResponse, AgentOwnerUpdate
from atp.dashboard.tokens import APIToken
from atp.dashboard.tournament.models import Participant, Tournament, TournamentStatus
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession, RequiredUser
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(prefix="/v1/agents", tags=["agent-management"])


@router.post("", response_model=AgentOwnerResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_agent(
    request: Request,
    session: DBSession,
    user: RequiredUser,
    body: AgentOwnerCreate,
) -> AgentOwnerResponse:
    """Create a new agent owned by the current user."""
    config = get_config()

    # Check agent limit
    count_result = await session.execute(
        select(func.count(Agent.id)).where(
            Agent.owner_id == user.id,
            Agent.deleted_at.is_(None),
        )
    )
    if count_result.scalar_one() >= config.max_agents_per_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent limit reached (max {config.max_agents_per_user})",
        )

    # Check uniqueness
    existing = await session.execute(
        select(Agent.id).where(
            Agent.owner_id == user.id,
            Agent.name == body.name,
            Agent.version == body.version,
            Agent.deleted_at.is_(None),
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{body.name}' version '{body.version}' already exists",
        )

    agent = Agent(
        name=body.name,
        version=body.version,
        agent_type=body.agent_type,
        config=body.config,
        description=body.description,
        owner_id=user.id,
    )
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    return AgentOwnerResponse.model_validate(agent)


@router.get("", response_model=list[AgentOwnerResponse])
async def list_my_agents(
    session: DBSession,
    user: RequiredUser,
) -> list[AgentOwnerResponse]:
    """List agents owned by the current user."""
    result = await session.execute(
        select(Agent)
        .where(Agent.owner_id == user.id, Agent.deleted_at.is_(None))
        .order_by(Agent.created_at.desc())
    )
    return [AgentOwnerResponse.model_validate(a) for a in result.scalars().all()]


@router.get("/{agent_id}", response_model=AgentOwnerResponse)
async def get_agent(
    session: DBSession,
    user: RequiredUser,
    agent_id: int,
) -> AgentOwnerResponse:
    """Get agent details (owner or admin only)."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this agent")
    return AgentOwnerResponse.model_validate(agent)


@router.patch("/{agent_id}", response_model=AgentOwnerResponse)
async def update_agent(
    session: DBSession,
    user: RequiredUser,
    agent_id: int,
    body: AgentOwnerUpdate,
) -> AgentOwnerResponse:
    """Update an agent (owner or admin only)."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this agent")

    if body.version is not None:
        # Check uniqueness of new version
        existing = await session.execute(
            select(Agent.id).where(
                Agent.owner_id == agent.owner_id,
                Agent.name == agent.name,
                Agent.version == body.version,
                Agent.id != agent.id,
                Agent.deleted_at.is_(None),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{agent.name}' version '{body.version}' already exists",
            )
        agent.version = body.version

    if body.config is not None:
        agent.config = body.config
    if body.description is not None:
        agent.description = body.description

    await session.flush()
    await session.refresh(agent)
    return AgentOwnerResponse.model_validate(agent)


@router.delete("/{agent_id}", response_model=AgentOwnerResponse)
async def delete_agent(
    session: DBSession,
    user: RequiredUser,
    agent_id: int,
) -> AgentOwnerResponse:
    """Soft-delete an agent and revoke all its tokens."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this agent")

    # Block if agent is in an active tournament
    active_participation = await session.execute(
        select(Participant.id)
        .join(Tournament, Participant.tournament_id == Tournament.id)
        .where(
            Participant.agent_id == agent_id,
            Participant.released_at.is_(None),
            Tournament.status.in_([TournamentStatus.PENDING, TournamentStatus.ACTIVE]),
        )
    )
    if active_participation.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent is in an active tournament",
        )

    # Soft delete
    agent.deleted_at = datetime.now()

    # Revoke all agent tokens
    await session.execute(
        update(APIToken)
        .where(APIToken.agent_id == agent_id, APIToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now())
    )

    await session.flush()
    await session.refresh(agent)
    return AgentOwnerResponse.model_validate(agent)
```

- [ ] **Step 4: Register router**

In `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`, add:

```python
from atp.dashboard.v2.routes.agent_management_api import router as agent_management_api_router
```

And include it:

```python
router.include_router(agent_management_api_router)
```

Add `"agent_management_api_router"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_agent_management_api.py -v`
Expected: PASS

- [ ] **Step 6: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py \
       packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py \
       tests/integration/dashboard/test_agent_management_api.py
git commit -m "feat: agent management API (create, list, update, soft-delete)"
```

---

## Task 7: Invite API + Registration Mode

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/invite_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Modify: `packages/atp-dashboard/atp/dashboard/schemas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/test_invite_api.py`:

```python
"""Test invite management and invite-gated registration."""

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.rbac.models import Role, RolePermission, UserRole
from atp.dashboard.tenancy.models import Tenant
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.factory import create_app, DashboardConfig


@pytest.fixture
async def setup():
    """App with admin user, invite mode enabled."""
    import os
    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "false"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    os.environ["ATP_REGISTRATION_MODE"] = "invite"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lambda c: Tenant.__table__.create(c, checkfirst=True))
        await conn.run_sync(lambda c: Role.__table__.create(c, checkfirst=True))
        await conn.run_sync(lambda c: RolePermission.__table__.create(c, checkfirst=True))
        await conn.run_sync(lambda c: UserRole.__table__.create(c, checkfirst=True))
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
        registration_mode="invite",
    )
    app = create_app(config=config)

    async with db.session() as session:
        admin = User(
            username="admin",
            email="admin@test.com",
            hashed_password=get_password_hash("pass"),
            is_admin=True,
            is_active=True,
        )
        session.add(admin)
        await session.commit()
        await session.refresh(admin)

    admin_jwt = create_access_token(data={"sub": "admin", "user_id": admin.id})
    admin_headers = {"Authorization": f"Bearer {admin_jwt}"}

    yield app, admin_headers, db
    await db.close()
    set_database(None)  # type: ignore
    get_config.cache_clear()
    os.environ.pop("ATP_REGISTRATION_MODE", None)


class TestInviteAPI:
    @pytest.mark.anyio
    async def test_create_invite(self, setup) -> None:
        app, headers, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/invites", json={}, headers=headers)
            assert resp.status_code == 201
            data = resp.json()
            assert data["code"].startswith("atp_inv_")

    @pytest.mark.anyio
    async def test_register_with_invite(self, setup) -> None:
        app, headers, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invite_resp = await client.post("/api/v1/invites", json={}, headers=headers)
            code = invite_resp.json()["code"]

            reg_resp = await client.post(
                "/api/auth/register",
                json={
                    "username": "newuser",
                    "email": "new@test.com",
                    "password": "password123",
                    "invite_code": code,
                },
            )
            assert reg_resp.status_code == 201

    @pytest.mark.anyio
    async def test_register_without_invite_fails(self, setup) -> None:
        app, headers, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/register",
                json={
                    "username": "noinvite",
                    "email": "noinvite@test.com",
                    "password": "password123",
                },
            )
            assert resp.status_code == 400
            assert "invite" in resp.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_register_with_invalid_invite(self, setup) -> None:
        app, headers, db = setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/register",
                json={
                    "username": "badinvite",
                    "email": "bad@test.com",
                    "password": "password123",
                    "invite_code": "atp_inv_doesnotexist",
                },
            )
            assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/dashboard/test_invite_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add invite_code to UserCreate schema**

In `packages/atp-dashboard/atp/dashboard/schemas.py`, modify `UserCreate`:

```python
class UserCreate(BaseModel):
    """User registration request."""

    username: str
    email: str
    password: str
    invite_code: str | None = None
```

- [ ] **Step 4: Create invite_api.py**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/invite_api.py`:

```python
"""Invite management API endpoints (admin only)."""

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select

from atp.dashboard.schemas import InviteCreate, InviteResponse
from atp.dashboard.tokens import Invite, generate_invite_code
from atp.dashboard.v2.dependencies import AdminUser, DBSession
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(prefix="/v1/invites", tags=["invites"])


@router.post("", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_invite(
    request: Request,
    session: DBSession,
    admin: AdminUser,
    body: InviteCreate = InviteCreate(),
) -> InviteResponse:
    """Create a new invite code (admin only)."""
    expires_at = None
    if body.expires_in_days is not None:
        expires_at = datetime.now() + timedelta(days=body.expires_in_days)

    invite = Invite(
        code=generate_invite_code(),
        created_by_id=admin.id,
        expires_at=expires_at,
    )
    session.add(invite)
    await session.flush()
    await session.refresh(invite)
    return InviteResponse.model_validate(invite)


@router.get("", response_model=list[InviteResponse])
async def list_invites(
    session: DBSession,
    admin: AdminUser,
) -> list[InviteResponse]:
    """List all invite codes (admin only)."""
    result = await session.execute(
        select(Invite).order_by(Invite.created_at.desc())
    )
    return [InviteResponse.model_validate(i) for i in result.scalars().all()]


@router.delete("/{invite_id}", response_model=InviteResponse)
async def deactivate_invite(
    session: DBSession,
    admin: AdminUser,
    invite_id: int,
) -> InviteResponse:
    """Deactivate an invite code (admin only)."""
    invite = await session.get(Invite, invite_id)
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    # Set max_uses = use_count to effectively deactivate
    invite.max_uses = invite.use_count
    await session.flush()
    await session.refresh(invite)
    return InviteResponse.model_validate(invite)
```

- [ ] **Step 5: Add invite validation to register endpoint**

In `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py`, modify the `register` function to validate invite codes. Add these imports at the top:

```python
from atp.dashboard.tokens import Invite
from atp.dashboard.v2.config import get_config
```

Replace the `register` function body (keep the decorator and signature):

```python
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("5/minute")
async def register(
    request: Request, session: DBSession, user_data: UserCreate
) -> UserResponse:
    """Register a new user."""
    try:
        config = get_config()

        # Invite validation
        if config.registration_mode == "invite":
            if not user_data.invite_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invite code required",
                )
            result = await session.execute(
                select(Invite).where(Invite.code == user_data.invite_code)
            )
            invite = result.scalar_one_or_none()
            if invite is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired invite code",
                )
            if invite.use_count >= invite.max_uses:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired invite code",
                )
            if invite.expires_at and invite.expires_at < datetime.now():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired invite code",
                )

        # First user becomes admin automatically
        result = await session.execute(select(func.count(User.id)))
        user_count = result.scalar_one()
        is_first_user = user_count == 0

        user = await create_user(
            session,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            is_admin=is_first_user,
        )

        # Assign default role: admin for first user, developer for invited users, viewer for open
        if is_first_user:
            default_role_name = "admin"
        elif config.registration_mode == "invite":
            default_role_name = "developer"
        else:
            default_role_name = "viewer"

        role_result = await session.execute(
            select(Role).where(Role.name == default_role_name)
        )
        role = role_result.scalar_one_or_none()
        if role is not None:
            session.add(UserRole(user_id=user.id, role_id=role.id))

        # Mark invite as used
        if config.registration_mode == "invite" and user_data.invite_code:
            invite_result = await session.execute(
                select(Invite).where(Invite.code == user_data.invite_code)
            )
            invite = invite_result.scalar_one_or_none()
            if invite:
                invite.use_count += 1
                invite.used_by_id = user.id
                invite.used_at = datetime.now()

        await session.commit()
        return UserResponse.model_validate(user)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
```

Add `from datetime import datetime` to the imports at the top of `auth.py` if not already present.

- [ ] **Step 6: Register invite router**

In `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`, add:

```python
from atp.dashboard.v2.routes.invite_api import router as invite_api_router
```

And include it:

```python
router.include_router(invite_api_router)
```

Add `"invite_api_router"` to `__all__`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_invite_api.py -v`
Expected: PASS

- [ ] **Step 8: Run full suite to check for breakage**

Run: `uv run pytest tests/ -v -x --timeout=60 -m "not slow"`
Expected: All pass

- [ ] **Step 9: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 10: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/invite_api.py \
       packages/atp-dashboard/atp/dashboard/v2/routes/auth.py \
       packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py \
       packages/atp-dashboard/atp/dashboard/schemas.py \
       tests/integration/dashboard/test_invite_api.py
git commit -m "feat: invite management API + invite-gated registration"
```

---

## Task 8: Dashboard UI — Sidebar + Agents Page

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agent_detail.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`

- [ ] **Step 1: Add sidebar navigation items**

In `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`, add after the Analytics `<li>` (line 25) and before `</ul>`:

```html
                {% if user %}
                <li><a href="/ui/agents" class="{% if active_page == 'agents' %}active{% endif %}">My Agents</a></li>
                <li><a href="/ui/tokens" class="{% if active_page == 'tokens' %}active{% endif %}">My Tokens</a></li>
                {% endif %}
                {% if user and user.is_admin %}
                <li><a href="/ui/invites" class="{% if active_page == 'invites' %}active{% endif %}">Invites</a></li>
                {% endif %}
```

- [ ] **Step 2: Create agents.html template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}My Agents — ATP Platform{% endblock %}

{% block content %}
<h1>My Agents</h1>

<details id="new-agent-form">
    <summary role="button" class="outline">New Agent</summary>
    <form hx-post="/api/v1/agents" hx-target="#agents-table" hx-swap="innerHTML"
          hx-headers='{"Authorization": "Bearer {{ token }}"}'>
        <div class="grid">
            <label>Name <input type="text" name="name" required></label>
            <label>Version <input type="text" name="version" value="latest"></label>
            <label>Type
                <select name="agent_type">
                    <option value="http">HTTP</option>
                    <option value="mcp">MCP</option>
                    <option value="cli">CLI</option>
                    <option value="container">Container</option>
                </select>
            </label>
        </div>
        <label>Description <input type="text" name="description"></label>
        <button type="submit">Create Agent</button>
    </form>
</details>

<div id="agents-table">
{% if agents %}
<figure>
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Version</th>
            <th>Type</th>
            <th>Active Tokens</th>
            <th>Created</th>
            <th>Actions</th>
        </tr>
    </thead>
    <tbody>
        {% for agent in agents %}
        <tr>
            <td><a href="/ui/agents/{{ agent.id }}">{{ agent.name }}</a></td>
            <td><code>{{ agent.version }}</code></td>
            <td>{{ agent.agent_type }}</td>
            <td>{{ agent.token_count }}</td>
            <td>{{ agent.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
            <td>
                <a href="/ui/agents/{{ agent.id }}">View</a>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
</figure>
{% else %}
<p>No agents yet. Create your first agent above.</p>
{% endif %}
</div>
{% endblock %}
```

- [ ] **Step 3: Create agent_detail.html template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agent_detail.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}{{ agent.name }} — ATP Platform{% endblock %}

{% block content %}
<h1>{{ agent.name }} <small><code>{{ agent.version }}</code></small></h1>

<article>
    <header>Agent Info</header>
    <div class="grid">
        <div><strong>Type:</strong> {{ agent.agent_type }}</div>
        <div><strong>Created:</strong> {{ agent.created_at.strftime('%Y-%m-%d %H:%M') }}</div>
    </div>
    {% if agent.description %}
    <p>{{ agent.description }}</p>
    {% endif %}
    {% if agent.config %}
    <details>
        <summary>Config (JSON)</summary>
        <pre><code>{{ agent.config | tojson(indent=2) }}</code></pre>
    </details>
    {% endif %}
</article>

<article>
    <header>API Tokens</header>
    {% if tokens %}
    <table>
        <thead>
            <tr><th>Name</th><th>Prefix</th><th>Expires</th><th>Last Used</th><th>Status</th><th>Actions</th></tr>
        </thead>
        <tbody>
            {% for t in tokens %}
            <tr>
                <td>{{ t.name }}</td>
                <td><code>{{ t.token_prefix }}...</code></td>
                <td>{{ t.expires_at.strftime('%Y-%m-%d') if t.expires_at else 'never' }}</td>
                <td>{{ t.last_used_at.strftime('%Y-%m-%d %H:%M') if t.last_used_at else '—' }}</td>
                <td>{% if t.revoked_at %}revoked{% elif t.expires_at and t.expires_at < now %}expired{% else %}active{% endif %}</td>
                <td>
                    {% if not t.revoked_at %}
                    <a href="#" hx-delete="/api/v1/tokens/{{ t.id }}"
                       hx-confirm="Revoke this token?" hx-swap="none"
                       onclick="location.reload()">Revoke</a>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No tokens for this agent.</p>
    {% endif %}
</article>

{% if tournament_history %}
<article>
    <header>Tournament History</header>
    <table>
        <thead>
            <tr><th>Tournament</th><th>Status</th><th>Score</th><th>Joined</th></tr>
        </thead>
        <tbody>
            {% for p in tournament_history %}
            <tr>
                <td><a href="/ui/tournaments/{{ p.tournament_id }}">Tournament #{{ p.tournament_id }}</a></td>
                <td>{{ p.tournament_status }}</td>
                <td>{{ p.total_score if p.total_score is not none else '—' }}</td>
                <td>{{ p.joined_at.strftime('%Y-%m-%d %H:%M') }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</article>
{% endif %}
{% endblock %}
```

- [ ] **Step 4: Add UI routes**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, add these imports at the top:

```python
from atp.dashboard.models import Agent
from atp.dashboard.tokens import APIToken
from atp.dashboard.tournament.models import Participant, Tournament
```

Then add routes at the end of the file:

```python
@router.get("/agents", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_agents(request: Request, session: DBSession) -> HTMLResponse:
    """My Agents page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)

    result = await session.execute(
        select(Agent)
        .where(Agent.owner_id == user.id, Agent.deleted_at.is_(None))
        .order_by(Agent.created_at.desc())
    )
    agents_raw = result.scalars().all()

    # Count active tokens per agent
    agents = []
    for a in agents_raw:
        count_result = await session.execute(
            select(func.count(APIToken.id)).where(
                APIToken.agent_id == a.id,
                APIToken.revoked_at.is_(None),
            )
        )
        agents.append(
            type("AgentRow", (), {
                **{k: getattr(a, k) for k in ["id", "name", "version", "agent_type", "created_at"]},
                "token_count": count_result.scalar_one(),
            })
        )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/agents.html",
        context={"active_page": "agents", "user": user, "agents": agents},
    )


@router.get("/agents/{agent_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_agent_detail(
    request: Request, session: DBSession, agent_id: int
) -> HTMLResponse:
    """Agent detail page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)

    agent = await session.get(Agent, agent_id)
    if not agent or agent.deleted_at or (agent.owner_id != user.id and not user.is_admin):
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/error.html",
            context={"active_page": "agents", "user": user, "error": "Agent not found"},
            status_code=404,
        )

    # Get tokens for this agent
    token_result = await session.execute(
        select(APIToken)
        .where(APIToken.agent_id == agent_id)
        .order_by(APIToken.created_at.desc())
    )
    tokens = token_result.scalars().all()

    # Get tournament history
    history_result = await session.execute(
        select(Participant, Tournament.status)
        .join(Tournament, Participant.tournament_id == Tournament.id)
        .where(Participant.agent_id == agent_id)
        .order_by(Participant.joined_at.desc())
    )
    tournament_history = [
        type("ParticipantRow", (), {
            "tournament_id": p.tournament_id,
            "tournament_status": t_status,
            "total_score": p.total_score,
            "joined_at": p.joined_at,
        })
        for p, t_status in history_result.all()
    ]

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/agent_detail.html",
        context={
            "active_page": "agents",
            "user": user,
            "agent": agent,
            "tokens": tokens,
            "tournament_history": tournament_history,
            "now": datetime.now(),
        },
    )
```

Add `from fastapi.responses import RedirectResponse` and `from sqlalchemy import func` imports if not already present.

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html \
       packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html \
       packages/atp-dashboard/atp/dashboard/v2/templates/ui/agent_detail.html \
       packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git commit -m "feat: dashboard UI — My Agents page + sidebar navigation"
```

---

## Task 9: Dashboard UI — Tokens + Invites Pages

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tokens.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/invites.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`

- [ ] **Step 1: Create tokens.html template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tokens.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}My Tokens — ATP Platform{% endblock %}

{% block content %}
<h1>My Tokens</h1>

<details id="new-token-form">
    <summary role="button" class="outline">New Token</summary>
    <form hx-post="/api/v1/tokens" hx-target="#token-result" hx-swap="innerHTML">
        <div class="grid">
            <label>Name <input type="text" name="name" required></label>
            <label>Type
                <select name="token_type" onchange="document.getElementById('agent-select').style.display = this.value === 'agent' ? '' : 'none'">
                    <option value="user">User-level</option>
                    <option value="agent">Agent-scoped</option>
                </select>
            </label>
        </div>
        <div id="agent-select" style="display:none">
            <label>Agent
                <select name="agent_id">
                    <option value="">Select agent...</option>
                    {% for a in agents %}
                    <option value="{{ a.id }}">{{ a.name }} ({{ a.version }})</option>
                    {% endfor %}
                </select>
            </label>
        </div>
        <label>Expires
            <select name="expires_in_days">
                <option value="7">7 days</option>
                <option value="30" selected>30 days</option>
                <option value="90">90 days</option>
                <option value="">Never</option>
            </select>
        </label>
        <button type="submit">Create Token</button>
    </form>
    <div id="token-result"></div>
</details>

{% if tokens %}
<figure>
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Agent</th>
            <th>Prefix</th>
            <th>Expires</th>
            <th>Last Used</th>
            <th>Status</th>
            <th>Actions</th>
        </tr>
    </thead>
    <tbody>
        {% for t in tokens %}
        <tr{% if t.revoked_at or (t.expires_at and t.expires_at < now) %} style="opacity:0.5"{% endif %}>
            <td>{{ t.name }}</td>
            <td>{{ 'agent' if t.agent_id else 'user' }}</td>
            <td>{{ t.agent_name or '—' }}</td>
            <td><code>{{ t.token_prefix }}...</code></td>
            <td>{{ t.expires_at.strftime('%Y-%m-%d') if t.expires_at else 'never' }}</td>
            <td>{{ t.last_used_at.strftime('%Y-%m-%d %H:%M') if t.last_used_at else '—' }}</td>
            <td>{% if t.revoked_at %}revoked{% elif t.expires_at and t.expires_at < now %}expired{% else %}active{% endif %}</td>
            <td>
                {% if not t.revoked_at and not (t.expires_at and t.expires_at < now) %}
                <a href="#" hx-delete="/api/v1/tokens/{{ t.id }}"
                   hx-confirm="Revoke this token?" hx-swap="none"
                   onclick="location.reload()">Revoke</a>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
</figure>
{% else %}
<p>No tokens yet.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 2: Create invites.html template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/invites.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}Invites — ATP Platform{% endblock %}

{% block content %}
<h1>Invite Management</h1>

<form hx-post="/api/v1/invites" hx-target="#invite-result" hx-swap="innerHTML" style="display:inline">
    <button type="submit" class="outline">Generate Invite</button>
</form>
<div id="invite-result" style="margin:1rem 0"></div>

{% if invites %}
<figure>
<table>
    <thead>
        <tr>
            <th>Code</th>
            <th>Created By</th>
            <th>Used By</th>
            <th>Status</th>
            <th>Created</th>
            <th>Actions</th>
        </tr>
    </thead>
    <tbody>
        {% for inv in invites %}
        <tr>
            <td><code>{{ inv.code[:20] }}...</code></td>
            <td>{{ inv.created_by_name or inv.created_by_id }}</td>
            <td>{{ inv.used_by_name or '—' }}</td>
            <td>
                {% if inv.use_count >= inv.max_uses %}used
                {% elif inv.expires_at and inv.expires_at < now %}expired
                {% else %}active{% endif %}
            </td>
            <td>{{ inv.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
            <td>
                {% if inv.use_count < inv.max_uses and not (inv.expires_at and inv.expires_at < now) %}
                <a href="#" hx-delete="/api/v1/invites/{{ inv.id }}"
                   hx-confirm="Deactivate this invite?" hx-swap="none"
                   onclick="location.reload()">Deactivate</a>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
</figure>
{% else %}
<p>No invites yet.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 3: Add UI routes for tokens and invites**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, add imports:

```python
from atp.dashboard.tokens import Invite
```

Then add routes:

```python
@router.get("/tokens", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tokens(request: Request, session: DBSession) -> HTMLResponse:
    """My Tokens page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)

    # Get all tokens for this user
    token_result = await session.execute(
        select(APIToken)
        .where(APIToken.user_id == user.id)
        .order_by(APIToken.created_at.desc())
    )
    tokens_raw = token_result.scalars().all()

    # Resolve agent names
    tokens = []
    for t in tokens_raw:
        agent_name = None
        if t.agent_id:
            agent = await session.get(Agent, t.agent_id)
            agent_name = agent.name if agent else f"#{t.agent_id}"
        tokens.append(
            type("TokenRow", (), {
                **{k: getattr(t, k) for k in [
                    "id", "name", "token_prefix", "agent_id",
                    "expires_at", "last_used_at", "revoked_at", "created_at",
                ]},
                "agent_name": agent_name,
            })
        )

    # Get user's agents for the create form
    agents_result = await session.execute(
        select(Agent)
        .where(Agent.owner_id == user.id, Agent.deleted_at.is_(None))
        .order_by(Agent.name)
    )
    agents = agents_result.scalars().all()

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tokens.html",
        context={
            "active_page": "tokens",
            "user": user,
            "tokens": tokens,
            "agents": agents,
            "now": datetime.now(),
        },
    )


@router.get("/invites", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_invites(request: Request, session: DBSession) -> HTMLResponse:
    """Invite management page (admin only)."""
    user = await _get_ui_user(request, session)
    if not user or not user.is_admin:
        return RedirectResponse(url="/ui/login", status_code=302)

    result = await session.execute(
        select(Invite).order_by(Invite.created_at.desc())
    )
    invites_raw = result.scalars().all()

    # Resolve creator/user names
    invites = []
    for inv in invites_raw:
        creator = await session.get(User, inv.created_by_id)
        used_by = await session.get(User, inv.used_by_id) if inv.used_by_id else None
        invites.append(
            type("InviteRow", (), {
                **{k: getattr(inv, k) for k in [
                    "id", "code", "created_by_id", "used_by_id",
                    "use_count", "max_uses", "expires_at", "created_at",
                ]},
                "created_by_name": creator.username if creator else None,
                "used_by_name": used_by.username if used_by else None,
            })
        )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/invites.html",
        context={
            "active_page": "invites",
            "user": user,
            "invites": invites,
            "now": datetime.now(),
        },
    )
```

- [ ] **Step 4: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/tokens.html \
       packages/atp-dashboard/atp/dashboard/v2/templates/ui/invites.html \
       packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git commit -m "feat: dashboard UI — My Tokens + Invites pages"
```

---

## Task 10: Login Page — Invite Code Field

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`

- [ ] **Step 1: Check current login.html**

Read: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html`

- [ ] **Step 2: Add invite code to registration section**

In the login template, find the registration form section and add an invite code field that's conditionally shown. Add before the registration submit button:

```html
{% if registration_mode == 'invite' %}
<label>
    Invite Code
    <input type="text" name="invite_code" placeholder="atp_inv_..." required>
</label>
{% endif %}
```

- [ ] **Step 3: Pass registration_mode to template**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, modify the `ui_login` route to pass the config:

```python
@router.get("/login", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_login(request: Request) -> HTMLResponse:
    """Render login page."""
    from atp.dashboard.v2.config import get_config
    config = get_config()
    expired = request.query_params.get("expired")
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/login.html",
        context={
            "expired": expired,
            "registration_mode": config.registration_mode,
        },
    )
```

- [ ] **Step 4: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html \
       packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git commit -m "feat: login page shows invite code field in invite mode"
```

---

## Task 11: Alembic Migration

**Files:**
- Create: `migrations/dashboard/versions/<hash>_agent_ownership_tokens_invites.py`

- [ ] **Step 1: Create the migration file**

Create `migrations/dashboard/versions/d7f3a2b1c4e5_agent_ownership_tokens_invites.py`:

```python
"""agent_ownership_tokens_invites

Revision ID: d7f3a2b1c4e5
Revises: c60b45e516be
Create Date: 2026-04-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7f3a2b1c4e5"
down_revision: str | Sequence[str] | None = "c60b45e516be"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- New tables ---

    op.create_table(
        "api_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False, server_default="default"),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("token_prefix", sa.String(12), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_api_token_hash"),
    )
    op.create_index("idx_api_token_user", "api_tokens", ["user_id"])
    op.create_index("idx_api_token_hash", "api_tokens", ["token_hash"])

    op.create_table(
        "invites",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(40), nullable=False, unique=True),
        sa.Column("created_by_id", sa.Integer(), nullable=False),
        sa.Column("used_by_id", sa.Integer(), nullable=True),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("max_uses", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["used_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_invite_code", "invites", ["code"])

    # --- Columns on existing tables ---

    op.add_column(
        "agents",
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.add_column(
        "agents",
        sa.Column("version", sa.String(50), nullable=False, server_default="latest"),
    )
    op.add_column(
        "agents",
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_agent_owner", "agents", ["owner_id"])

    # Replace old unique constraint with new one
    # Note: SQLite doesn't support DROP CONSTRAINT natively.
    # For SQLite, the old constraint continues to exist but the new one
    # is more permissive (adds owner_id and version).
    # For PostgreSQL, uncomment the drop:
    # op.drop_constraint("uq_agent_tenant_name", "agents", type_="unique")
    op.create_unique_constraint(
        "uq_agent_tenant_owner_name_version",
        "agents",
        ["tenant_id", "owner_id", "name", "version"],
    )

    op.add_column(
        "tournament_participants",
        sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tournament_participants", "agent_id")
    op.drop_constraint("uq_agent_tenant_owner_name_version", "agents", type_="unique")
    op.drop_index("idx_agent_owner", table_name="agents")
    op.drop_column("agents", "deleted_at")
    op.drop_column("agents", "version")
    op.drop_column("agents", "owner_id")
    op.drop_table("invites")
    op.drop_index("idx_api_token_hash", table_name="api_tokens")
    op.drop_index("idx_api_token_user", table_name="api_tokens")
    op.drop_table("api_tokens")
```

- [ ] **Step 2: Run ruff on the migration**

Run: `uv run ruff format migrations/ && uv run ruff check migrations/ --fix`

- [ ] **Step 3: Commit**

```bash
git add migrations/dashboard/versions/d7f3a2b1c4e5_agent_ownership_tokens_invites.py
git commit -m "feat: Alembic migration for agent ownership + tokens + invites"
```

---

## Task 12: Integration Smoke Test + Full Suite

**Files:**
- All previously created test files

- [ ] **Step 1: Run all new tests together**

Run: `uv run pytest tests/unit/dashboard/test_token_helpers.py tests/unit/dashboard/test_config_new_fields.py tests/integration/dashboard/test_token_api.py tests/integration/dashboard/test_agent_management_api.py tests/integration/dashboard/test_invite_api.py tests/integration/dashboard/test_api_token_auth.py -v`
Expected: All pass

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60 -m "not slow"`
Expected: All pass (no regressions)

- [ ] **Step 3: Run ruff + pyrefly on entire project**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 4: Fix any issues found**

Address any failures, then re-run the failing test(s).

- [ ] **Step 5: Final commit if needed**

```bash
git add -u
git commit -m "fix: address test/lint issues from integration"
```

---

## Spec Coverage Check

| Spec Section | Task(s) |
|-------------|---------|
| Agent model changes (owner_id, version, deleted_at) | Task 2 |
| APIToken model | Task 2 |
| Invite model | Task 2 |
| Participant.agent_id | Task 2 |
| Token generation helpers | Task 2 |
| Config env vars | Task 1 |
| Pydantic schemas | Task 3 |
| Auth middleware (API token resolution) | Task 4 |
| Token CRUD API | Task 5 |
| Agent management API | Task 6 |
| Invite API + registration mode | Task 7 |
| Dashboard sidebar | Task 8 |
| My Agents page | Task 8 |
| Agent detail page | Task 8 |
| My Tokens page | Task 9 |
| Invites page | Task 9 |
| Login invite code field | Task 10 |
| Alembic migration | Task 11 |
| Error handling (401/403/409/400) | Tasks 5, 6, 7 |
| Rate limiting (10/min on creation) | Tasks 5, 6 |
| Soft delete + token revocation cascade | Task 6 |
| last_used_at atomic debounce | Task 4 |
| Token prefix 12 chars | Task 2 |
| Invite-before-OAuth flow | Task 7 (registration endpoint); GitHub OAuth callback integration deferred to separate task |
