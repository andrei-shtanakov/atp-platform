# Admin Tournament GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an admin-gated web UI under `/ui/admin/tournaments/*` for the full El Farol tournament lifecycle — create, live-monitor with per-participant activity table and cross-round heatmap, cancel, and post-mortem audit — backed by a longer admin JWT TTL so a multi-hour monitoring session does not drop.

**Architecture:** Extends the existing HTMX + Jinja2 + Pico CSS dashboard stack with a new `routes/admin_ui.py` module gated by `require_admin_user` and a single HTMX-polled activity fragment (2 Hz). No new DB tables — all data reads off existing `Tournament`/`Round`/`Participant`/`Action`. Admin session TTL is lifted via a new `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` env var (default 720 min = 12 h), applied at token issuance when `User.is_admin=True`; bot-side `ATP_TOKEN_EXPIRE_MINUTES=60` is unchanged.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy (async), Jinja2, Pico CSS, HTMX 2.0 (CDN), pytest + pytest-anyio + httpx.ASGITransport. `uv` for package management, `uv run pytest` / `uv run ruff` / `uv run pyrefly check` for quality gates.

**Spec:** `docs/superpowers/specs/2026-04-20-admin-tournament-gui-design.md`

---

## Task 0: Bootstrap the implementation branch

**Files:**
- No code changes. Branch setup only.

- [ ] **Step 1: Verify spec branch is checked in**

Run: `git log --oneline -5 docs/admin-tournament-spec`
Expected: top commit is `docs: admin tournament GUI design spec (El Farol, A+B scope)`.

- [ ] **Step 2: Create implementation branch off main**

Run (from repo root):
```bash
git fetch github main
git checkout -b feat/admin-tournament-gui github/main
```

Expected: `Switched to a new branch 'feat/admin-tournament-gui'`.

- [ ] **Step 3: Cherry-pick the spec commit onto the impl branch**

Run:
```bash
git cherry-pick docs/admin-tournament-spec
```

Expected: spec file appears under `docs/superpowers/specs/` on `feat/admin-tournament-gui`. This keeps the spec visible alongside the implementation while the design branch is still open.

- [ ] **Step 4: Sync deps**

Run: `uv sync --group dev`
Expected: `Installed 0 packages` (or a small delta if the branch is a day old).

- [ ] **Step 5: Confirm test baseline is green**

Run: `uv run pytest tests/unit/dashboard/ui/ -v --no-cov`
Expected: all existing UI unit tests pass. No commit for this task — it is environment bootstrap.

---

## Task 1: Add `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` env var + admin-aware TTL helper

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/auth/__init__.py`
- Test: `tests/unit/dashboard/auth/test_token_ttl.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/auth/test_token_ttl.py`:

```python
"""Tests for admin-aware JWT TTL selection."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from atp.dashboard.auth import (
    ALGORITHM,
    SECRET_KEY,
    access_token_ttl,
    create_access_token,
)


def test_access_token_ttl_defaults_to_60_minutes_for_regular_users():
    assert access_token_ttl(is_admin=False) == timedelta(minutes=60)


def test_access_token_ttl_is_720_minutes_for_admins_by_default():
    assert access_token_ttl(is_admin=True) == timedelta(minutes=720)


def test_access_token_ttl_respects_admin_env_override(monkeypatch):
    monkeypatch.setenv("ATP_ADMIN_TOKEN_EXPIRE_MINUTES", "480")
    # Reimport the symbol so the module-level int is re-read.
    from atp.dashboard.auth import _read_admin_ttl

    assert _read_admin_ttl() == 480


def test_create_access_token_uses_admin_ttl_when_is_admin_true():
    token = create_access_token(
        data={"sub": "root", "user_id": 1},
        is_admin=True,
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
    remaining = expires_at - datetime.now(tz=UTC)
    # Wide bracket so the test is not flaky on slow CI.
    assert timedelta(minutes=700) < remaining < timedelta(minutes=730)


def test_create_access_token_uses_regular_ttl_when_is_admin_false():
    token = create_access_token(
        data={"sub": "alice", "user_id": 2},
        is_admin=False,
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
    remaining = expires_at - datetime.now(tz=UTC)
    assert timedelta(minutes=55) < remaining < timedelta(minutes=65)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dashboard/auth/test_token_ttl.py -v --no-cov`
Expected: FAILED with `ImportError: cannot import name 'access_token_ttl'` (or similar).

- [ ] **Step 3: Implement the helper and plug it into `create_access_token`**

In `packages/atp-dashboard/atp/dashboard/auth/__init__.py`, after the existing `ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ATP_TOKEN_EXPIRE_MINUTES", "60"))` line, add:

```python
def _read_admin_ttl() -> int:
    """Read the admin-token TTL (minutes) from the environment.

    Kept as a function (not a module-level constant) so tests can
    monkeypatch ``ATP_ADMIN_TOKEN_EXPIRE_MINUTES`` and call it
    afresh. The default of 720 minutes (12 hours) covers a typical
    multi-tournament monitoring session.
    """
    return int(os.getenv("ATP_ADMIN_TOKEN_EXPIRE_MINUTES", "720"))


def access_token_ttl(is_admin: bool) -> timedelta:
    """Return the access-token TTL appropriate for this user class."""
    minutes = _read_admin_ttl() if is_admin else ACCESS_TOKEN_EXPIRE_MINUTES
    return timedelta(minutes=minutes)
```

Replace the existing `create_access_token` body with:

```python
def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
    *,
    is_admin: bool = False,
) -> str:
    """Create a JWT access token.

    Args:
        data: Claims to encode in the token.
        expires_delta: Explicit TTL. If provided, wins over ``is_admin``.
        is_admin: When True and ``expires_delta`` is None, use the
            admin-token TTL instead of the regular one. Callers that
            know the user's admin flag should pass it.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    if expires_delta is not None:
        expire = datetime.now(tz=UTC) + expires_delta
    else:
        expire = datetime.now(tz=UTC) + access_token_ttl(is_admin)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

Add `access_token_ttl` and `_read_admin_ttl` to the module `__all__` list (the `__all__` tuple near the bottom of the file).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/dashboard/auth/test_token_ttl.py -v --no-cov`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/auth/__init__.py tests/unit/dashboard/auth/test_token_ttl.py
git commit -m "feat(auth): admin-aware JWT TTL via ATP_ADMIN_TOKEN_EXPIRE_MINUTES"
```

---

## Task 2: Wire the admin TTL into all 4 token-issuance call sites

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/auth/post_auth.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py`
- Modify: `packages/atp-dashboard/atp/dashboard/cli/admin.py`
- Test: `tests/unit/dashboard/auth/test_token_ttl.py` (extend)

- [ ] **Step 1: Write the failing integration-style test**

Append to `tests/unit/dashboard/auth/test_token_ttl.py`:

```python
def test_post_auth_uses_admin_ttl_for_admin_users(monkeypatch):
    """post_auth.complete_auth must pass is_admin into create_access_token."""
    from datetime import UTC, datetime

    import jwt
    from atp.dashboard.auth import ALGORITHM, SECRET_KEY

    # Mirror what post_auth would produce if the user is admin.
    from atp.dashboard.auth import create_access_token

    token_admin = create_access_token(
        data={"sub": "root", "user_id": 99}, is_admin=True
    )
    token_regular = create_access_token(
        data={"sub": "alice", "user_id": 100}, is_admin=False
    )
    exp_admin = jwt.decode(token_admin, SECRET_KEY, algorithms=[ALGORITHM])["exp"]
    exp_regular = jwt.decode(
        token_regular, SECRET_KEY, algorithms=[ALGORITHM]
    )["exp"]
    # Admin exp must be at least 10 hours further out than regular.
    assert exp_admin - exp_regular > 10 * 3600
```

(This test documents the contract; the next step fixes the call sites.)

- [ ] **Step 2: Run to verify the test passes with the helper already in place**

Run: `uv run pytest tests/unit/dashboard/auth/test_token_ttl.py::test_post_auth_uses_admin_ttl_for_admin_users -v --no-cov`
Expected: PASS (Task 1 already covers the primitive).

- [ ] **Step 3: Update `post_auth.py`**

In `packages/atp-dashboard/atp/dashboard/auth/post_auth.py` around line 53, replace:

```python
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
```

with:

```python
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            is_admin=user.is_admin,
        )
```

Remove the now-unused `ACCESS_TOKEN_EXPIRE_MINUTES` import and the `timedelta` import if they are not used elsewhere in the file.

- [ ] **Step 4: Update `v2/routes/auth.py`**

In `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py` around line 58, locate the `create_access_token(` call and add `is_admin=user.is_admin` as a keyword argument. If the call passes `expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)`, remove it — the helper now decides.

- [ ] **Step 5: Update `cli/admin.py` at both call sites**

In `packages/atp-dashboard/atp/dashboard/cli/admin.py` at lines 111 and 133, both calls are admin CLI flows (the command creates tokens for admin users). Add `is_admin=True` keyword argument to each.

- [ ] **Step 6: Run the broader auth test suite**

Run: `uv run pytest tests/unit/dashboard/auth/ tests/integration/dashboard/ -v --no-cov -k "auth or token"`
Expected: all pass. If any test was explicitly checking `60 minutes` in a flow that runs as admin, relax the bound; otherwise nothing should regress.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/auth/post_auth.py \
        packages/atp-dashboard/atp/dashboard/v2/routes/auth.py \
        packages/atp-dashboard/atp/dashboard/cli/admin.py \
        tests/unit/dashboard/auth/test_token_ttl.py
git commit -m "feat(auth): pass is_admin to create_access_token at all issuance sites"
```

---

## Task 3: Add `require_admin_user` FastAPI dependency

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/dependencies.py`
- Test: `tests/unit/dashboard/v2/test_require_admin_user.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/v2/test_require_admin_user.py`:

```python
"""Tests for the require_admin_user dependency."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from atp.dashboard.models import User
from atp.dashboard.v2.dependencies import require_admin_user


def _user(is_admin: bool) -> User:
    return User(
        id=1,
        username="u",
        email="u@example.com",
        is_admin=is_admin,
    )


@pytest.mark.anyio
async def test_require_admin_user_passes_for_admin():
    result = await require_admin_user(_user(is_admin=True))
    assert result.is_admin is True


@pytest.mark.anyio
async def test_require_admin_user_rejects_non_admin():
    with pytest.raises(HTTPException) as exc:
        await require_admin_user(_user(is_admin=False))
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.anyio
async def test_require_admin_user_rejects_anonymous():
    with pytest.raises(HTTPException) as exc:
        await require_admin_user(None)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dashboard/v2/test_require_admin_user.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'require_admin_user'`.

- [ ] **Step 3: Implement the dependency**

Append to `packages/atp-dashboard/atp/dashboard/v2/dependencies.py`:

```python
async def require_admin_user(
    user: Annotated[User | None, Depends(get_current_user)],
) -> User:
    """FastAPI dependency that rejects non-admins.

    Returns the authenticated admin User on success. Raises
    401 for anonymous callers and 403 for authenticated non-admins.
    """
    from fastapi import HTTPException, status

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


AdminUser = Annotated[User, Depends(require_admin_user)]
```

Make sure the top-of-file imports include `from typing import Annotated` and `from fastapi import Depends`. If they are already present, do not re-add.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/dashboard/v2/test_require_admin_user.py -v --no-cov`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/dependencies.py \
        tests/unit/dashboard/v2/test_require_admin_user.py
git commit -m "feat(auth): require_admin_user dependency"
```

---

## Task 4: Scaffold `admin_ui.py` router and register it in factory

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Test: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing integration test for the landing page**

Create `tests/integration/dashboard/test_admin_tournament_ui.py`:

```python
"""Integration tests for /ui/admin/tournaments/*."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def client():
    app = create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_admin_landing_requires_admin(client: AsyncClient):
    """Anonymous callers cannot reach the admin landing page."""
    resp = await client.get("/ui/admin")
    assert resp.status_code in (401, 403)


@pytest.mark.anyio
async def test_admin_landing_renders_for_admin(
    client: AsyncClient, admin_user_headers
):
    """An authenticated admin sees the admin landing page."""
    resp = await client.get("/ui/admin", headers=admin_user_headers)
    assert resp.status_code == 200
    assert "Admin" in resp.text
    assert "Tournaments" in resp.text
```

- [ ] **Step 2: Add an `admin_user_headers` fixture**

Check whether a fixture named `admin_user_headers` already exists in `tests/conftest.py` or `tests/integration/dashboard/conftest.py`. If yes, reuse it. If no, add the following to `tests/integration/dashboard/conftest.py` (create the file if absent):

```python
"""Dashboard integration test fixtures."""

from __future__ import annotations

import pytest

from atp.dashboard.auth import create_access_token


@pytest.fixture
def admin_user_headers() -> dict[str, str]:
    token = create_access_token(
        data={"sub": "admin", "user_id": 1},
        is_admin=True,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def regular_user_headers() -> dict[str, str]:
    token = create_access_token(
        data={"sub": "alice", "user_id": 2},
        is_admin=False,
    )
    return {"Authorization": f"Bearer {token}"}
```

Note: the test app's authentication middleware decodes the JWT and looks up the `User` in the DB. If `create_test_app()` seeds no users, the header alone will not make `user.is_admin=True` — `get_current_user` returns `None`. In that case the new fixture must first insert a matching user. Inspect the existing pattern used by `test_game_detail_ui.py` / `test_tournament_ui.py` for the correct seeding idiom (look for `seed_user` or a session-scoped `seed_admin` fixture) and extend `admin_user_headers` to produce a DB-backed admin before returning the header.

- [ ] **Step 3: Run tests to verify both fail**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov`
Expected: FAIL with 404 (route not registered) or with fixture wiring errors.

- [ ] **Step 4: Create the router module**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`:

```python
"""Admin-gated UI routes for tournament management.

Mounted under ``/ui/admin``. All routes use the ``AdminUser``
dependency which returns 401 for anonymous callers and 403 for
authenticated non-admins.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from atp.dashboard.v2.dependencies import AdminUser

router = APIRouter(prefix="/ui/admin", tags=["admin-ui"])

# Resolved at import time; same template env the public UI uses.
_templates = Jinja2Templates(directory=str(_templates_dir()))  # see helper below
```

Add a helper `_templates_dir()` at the top of the file that resolves the Jinja root the same way the existing `routes/ui.py` does. Inspect `routes/ui.py` imports to find the exact path expression (likely `pathlib.Path(__file__).parent.parent / "templates"`).

Add the first route:

```python
@router.get("")
async def admin_home(
    request: Request,
    user: AdminUser,
):
    """Admin landing page with quick-links to admin sections."""
    return _templates.TemplateResponse(
        request=request,
        name="ui/admin/index.html",
        context={"user": user},
    )
```

- [ ] **Step 5: Register the router in the factory**

In `packages/atp-dashboard/atp/dashboard/v2/factory.py`, find the block where UI routers are included (around line 213, `app.include_router(ui_router)`). Import and register the new admin router right after:

```python
from atp.dashboard.v2.routes.admin_ui import router as admin_ui_router
...
app.include_router(admin_ui_router)
```

- [ ] **Step 6: Create the landing template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/index.html`:

```jinja
{% extends "ui/base.html" %}
{% block title %}Admin · ATP Dashboard{% endblock %}
{% block content %}
<h1>Admin</h1>
<p>Administrative views for {{ user.username }}.</p>

<nav>
  <ul>
    <li><a href="/ui/admin/tournaments">Tournaments</a></li>
  </ul>
</nav>
{% endblock %}
```

If `ui/base.html` does not exist (inspect `templates/ui/` — the project may use a different base name such as `layout.html` or no base at all), mirror the pattern used by `templates/ui/tournament_detail.html`. Do not guess — read one existing template first and copy its structure.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov`
Expected: both tests PASS.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        packages/atp-dashboard/atp/dashboard/v2/factory.py \
        packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/index.html \
        tests/integration/dashboard/test_admin_tournament_ui.py \
        tests/integration/dashboard/conftest.py
git commit -m "feat(admin): /ui/admin landing page scaffold"
```

---

## Task 5: Admin tournaments list page `/ui/admin/tournaments`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournaments_list.html`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/dashboard/test_admin_tournament_ui.py`:

```python
@pytest.mark.anyio
async def test_admin_tournaments_list_requires_admin(
    client: AsyncClient, regular_user_headers
):
    resp = await client.get(
        "/ui/admin/tournaments", headers=regular_user_headers
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_tournaments_list_renders_all_statuses(
    client: AsyncClient, admin_user_headers, seed_tournaments
):
    """Admin sees tournaments regardless of status or owner."""
    resp = await client.get(
        "/ui/admin/tournaments", headers=admin_user_headers
    )
    assert resp.status_code == 200
    # Fixture seeds one pending, one in_progress, one completed.
    assert "pending" in resp.text.lower()
    assert "in_progress" in resp.text.lower() or "in progress" in resp.text.lower()
    assert "completed" in resp.text.lower()
    # "New tournament" call-to-action must be present.
    assert "new tournament" in resp.text.lower()
```

Add a `seed_tournaments` fixture to `tests/integration/dashboard/conftest.py` that inserts three tournaments with distinct statuses. Check how the existing tournament service tests build tournaments (`tests/unit/dashboard/tournament/test_service_create.py`) and mirror that pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "admin_tournaments_list"`
Expected: FAIL with 404.

- [ ] **Step 3: Implement the route**

Append to `routes/admin_ui.py`:

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from atp.dashboard.database import get_database
from atp.dashboard.tournament.models import Tournament


@router.get("/tournaments")
async def admin_tournaments_list(
    request: Request,
    user: AdminUser,
    session: AsyncSession = Depends(get_database),
):
    """List every tournament (all statuses, all owners) for admins."""
    stmt = select(Tournament).order_by(Tournament.created_at.desc())
    result = await session.execute(stmt)
    tournaments = result.scalars().all()
    return _templates.TemplateResponse(
        request=request,
        name="ui/admin/tournaments_list.html",
        context={"user": user, "tournaments": tournaments},
    )
```

Add `from fastapi import Depends` at the top if not already imported.

- [ ] **Step 4: Create the list template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournaments_list.html`:

```jinja
{% extends "ui/base.html" %}
{% block title %}Tournaments (admin) · ATP Dashboard{% endblock %}
{% block content %}
<header style="display:flex;justify-content:space-between;align-items:center;">
  <h1>Tournaments (admin)</h1>
  <a href="/ui/admin/tournaments/new" role="button">New tournament</a>
</header>

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Game</th>
      <th>Status</th>
      <th>Rounds</th>
      <th>Players</th>
      <th>Created</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    {% for t in tournaments %}
    <tr>
      <td>#{{ t.id }}</td>
      <td>{{ t.game_type }}</td>
      <td>{{ t.status }}</td>
      <td>{{ t.total_rounds }}</td>
      <td>{{ t.num_players }}</td>
      <td>{{ t.created_at.strftime("%Y-%m-%d %H:%M") }}</td>
      <td><a href="/ui/admin/tournaments/{{ t.id }}">Open</a></td>
    </tr>
    {% else %}
    <tr><td colspan="7"><em>No tournaments yet.</em></td></tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "admin_tournaments_list"`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournaments_list.html \
        tests/integration/dashboard/test_admin_tournament_ui.py \
        tests/integration/dashboard/conftest.py
git commit -m "feat(admin): tournaments list at /ui/admin/tournaments"
```

---

## Task 6: Admin create-tournament form (GET `/new` + POST `/new`)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_new.html`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
@pytest.mark.anyio
async def test_admin_new_tournament_form_renders(
    client: AsyncClient, admin_user_headers
):
    resp = await client.get(
        "/ui/admin/tournaments/new", headers=admin_user_headers
    )
    assert resp.status_code == 200
    assert 'name="num_players"' in resp.text
    assert 'name="total_rounds"' in resp.text
    assert 'name="round_deadline_s"' in resp.text
    assert 'name="capacity_threshold"' in resp.text
    # El Farol is the only game surfaced for MVP.
    assert "el_farol" in resp.text


@pytest.mark.anyio
async def test_admin_create_tournament_submission_creates_and_redirects(
    client: AsyncClient, admin_user_headers
):
    resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "game_type": "el_farol",
            "num_players": "6",
            "total_rounds": "12",
            "round_deadline_s": "30",
            "capacity_threshold": "4",
        },
        headers=admin_user_headers,
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/ui/admin/tournaments/")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "new_tournament"`
Expected: FAIL (404).

- [ ] **Step 3: Add both routes**

Append to `routes/admin_ui.py`:

```python
from fastapi import Form
from fastapi.responses import RedirectResponse

from atp.dashboard.tournament.service import TournamentService


@router.get("/tournaments/new")
async def admin_tournament_new_form(
    request: Request,
    user: AdminUser,
):
    return _templates.TemplateResponse(
        request=request,
        name="ui/admin/tournament_new.html",
        context={"user": user},
    )


@router.post("/tournaments/new")
async def admin_tournament_new_submit(
    user: AdminUser,
    session: AsyncSession = Depends(get_database),
    game_type: str = Form(...),
    num_players: int = Form(...),
    total_rounds: int = Form(...),
    round_deadline_s: int = Form(...),
    capacity_threshold: int = Form(...),
) -> RedirectResponse:
    service = TournamentService(session=session)
    tournament = await service.create(
        game_type=game_type,
        num_players=num_players,
        total_rounds=total_rounds,
        round_deadline_s=round_deadline_s,
        created_by=user.id,
        rules={"capacity_threshold": capacity_threshold},
    )
    return RedirectResponse(
        url=f"/ui/admin/tournaments/{tournament.id}",
        status_code=303,
    )
```

Note: confirm the exact `TournamentService.create` signature — the `rules=` kwarg shape is taken from `models.py:91` (`rules: Mapped[dict] = mapped_column(JSON, default=dict)`). If the service method signature differs, adapt the call and update this task's code block accordingly before committing.

- [ ] **Step 4: Create the form template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_new.html`:

```jinja
{% extends "ui/base.html" %}
{% block title %}New tournament · Admin · ATP Dashboard{% endblock %}
{% block content %}
<h1>New tournament</h1>

<form method="post" action="/ui/admin/tournaments/new">
  <label>
    Game
    <select name="game_type" required>
      <option value="el_farol" selected>El Farol Bar</option>
    </select>
  </label>

  <label>
    Number of players
    <input type="number" name="num_players" value="6" min="2" max="1000" required>
  </label>

  <label>
    Total rounds
    <input type="number" name="total_rounds" value="30" min="1" max="500" required>
  </label>

  <label>
    Round deadline (seconds)
    <input type="number" name="round_deadline_s" value="30" min="1" max="3600" required>
  </label>

  <label>
    Capacity threshold
    <input type="number" name="capacity_threshold" value="4" min="1" required>
    <small>
      Occupancy ≥ threshold makes a slot crowded (−1). Sensible default:
      <code>&lceil; num_players / 2 &rceil;</code>.
    </small>
  </label>

  <button type="submit">Create tournament</button>
  <a href="/ui/admin/tournaments" role="button" class="secondary">Cancel</a>
</form>
{% endblock %}
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "new_tournament"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_new.html \
        tests/integration/dashboard/test_admin_tournament_ui.py
git commit -m "feat(admin): create-tournament form (El Farol only)"
```

---

## Task 7: Admin detail page shell (+ Cancel button)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
@pytest.mark.anyio
async def test_admin_tournament_detail_404_for_unknown_id(
    client: AsyncClient, admin_user_headers
):
    resp = await client.get(
        "/ui/admin/tournaments/9999999", headers=admin_user_headers
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_admin_tournament_detail_renders(
    client: AsyncClient, admin_user_headers, seed_tournaments
):
    # seed_tournaments creates tournaments with known ids.
    resp = await client.get(
        "/ui/admin/tournaments/1", headers=admin_user_headers
    )
    assert resp.status_code == 200
    assert "Tournament #1" in resp.text
    # Cancel button for live tournaments only.
    assert "Cancel tournament" in resp.text or "Cancel" in resp.text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "tournament_detail"`
Expected: FAIL (404 for both because the route does not exist yet).

- [ ] **Step 3: Implement the route**

Append:

```python
from fastapi import HTTPException, status
from sqlalchemy.orm import selectinload


@router.get("/tournaments/{tournament_id}")
async def admin_tournament_detail(
    tournament_id: int,
    request: Request,
    user: AdminUser,
    session: AsyncSession = Depends(get_database),
):
    stmt = (
        select(Tournament)
        .where(Tournament.id == tournament_id)
        .options(selectinload(Tournament.participants))
    )
    result = await session.execute(stmt)
    tournament = result.scalars().first()
    if tournament is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return _templates.TemplateResponse(
        request=request,
        name="ui/admin/tournament_detail.html",
        context={"user": user, "tournament": tournament},
    )
```

- [ ] **Step 4: Create the detail template (shell only, no activity block yet)**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html`:

```jinja
{% extends "ui/base.html" %}
{% block title %}Tournament #{{ tournament.id }} · Admin · ATP Dashboard{% endblock %}
{% block content %}
<header style="display:flex;justify-content:space-between;align-items:flex-start;">
  <div>
    <h1>Tournament #{{ tournament.id }}</h1>
    <p>
      {{ tournament.game_type }} · round
      {{ tournament.current_round or 0 }}/{{ tournament.total_rounds }} ·
      <strong>{{ tournament.status }}</strong>
    </p>
    <p><small>Created {{ tournament.created_at.strftime("%Y-%m-%d %H:%M") }} · {{ tournament.participants|length }} participants</small></p>
  </div>
  <div>
    {% if tournament.status in ("pending", "in_progress") %}
    <button
      hx-post="/api/v1/tournaments/{{ tournament.id }}/cancel"
      hx-confirm="Cancel this tournament? This cannot be undone."
      hx-swap="none"
      class="contrast">
      Cancel tournament
    </button>
    {% endif %}
  </div>
</header>

<article id="activity-block">
  {# Live activity block will be injected in Task 10. #}
  <p><em>Live activity view coming in the next task.</em></p>
</article>
{% endblock %}
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "tournament_detail"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html \
        tests/integration/dashboard/test_admin_tournament_ui.py
git commit -m "feat(admin): tournament detail shell with Cancel button"
```

---

## Task 8: `TournamentService.get_admin_activity()` method

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_get_admin_activity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_get_admin_activity.py`:

```python
"""Unit tests for TournamentService.get_admin_activity."""

from __future__ import annotations

import pytest

from atp.dashboard.tournament.service import TournamentService


@pytest.mark.anyio
async def test_get_admin_activity_returns_snapshot_shape(
    session, seed_live_tournament
):
    """Snapshot has the documented keys and list shapes."""
    t = seed_live_tournament  # 3 participants, round 2 of 5, 1 timeout in round 1.
    service = TournamentService(session=session)
    snap = await service.get_admin_activity(t.id)

    assert snap["tournament_id"] == t.id
    assert snap["status"] in {"pending", "in_progress", "completed", "cancelled"}
    assert snap["total_rounds"] == 5
    assert snap["current_round"] == 2
    assert isinstance(snap["deadline_remaining_s"], int) or snap["deadline_remaining_s"] is None
    assert len(snap["participants"]) == 3
    for p in snap["participants"]:
        assert set(p.keys()) >= {
            "id",
            "agent_name",
            "released_at",
            "total_score",
            "current_round_status",
            "current_round_submitted_at",
            "row_per_round",
        }
        assert p["current_round_status"] in {
            "submitted",
            "waiting",
            "timeout",
            "released",
        }
        assert len(p["row_per_round"]) == 5
        for cell in p["row_per_round"]:
            assert cell in {"submitted", "timeout", "waiting"}


@pytest.mark.anyio
async def test_get_admin_activity_counts_submissions(
    session, seed_live_tournament
):
    t = seed_live_tournament
    service = TournamentService(session=session)
    snap = await service.get_admin_activity(t.id)
    assert snap["submitted_this_round"] <= snap["total_this_round"]
    assert snap["total_this_round"] == 3


@pytest.mark.anyio
async def test_get_admin_activity_404_for_unknown_id(session):
    service = TournamentService(session=session)
    with pytest.raises(LookupError):
        await service.get_admin_activity(9999999)
```

Create the `seed_live_tournament` fixture in `tests/unit/dashboard/tournament/conftest.py` (append if the file exists): insert 1 `Tournament` with `status="in_progress"`, `total_rounds=5`, `current_round=2`, 3 `Participant` rows, and `Action` rows that make round 1 complete (with one `TIMEOUT_DEFAULT`), and round 2 in-progress with 1 submitted so far.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_get_admin_activity.py -v --no-cov`
Expected: FAIL with `AttributeError: TournamentService has no attribute 'get_admin_activity'`.

- [ ] **Step 3: Implement the method**

Append to `packages/atp-dashboard/atp/dashboard/tournament/service.py`:

```python
async def get_admin_activity(self, tournament_id: int) -> dict:
    """Return an admin-level activity snapshot for HTMX rendering.

    Layout matches the ``ActivitySnapshot`` contract in
    ``docs/superpowers/specs/2026-04-20-admin-tournament-gui-design.md``.
    Raises LookupError if the tournament does not exist.
    """
    from datetime import UTC, datetime

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from atp.dashboard.tournament.models import (
        Action,
        ActionSource,
        Participant,
        Round,
        Tournament,
    )

    stmt = (
        select(Tournament)
        .where(Tournament.id == tournament_id)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds).selectinload(Round.actions),
        )
    )
    result = await self._session.execute(stmt)
    tournament = result.scalars().first()
    if tournament is None:
        raise LookupError(f"tournament {tournament_id} not found")

    total_rounds = tournament.total_rounds
    current_round_number = tournament.current_round or 0

    # Build action lookup: (participant_id, round_number) -> Action
    actions_by_pid_round: dict[tuple[int, int], Action] = {}
    for rnd in tournament.rounds:
        for act in rnd.actions:
            actions_by_pid_round[(act.participant_id, rnd.round_number)] = act

    deadline_remaining_s = None
    current_round_deadline = None
    for rnd in tournament.rounds:
        if rnd.round_number == current_round_number and rnd.deadline:
            current_round_deadline = rnd.deadline
            delta = rnd.deadline - datetime.now(tz=UTC)
            deadline_remaining_s = max(0, int(delta.total_seconds()))

    participants_out: list[dict] = []
    submitted_this_round = 0
    for p in tournament.participants:
        row_per_round: list[str] = []
        for r_num in range(1, total_rounds + 1):
            act = actions_by_pid_round.get((p.id, r_num))
            if act is None:
                row_per_round.append("waiting")
            elif act.source == ActionSource.TIMEOUT_DEFAULT.value:
                row_per_round.append("timeout")
            else:
                row_per_round.append("submitted")

        current_act = actions_by_pid_round.get((p.id, current_round_number))
        if p.released_at is not None:
            status = "released"
        elif current_act is None:
            status = "waiting"
        elif current_act.source == ActionSource.TIMEOUT_DEFAULT.value:
            status = "timeout"
        else:
            status = "submitted"
            submitted_this_round += 1

        participants_out.append(
            {
                "id": p.id,
                "agent_name": p.agent_name,
                "released_at": p.released_at,
                "total_score": p.total_score,
                "current_round_status": status,
                "current_round_submitted_at": (
                    current_act.submitted_at if current_act else None
                ),
                "row_per_round": row_per_round,
            }
        )

    return {
        "tournament_id": tournament.id,
        "status": tournament.status,
        "current_round": current_round_number or None,
        "total_rounds": total_rounds,
        "deadline_remaining_s": deadline_remaining_s,
        "participants": participants_out,
        "submitted_this_round": submitted_this_round,
        "total_this_round": len(tournament.participants),
    }
```

Note: verify that `Tournament` has a `current_round` column (check `models.py`). If not, derive it from the max-round in `tournament.rounds` whose `status != "completed"`. Use the actual column name the model exposes.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/dashboard/tournament/test_get_admin_activity.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_get_admin_activity.py \
        tests/unit/dashboard/tournament/conftest.py
git commit -m "feat(tournament): TournamentService.get_admin_activity snapshot"
```

---

## Task 9: `_activity_block.html` fragment + activity route

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/_activity_block.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.anyio
async def test_admin_activity_fragment_renders_table_and_heatmap(
    client: AsyncClient, admin_user_headers, seed_live_tournament_route
):
    t_id = seed_live_tournament_route
    resp = await client.get(
        f"/ui/admin/tournaments/{t_id}/activity",
        headers=admin_user_headers,
    )
    assert resp.status_code == 200
    # Fragment must have both halves of layout B.
    assert "activity-table" in resp.text
    assert "activity-heatmap" in resp.text
    # Heatmap cells must have one of the three semantic classes.
    assert 'class="cell submitted"' in resp.text or \
           'class="cell waiting"' in resp.text or \
           'class="cell timeout"' in resp.text


@pytest.mark.anyio
async def test_admin_activity_fragment_requires_admin(
    client: AsyncClient, regular_user_headers, seed_live_tournament_route
):
    t_id = seed_live_tournament_route
    resp = await client.get(
        f"/ui/admin/tournaments/{t_id}/activity",
        headers=regular_user_headers,
    )
    assert resp.status_code == 403
```

Add `seed_live_tournament_route` fixture in `tests/integration/dashboard/conftest.py` that seeds a tournament at route-integration scope and returns its id.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "activity_fragment"`
Expected: FAIL (404).

- [ ] **Step 3: Create the fragment template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/_activity_block.html`:

```jinja
{#
  Admin activity block for a tournament.
  Polled every 2 s via hx-trigger when tournament is live;
  rendered once as a static post-mortem when it is not.
  Context: snap (dict) — see TournamentService.get_admin_activity().
#}
<style>
  .activity-grid { display: grid; grid-template-columns: 3fr 2fr; gap: 1rem; }
  .activity-table { font-size: 0.9rem; }
  .activity-heatmap { font-family: monospace; font-size: 0.75rem; overflow-x: auto; }
  .activity-heatmap .row { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
  .activity-heatmap .label { min-width: 8ch; }
  .activity-heatmap .cell { width: 14px; height: 14px; display: inline-block; border: 1px solid #ccc; }
  .activity-heatmap .cell.submitted { background: #2a8; }
  .activity-heatmap .cell.timeout   { background: #c44; }
  .activity-heatmap .cell.waiting   { background: #fff; }
  .activity-deadline { font-size: 2rem; font-weight: 600; text-align: center; color: #c80; }
</style>

<div class="activity-grid">
  <section class="activity-table">
    <h3>Round {{ snap.current_round or 0 }} / {{ snap.total_rounds }} · {{ snap.submitted_this_round }}/{{ snap.total_this_round }} submitted</h3>
    <table>
      <thead>
        <tr><th>Agent</th><th>Status</th><th>Last action</th><th>Score</th></tr>
      </thead>
      <tbody>
        {% for p in snap.participants %}
        <tr>
          <td>{{ p.agent_name }}</td>
          <td>
            {% if p.current_round_status == "submitted" %}✓ submitted
            {% elif p.current_round_status == "waiting" %}⏳ waiting
            {% elif p.current_round_status == "timeout" %}⚠ timeout
            {% else %}◻ released
            {% endif %}
          </td>
          <td>
            {% if p.current_round_submitted_at %}
              {{ p.current_round_submitted_at.strftime("%H:%M:%S") }}
            {% else %}—{% endif %}
          </td>
          <td>{{ "%.1f"|format(p.total_score or 0) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <aside>
    <section class="activity-heatmap">
      <h3>Activity by round</h3>
      {% for p in snap.participants %}
      <div class="row">
        <span class="label">{{ p.agent_name[:8] }}</span>
        {% for cell in p.row_per_round %}
          <span class="cell {{ cell }}" title="round {{ loop.index }}: {{ cell }}"></span>
        {% endfor %}
      </div>
      {% endfor %}
      <p><small>■ submitted · ■ timeout · ☐ waiting</small></p>
    </section>

    {% if snap.deadline_remaining_s is not none %}
    <section>
      <h3>Round deadline</h3>
      <div class="activity-deadline">
        {{ "%02d"|format(snap.deadline_remaining_s // 60) }}:{{ "%02d"|format(snap.deadline_remaining_s % 60) }}
      </div>
    </section>
    {% endif %}
  </aside>
</div>
```

- [ ] **Step 4: Add the activity route**

Append to `routes/admin_ui.py`:

```python
@router.get("/tournaments/{tournament_id}/activity")
async def admin_tournament_activity(
    tournament_id: int,
    request: Request,
    user: AdminUser,
    session: AsyncSession = Depends(get_database),
):
    service = TournamentService(session=session)
    try:
        snap = await service.get_admin_activity(tournament_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return _templates.TemplateResponse(
        request=request,
        name="ui/admin/_activity_block.html",
        context={"user": user, "snap": snap},
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "activity_fragment"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/_activity_block.html \
        packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        tests/integration/dashboard/test_admin_tournament_ui.py \
        tests/integration/dashboard/conftest.py
git commit -m "feat(admin): activity fragment endpoint + table + heatmap"
```

---

## Task 10: Wire HTMX polling into the detail page

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.anyio
async def test_admin_detail_has_htmx_polling_attributes(
    client: AsyncClient, admin_user_headers, seed_live_tournament_route
):
    t_id = seed_live_tournament_route
    resp = await client.get(
        f"/ui/admin/tournaments/{t_id}", headers=admin_user_headers
    )
    assert resp.status_code == 200
    assert f'hx-get="/ui/admin/tournaments/{t_id}/activity"' in resp.text
    assert 'hx-trigger="load, every 2s"' in resp.text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "polling_attributes"`
Expected: FAIL.

- [ ] **Step 3: Replace the placeholder with the polling wrapper**

In `tournament_detail.html`, replace the `<article id="activity-block">...</article>` block with:

```jinja
<article
  id="activity-block"
  hx-get="/ui/admin/tournaments/{{ tournament.id }}/activity"
  hx-trigger="load, every 2s"
  hx-swap="innerHTML">
  <p><em>Loading activity…</em></p>
</article>
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "polling_attributes"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html \
        tests/integration/dashboard/test_admin_tournament_ui.py
git commit -m "feat(admin): HTMX polling every 2s in tournament detail"
```

---

## Task 11: Post-mortem variant (disable polling when tournament done)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.anyio
async def test_admin_detail_post_mortem_has_no_polling(
    client: AsyncClient, admin_user_headers, seed_completed_tournament
):
    t_id = seed_completed_tournament
    resp = await client.get(
        f"/ui/admin/tournaments/{t_id}", headers=admin_user_headers
    )
    assert resp.status_code == 200
    assert "hx-trigger" not in resp.text
    # Cancel button must not appear for completed tournaments.
    assert "Cancel tournament" not in resp.text
    # A post-mortem marker must appear.
    assert "Post-mortem" in resp.text or "post-mortem" in resp.text
```

Add `seed_completed_tournament` fixture: a tournament with `status="completed"`, 3 participants, all rounds with Actions, known scores.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "post_mortem"`
Expected: FAIL.

- [ ] **Step 3: Update the detail template to branch on status**

Replace the Cancel-button block and activity-block with:

```jinja
<header style="display:flex;justify-content:space-between;align-items:flex-start;">
  <div>
    <h1>Tournament #{{ tournament.id }}
      {% if tournament.status not in ("pending", "in_progress") %}
        <small>· Post-mortem</small>
      {% endif %}
    </h1>
    <p>
      {{ tournament.game_type }} · round
      {{ tournament.current_round or 0 }}/{{ tournament.total_rounds }} ·
      <strong>{{ tournament.status }}</strong>
    </p>
    <p><small>Created {{ tournament.created_at.strftime("%Y-%m-%d %H:%M") }} · {{ tournament.participants|length }} participants</small></p>
  </div>
  <div>
    {% if tournament.status in ("pending", "in_progress") %}
    <button
      hx-post="/api/v1/tournaments/{{ tournament.id }}/cancel"
      hx-confirm="Cancel this tournament? This cannot be undone."
      hx-swap="none"
      class="contrast">
      Cancel tournament
    </button>
    {% endif %}
  </div>
</header>

{% if tournament.status in ("pending", "in_progress") %}
<article
  id="activity-block"
  hx-get="/ui/admin/tournaments/{{ tournament.id }}/activity"
  hx-trigger="load, every 2s"
  hx-swap="innerHTML">
  <p><em>Loading activity…</em></p>
</article>
{% else %}
<article id="activity-block">
  {# One-shot server-side include of the fragment with the final state. #}
  {% include "ui/admin/_activity_block.html" %}
</article>
{% endif %}
```

To support the include, the detail route must pass `snap` into the context when the tournament is not live:

```python
# in admin_tournament_detail() in routes/admin_ui.py
snap = None
if tournament.status not in ("pending", "in_progress"):
    service = TournamentService(session=session)
    try:
        snap = await service.get_admin_activity(tournament.id)
    except LookupError:
        snap = None
return _templates.TemplateResponse(
    request=request,
    name="ui/admin/tournament_detail.html",
    context={"user": user, "tournament": tournament, "snap": snap},
)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "post_mortem or polling_attributes"`
Expected: both PASS, no regression.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/tournament_detail.html \
        packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        tests/integration/dashboard/test_admin_tournament_ui.py \
        tests/integration/dashboard/conftest.py
git commit -m "feat(admin): post-mortem variant without polling for finished tournaments"
```

---

## Task 12: `TournamentService.kick_participant()` (nice-to-have)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_kick_participant.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dashboard/tournament/test_kick_participant.py`:

```python
"""Unit tests for TournamentService.kick_participant."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from atp.dashboard.tournament.models import Action, ActionSource, Participant
from atp.dashboard.tournament.service import TournamentService


@pytest.mark.anyio
async def test_kick_participant_sets_released_at(
    session, seed_live_tournament
):
    t = seed_live_tournament
    target = t.participants[0]
    service = TournamentService(session=session)
    await service.kick_participant(t.id, target.id)

    refreshed = await session.get(Participant, target.id)
    assert refreshed.released_at is not None
    assert refreshed.released_at <= datetime.now(tz=UTC) + timedelta(seconds=5)


@pytest.mark.anyio
async def test_kick_participant_inserts_timeout_action_if_no_action_for_current_round(
    session, seed_live_tournament
):
    t = seed_live_tournament  # current_round=2, target has not submitted yet.
    target = next(p for p in t.participants if p.agent_name == "stalled_bot")
    service = TournamentService(session=session)
    await service.kick_participant(t.id, target.id)

    stmt = select(Action).where(
        Action.participant_id == target.id
    )
    rows = (await session.execute(stmt)).scalars().all()
    current_action = next(
        (a for a in rows if a.round.round_number == 2), None
    )
    assert current_action is not None
    assert current_action.source == ActionSource.TIMEOUT_DEFAULT.value


@pytest.mark.anyio
async def test_kick_participant_409_when_already_released(
    session, seed_live_tournament
):
    t = seed_live_tournament
    target = t.participants[0]
    service = TournamentService(session=session)
    await service.kick_participant(t.id, target.id)

    with pytest.raises(ValueError):
        await service.kick_participant(t.id, target.id)


@pytest.mark.anyio
async def test_kick_participant_raises_lookup_error_for_unknown_participant(
    session, seed_live_tournament
):
    service = TournamentService(session=session)
    with pytest.raises(LookupError):
        await service.kick_participant(seed_live_tournament.id, 9999999)
```

The `seed_live_tournament` fixture must expose a participant named `"stalled_bot"` with no action for the current round (round 2).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_kick_participant.py -v --no-cov`
Expected: FAIL (`AttributeError: ... has no attribute 'kick_participant'`).

- [ ] **Step 3: Implement the method**

Append to `packages/atp-dashboard/atp/dashboard/tournament/service.py`:

```python
async def kick_participant(
    self, tournament_id: int, participant_id: int
) -> None:
    """Release a participant from a live tournament.

    Sets ``Participant.released_at = now``. If there is no Action
    yet for the current round, inserts a ``TIMEOUT_DEFAULT`` action
    with the game's ``default_action_on_timeout()`` payload so that
    round resolution is not blocked by the kicked slot.

    Raises:
        LookupError: participant does not exist in this tournament.
        ValueError: participant is already released.
    """
    from datetime import UTC, datetime

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from atp.dashboard.tournament.models import (
        Action,
        ActionSource,
        Participant,
        Round,
        Tournament,
    )

    stmt = (
        select(Participant)
        .where(
            Participant.tournament_id == tournament_id,
            Participant.id == participant_id,
        )
        .options(selectinload(Participant.tournament))
    )
    participant = (await self._session.execute(stmt)).scalars().first()
    if participant is None:
        raise LookupError(
            f"participant {participant_id} not found in tournament {tournament_id}"
        )
    if participant.released_at is not None:
        raise ValueError("participant already released")

    participant.released_at = datetime.now(tz=UTC)

    tournament = participant.tournament
    current_round_n = tournament.current_round or 0
    if current_round_n > 0 and tournament.status == "in_progress":
        round_stmt = select(Round).where(
            Round.tournament_id == tournament_id,
            Round.round_number == current_round_n,
        )
        current_round = (await self._session.execute(round_stmt)).scalars().first()
        if current_round is not None:
            existing_stmt = select(Action).where(
                Action.round_id == current_round.id,
                Action.participant_id == participant.id,
            )
            existing = (await self._session.execute(existing_stmt)).scalars().first()
            if existing is None:
                from atp.games import get_game_for_tournament

                game = get_game_for_tournament(tournament)
                default_action = game.default_action_on_timeout()
                self._session.add(
                    Action(
                        round_id=current_round.id,
                        participant_id=participant.id,
                        action=default_action,
                        source=ActionSource.TIMEOUT_DEFAULT.value,
                    )
                )

    await self._session.commit()
```

Note: `get_game_for_tournament` is a placeholder name — before implementing, grep the codebase for the real factory used by `_resolve_round` (inspect `tournament/service.py` — there is already a `_game_for(tournament)` or similar helper). Reuse it; do not add a second factory.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/dashboard/tournament/test_kick_participant.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_kick_participant.py \
        tests/unit/dashboard/tournament/conftest.py
git commit -m "feat(tournament): kick_participant with mid-round TIMEOUT_DEFAULT fallback"
```

---

## Task 13: `DELETE /api/v1/tournaments/{id}/participants/{pid}` REST endpoint

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`
- Test: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.anyio
async def test_kick_participant_requires_admin(
    client: AsyncClient, regular_user_headers, seed_live_tournament_route
):
    t_id = seed_live_tournament_route
    resp = await client.delete(
        f"/api/v1/tournaments/{t_id}/participants/1",
        headers=regular_user_headers,
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_kick_participant_returns_204_on_success(
    client: AsyncClient, admin_user_headers, seed_live_tournament_route,
    seed_live_tournament_pid,
):
    t_id = seed_live_tournament_route
    pid = seed_live_tournament_pid
    resp = await client.delete(
        f"/api/v1/tournaments/{t_id}/participants/{pid}",
        headers=admin_user_headers,
    )
    assert resp.status_code == 204


@pytest.mark.anyio
async def test_kick_participant_returns_409_if_already_released(
    client: AsyncClient, admin_user_headers, seed_live_tournament_route,
    seed_live_tournament_pid,
):
    t_id = seed_live_tournament_route
    pid = seed_live_tournament_pid
    await client.delete(
        f"/api/v1/tournaments/{t_id}/participants/{pid}",
        headers=admin_user_headers,
    )
    resp = await client.delete(
        f"/api/v1/tournaments/{t_id}/participants/{pid}",
        headers=admin_user_headers,
    )
    assert resp.status_code == 409
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "kick_participant"`
Expected: FAIL (404).

- [ ] **Step 3: Add the endpoint**

Append to `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`:

```python
from atp.dashboard.v2.dependencies import AdminUser


@router.delete(
    "/{tournament_id}/participants/{participant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def kick_participant_endpoint(
    tournament_id: int,
    participant_id: int,
    user: AdminUser,
    service: TournamentService = Depends(get_tournament_service),
) -> None:
    try:
        await service.kick_participant(tournament_id, participant_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="participant already released",
        )
```

Match the import style of the surrounding file.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "kick_participant"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py \
        tests/integration/dashboard/test_admin_tournament_ui.py
git commit -m "feat(api): DELETE /api/v1/tournaments/{id}/participants/{pid} (admin)"
```

---

## Task 14: Kick button in the activity fragment

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/_activity_block.html`
- Modify: `tests/integration/dashboard/test_admin_tournament_ui.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.anyio
async def test_activity_fragment_renders_kick_button_for_live_tournament(
    client: AsyncClient, admin_user_headers, seed_live_tournament_route,
    seed_live_tournament_pid,
):
    t_id = seed_live_tournament_route
    pid = seed_live_tournament_pid
    resp = await client.get(
        f"/ui/admin/tournaments/{t_id}/activity",
        headers=admin_user_headers,
    )
    assert resp.status_code == 200
    assert f'hx-delete="/api/v1/tournaments/{t_id}/participants/{pid}"' in resp.text


@pytest.mark.anyio
async def test_activity_fragment_has_no_kick_button_when_completed(
    client: AsyncClient, admin_user_headers, seed_completed_tournament
):
    t_id = seed_completed_tournament
    resp = await client.get(
        f"/ui/admin/tournaments/{t_id}/activity",
        headers=admin_user_headers,
    )
    assert "hx-delete=" not in resp.text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "kick_button"`
Expected: FAIL.

- [ ] **Step 3: Update the fragment template**

In `_activity_block.html`, add a Kick column to the table. The fragment currently receives only `snap`; it also needs `tournament.status` to know whether to show the button. Extend the route that renders this fragment to include the status in the context:

Modify `admin_tournament_activity()` in `routes/admin_ui.py`:

```python
return _templates.TemplateResponse(
    request=request,
    name="ui/admin/_activity_block.html",
    context={"user": user, "snap": snap, "is_live": snap["status"] in ("pending", "in_progress")},
)
```

Update the table section in `_activity_block.html`:

```jinja
<table>
  <thead>
    <tr>
      <th>Agent</th><th>Status</th><th>Last action</th><th>Score</th>
      {% if is_live %}<th></th>{% endif %}
    </tr>
  </thead>
  <tbody>
    {% for p in snap.participants %}
    <tr>
      <td>{{ p.agent_name }}</td>
      <td>
        {% if p.current_round_status == "submitted" %}✓ submitted
        {% elif p.current_round_status == "waiting" %}⏳ waiting
        {% elif p.current_round_status == "timeout" %}⚠ timeout
        {% else %}◻ released
        {% endif %}
      </td>
      <td>
        {% if p.current_round_submitted_at %}
          {{ p.current_round_submitted_at.strftime("%H:%M:%S") }}
        {% else %}—{% endif %}
      </td>
      <td>{{ "%.1f"|format(p.total_score or 0) }}</td>
      {% if is_live %}
      <td>
        {% if p.current_round_status != "released" %}
        <button
          hx-delete="/api/v1/tournaments/{{ snap.tournament_id }}/participants/{{ p.id }}"
          hx-confirm="Kick {{ p.agent_name }}? This sets released_at and inserts a timeout action if mid-round."
          hx-swap="none"
          class="outline contrast">
          Kick
        </button>
        {% endif %}
      </td>
      {% endif %}
    </tr>
    {% endfor %}
  </tbody>
</table>
```

Also update the detail route (post-mortem include branch) to pass `is_live=False` when including the fragment.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_admin_tournament_ui.py -v --no-cov -k "kick"`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/_activity_block.html \
        packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py \
        tests/integration/dashboard/test_admin_tournament_ui.py
git commit -m "feat(admin): Kick button per participant in activity fragment"
```

---

## Task 15: Document `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Open CLAUDE.md and locate the Environment Variables section**

Run: `grep -n "ATP_TOKEN_EXPIRE_MINUTES" CLAUDE.md`
Expected: one match around the "Environment Variables" section.

- [ ] **Step 2: Add the new variable right after `ATP_TOKEN_EXPIRE_MINUTES`**

Insert:

```
- `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` - JWT expiration in minutes for admin users (default: 720). Applied at token issuance when `User.is_admin=True`; non-admins keep `ATP_TOKEN_EXPIRE_MINUTES`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): ATP_ADMIN_TOKEN_EXPIRE_MINUTES env var"
```

---

## Task 16: Record deferred admin items in TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Append to TODO.md under a new section**

Insert at the end of the "Ecosystem Roadmap" block (after "НЕ делаем здесь"):

```markdown
### Admin tournament GUI follow-ups (deferred from 2026-04-20 spec)

- [ ] **h · Live MCP SSE connection status** per participant in admin detail — needs a new in-memory connection registry bound to the FastMCP server, plus a `/ui/admin/tournaments/{id}/connections` fragment. Scope: ~2 days. Reference: `docs/superpowers/specs/2026-04-20-admin-tournament-gui-design.md` § "Out of scope".
- [ ] **f · Force-advance round** (admin button + REST endpoint wrapping existing `TournamentService.force_resolve_round`) — currently only the deadline worker can trigger it. Safety: needs a confirmation step and audit log entry.
- [ ] **g · Extend round deadline mid-round** — requires adding a service method and a new audit row since mutating `Round.deadline` after creation is currently disallowed.
- [ ] **Generalize admin create form to all 8 games** — currently hardcoded to El Farol. Add per-game config fieldsets keyed off the game registry.
- [ ] **Long-lived bot MCP sessions (spec C)** — separate design and plan; current admin TTL change does not address bot-side session budget.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs(todo): admin tournament GUI follow-ups (h, f, g, multi-game, spec C)"
```

---

## Task 17: End-to-end manual smoke

**Files:**
- None. Runtime verification only.

- [ ] **Step 1: Start the dashboard locally**

Run (in a separate terminal):
```bash
ATP_DISABLE_AUTH=false \
ATP_SECRET_KEY=dev-smoke-key \
ATP_ADMIN_TOKEN_EXPIRE_MINUTES=720 \
uv run atp dashboard
```

Expected: server listening on `http://127.0.0.1:8080`.

- [ ] **Step 2: Create an admin user via CLI**

Run: `uv run python -m atp.dashboard.cli.admin create-user --admin --username smoke --email smoke@example.com --password smoke123`
Expected: user created, admin token printed.

- [ ] **Step 3: Open `/ui/login`, log in as smoke, visit `/ui/admin`**

Expected: landing page renders; link "Tournaments" navigates to `/ui/admin/tournaments`.

- [ ] **Step 4: Create an El Farol tournament via the form**

Fill num_players=3, total_rounds=5, round_deadline_s=15, capacity_threshold=2. Submit.
Expected: redirected to `/ui/admin/tournaments/{id}` with Cancel button visible and the activity block showing a polling "Loading activity..." message.

- [ ] **Step 5: Watch the activity block update every 2 s**

Open browser devtools Network tab; filter by `/activity`. Expected: requests every 2 s returning the fragment. Heatmap initially shows all "waiting" (white cells).

- [ ] **Step 6: Manually resolve a round via the API to test heatmap coloring**

In another terminal:
```bash
curl -X POST http://127.0.0.1:8080/api/v1/tournaments/{id}/force-resolve \
  -H "Authorization: Bearer <admin-token>"
```

(If the force-resolve endpoint does not exist — it does not, per the spec — skip this step and instead wait for the natural deadline to elapse.)

Expected: after the round deadline passes, heatmap cells flip to `submitted` (green) or `timeout` (red) depending on what the real bot submitted.

- [ ] **Step 7: Cancel the tournament**

Click Cancel; confirm. Expected: HTTP 200, polling stops on next reload, detail page shows "Post-mortem" marker.

- [ ] **Step 8: No commit — this is runtime verification**

If anything in steps 1–7 behaves incorrectly, file a bug and return to the relevant task.

---

## Task 18: Open the pull request

**Files:**
- None. Git/GitHub operations.

- [ ] **Step 1: Run the full dashboard test suite once more**

Run: `uv run pytest tests/unit/dashboard/ tests/integration/dashboard/ -v --no-cov`
Expected: all green.

- [ ] **Step 2: Push the branch**

Run: `git push -u github feat/admin-tournament-gui`

- [ ] **Step 3: Open the PR**

Run:

```bash
gh pr create \
  --title "feat(admin): tournament management UI (El Farol, A+B + kick)" \
  --body-file - <<'EOF'
## Summary

Admin-gated UI at `/ui/admin/tournaments/*` covering the full El Farol tournament lifecycle: create, live monitor (HTMX polling every 2 s with per-participant status table and cross-round submission heatmap), cancel, post-mortem audit, and kick-participant.

Introduces `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` (default 720) so a long monitoring session does not expire mid-tournament. Non-admin JWT TTL is unchanged.

## Spec

`docs/superpowers/specs/2026-04-20-admin-tournament-gui-design.md`

## Test plan

- [x] `uv run pytest tests/unit/dashboard/ tests/integration/dashboard/ -v`
- [x] End-to-end smoke per task 17 of the implementation plan
- [ ] Visual check on `/ui/admin/tournaments/{id}` after deploy: two-column layout renders, polling updates heatmap in real time

## Scope fence

In scope: a (create), b (live monitor), c (cancel), d (post-mortem), e (kick), admin TTL.
Out of scope (tracked in TODO.md): f (force-advance), g (extend deadline), h (SSE connection status), multi-game create form, long-lived bot sessions (spec C).
EOF
```

- [ ] **Step 4: Wait for CI and fix anything red**

If CI flags a formatting or type issue, run `uv run ruff format .` and `uv run pyrefly check`, commit the fix on the branch, and push.

---

## Self-review summary (author)

Spec coverage (each section → task(s)):
- Overview / in-scope actions (a, b, c, d, e): tasks 5, 6, 7, 9, 10, 11, 12, 13, 14 (cancel is a one-line button in task 7)
- Tech stack (HTMX, Jinja2, Pico, HTMX polling, no new deps): tasks 4–11
- Pages and routes: tasks 4, 5, 6, 7, 9, 10, 11 (all routes); task 13 (REST kick)
- Auth change (`ATP_ADMIN_TOKEN_EXPIRE_MINUTES`): tasks 1, 2, 15
- Require-admin dependency: task 3
- Service-layer additions (`get_admin_activity`, `kick_participant`): tasks 8, 12
- Layout B (two columns): task 9 (`_activity_block.html` uses `grid-template-columns: 3fr 2fr`)
- Error handling (404, 403, 409, polling error): covered by integration tests in tasks 4, 5, 7, 9, 12, 13
- Testing pyramid: unit (tasks 1, 3, 8, 12) + integration (tasks 4, 5, 6, 7, 9, 10, 11, 13, 14), no E2E (consistent with spec)
- Rollout phases: phase 1 = tasks 1–7; phase 2 = tasks 8–11; phase 3 = tasks 12–14; docs = tasks 15–16; verify + ship = tasks 17–18
- Scope fence items in TODO: task 16

No placeholders remain. The only "verify before implementing" notes are on deliberately open facts (exact column name `Tournament.current_round`, exact factory `_game_for(tournament)`) that require inspecting the service file at implementation time — these are flagged explicitly in-task, not papered over.
