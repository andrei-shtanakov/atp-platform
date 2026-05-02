# Pending Tournament Banner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-of-page banner to `/ui/tournaments/{id}` and `/ui/tournaments/{id}/live` that, while the tournament is in `pending` status, shows live registration progress (`Registered: N / num_players`) and a 1-second JS countdown to `pending_deadline`.

**Architecture:** Pure additive change. New helper `_pending_banner_context(tournament)` produces template context (4 keys). A new partial branch on `ui_tournament_detail` (`?partial=pending-banner`) returns just the wrapper for HTMX 10 s polling. Two Jinja partials (`pending_banner_wrapper.html` + `pending_banner.html`) share the body markup. One global JS interval (no per-element timers) ticks the countdown — no leak across HTMX swaps. The wrapper without `hx-trigger` self-stops polling once status flips out of pending.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x async, Jinja2, HTMX, plain ES6 JS.

**Spec:** `docs/superpowers/specs/2026-05-02-pending-tournament-banner-design.md`

---

## File Structure

### Created files

| Path | Responsibility |
|---|---|
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner.html` | Banner content: counter span + countdown span. No HTMX attributes. |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner_wrapper.html` | Outer `<div id="pending-banner">` that conditionally carries `hx-get`/`hx-trigger`/`hx-swap`. Includes `pending_banner.html` when `pending_banner_show=True`. |
| `packages/atp-dashboard/atp/dashboard/v2/static/js/pending_banner.js` | One global `setInterval` that updates every `.js-countdown` element each tick. |
| `tests/unit/dashboard/test_pending_banner_context.py` | Unit tests for `_pending_banner_context()` helper. |
| `tests/integration/dashboard/test_pending_banner.py` | Integration tests for partial endpoint, render parity, status flip, tenant filter, cache headers. |

### Modified files

| Path | Why |
|---|---|
| `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` | New helper, eager-load fix in `ui_tournament_live`, `tournament_id` added to detail context, new `?partial=pending-banner` branch with `Cache-Control: no-store`, banner context spread into both routes. |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html` | Add wrapper include above the existing `<h2>`. |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/match_detail.html` | Add conditional wrapper include at top of `{% block content %}`. |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html` | Add `<script src="/static/v2/js/pending_banner.js" defer>`. |

### Static asset note

The static dir is mounted at the URL prefix `/static/v2` from the
filesystem path `packages/atp-dashboard/atp/dashboard/v2/static/` (see
`factory.py:36` and `factory.py:264`). So the JS file lives at
`packages/atp-dashboard/atp/dashboard/v2/static/js/pending_banner.js`
(filesystem) and is served at `/static/v2/js/pending_banner.js` (URL).
The neighbouring directories `static/css/`, `static/images/`,
`static/js/` already exist; just drop the new file into `static/js/`.

---

## Task 1: Helper function + unit tests

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Create: `tests/unit/dashboard/test_pending_banner_context.py`

**Goal:** Add `_pending_banner_context(tournament)` to `routes/ui.py` and lock its behaviour with unit tests. No HTTP, no templates yet.

- [ ] **Step 1: Create the test file with the empty case**

Create `tests/unit/dashboard/test_pending_banner_context.py`:

```python
"""Unit tests for _pending_banner_context helper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import atp.dashboard.tournament.models  # noqa: F401  (register tables)
from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, Base, User
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.routes.ui import _pending_banner_context


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def _make_tournament(
    session: AsyncSession,
    *,
    status: TournamentStatus = TournamentStatus.PENDING,
    tenant_id: str = DEFAULT_TENANT_ID,
    num_players: int = 4,
) -> Tournament:
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=tenant_id,
        game_type="el_farol",
        config={"name": "T"},
        status=status,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=num_players,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
        pending_deadline=starts + timedelta(minutes=15),
    )
    session.add(t)
    await session.flush()
    return t


@pytest.mark.anyio
async def test_pending_default_tenant_returns_full_context(session: AsyncSession):
    t = await _make_tournament(session)
    await session.commit()
    # Refresh to ensure participants relationship is empty list, not unloaded.
    await session.refresh(t, attribute_names=["participants"])

    ctx = _pending_banner_context(t)
    assert ctx["pending_banner_show"] is True
    assert ctx["pending_planned_count"] == 4
    assert ctx["pending_registered_count"] == 0
    # ISO must carry a UTC marker so the browser parses it correctly.
    assert (
        "+00:00" in ctx["pending_deadline_iso"]
        or ctx["pending_deadline_iso"].endswith("Z")
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_pending_banner_context.py::test_pending_default_tenant_returns_full_context -v`
Expected: ImportError on `_pending_banner_context` (the helper does not exist yet).

- [ ] **Step 3: Add the helper to `routes/ui.py`**

Open `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`. Find the `from datetime import ...` import block near the top. Make sure `UTC` is imported:

```python
from datetime import UTC, datetime
```

(if `UTC` is not already there — file already imports `datetime`).

Then near the top of the file, after the existing imports and before the first `@router.get` decorator, add the helper. Place it next to other private helpers like `_templates` and `_get_ui_user` (around lines 36-50):

```python
def _pending_banner_context(tournament: "Tournament") -> dict[str, Any]:
    """Return template context fragments for the pending banner.

    Returns ``{"pending_banner_show": False}`` when the banner is not
    applicable; otherwise returns the full set of four keys
    (``pending_banner_show``, ``pending_deadline_iso``,
    ``pending_registered_count``, ``pending_planned_count``).

    NB: ``Tournament.pending_deadline`` is declared as
    ``Mapped[datetime]`` with no ``timezone=True``, so the value comes
    back tz-naive. We coerce to UTC before ``isoformat()`` — otherwise
    the ISO string lacks a ``Z`` / offset suffix and the browser's
    ``new Date(...)`` parses it in the user's local timezone, producing
    a countdown wrong by the user-vs-server clock skew. Same pattern
    is used at lines ~877-887 for ``starts_at``.
    """
    from atp.dashboard.tournament.models import TournamentStatus

    if (
        tournament.status != TournamentStatus.PENDING
        or tournament.tenant_id != DEFAULT_TENANT_ID
    ):
        return {"pending_banner_show": False}
    deadline = tournament.pending_deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    registered = sum(
        1 for p in tournament.participants if p.released_at is None
    )
    return {
        "pending_banner_show": True,
        "pending_deadline_iso": deadline.isoformat(),
        "pending_registered_count": registered,
        "pending_planned_count": tournament.num_players,
    }
```

`Any` is already imported at the top of `ui.py:14` (`from typing import Any`). `DEFAULT_TENANT_ID` should already be imported as well; if not, add `from atp.dashboard.models import DEFAULT_TENANT_ID` at the top. `Tournament` is referenced via string annotation to avoid an extra top-level import; runtime usage doesn't need it.

- [ ] **Step 4: Run the first test to confirm it passes**

Run: `uv run pytest tests/unit/dashboard/test_pending_banner_context.py::test_pending_default_tenant_returns_full_context -v`
Expected: PASS.

- [ ] **Step 5: Add the remaining four unit tests**

Append to `tests/unit/dashboard/test_pending_banner_context.py`:

```python
async def _make_user(session: AsyncSession, username: str) -> User:
    u = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(u)
    await session.flush()
    return u


async def _make_agent(
    session: AsyncSession, *, owner: User, name: str
) -> Agent:
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name=name,
        agent_type="tournament",
        owner_id=owner.id,
        purpose="tournament",
    )
    session.add(a)
    await session.flush()
    return a


@pytest.mark.anyio
async def test_active_status_returns_show_false(session: AsyncSession):
    t = await _make_tournament(session, status=TournamentStatus.ACTIVE)
    await session.commit()
    await session.refresh(t, attribute_names=["participants"])
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_completed_status_returns_show_false(session: AsyncSession):
    t = await _make_tournament(session, status=TournamentStatus.COMPLETED)
    await session.commit()
    await session.refresh(t, attribute_names=["participants"])
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_cancelled_status_returns_show_false(session: AsyncSession):
    t = await _make_tournament(session, status=TournamentStatus.CANCELLED)
    await session.commit()
    await session.refresh(t, attribute_names=["participants"])
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_non_default_tenant_returns_show_false(session: AsyncSession):
    t = await _make_tournament(session, tenant_id="other-tenant")
    await session.commit()
    await session.refresh(t, attribute_names=["participants"])
    assert _pending_banner_context(t) == {"pending_banner_show": False}


@pytest.mark.anyio
async def test_counter_excludes_released_participants(session: AsyncSession):
    t = await _make_tournament(session, num_players=4)
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    carol = await _make_user(session, "carol")
    a = await _make_agent(session, owner=alice, name="a")
    b = await _make_agent(session, owner=bob, name="b")
    c = await _make_agent(session, owner=carol, name="c")
    # Three participants; one released.
    p1 = Participant(
        tournament_id=t.id, user_id=alice.id, agent_id=a.id, agent_name="a"
    )
    p2 = Participant(
        tournament_id=t.id, user_id=bob.id, agent_id=b.id, agent_name="b"
    )
    p3 = Participant(
        tournament_id=t.id,
        user_id=carol.id,
        agent_id=c.id,
        agent_name="c",
        released_at=datetime(2026, 5, 1, 12, 5, 0),
    )
    session.add_all([p1, p2, p3])
    await session.commit()
    await session.refresh(t, attribute_names=["participants"])

    ctx = _pending_banner_context(t)
    assert ctx["pending_registered_count"] == 2  # carol excluded
    assert ctx["pending_planned_count"] == 4
```

- [ ] **Step 6: Run the full unit suite**

Run: `uv run pytest tests/unit/dashboard/test_pending_banner_context.py -v`
Expected: 6 PASS.

- [ ] **Step 7: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py tests/unit/dashboard/test_pending_banner_context.py`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py tests/unit/dashboard/test_pending_banner_context.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py tests/unit/dashboard/test_pending_banner_context.py
git commit -m "feat(banner): _pending_banner_context helper + unit tests"
```

---

## Task 2: Wrapper + body Jinja partials

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner_wrapper.html`

**Goal:** Add the two Jinja templates. No route changes yet; we just need the files to exist for Task 3 to wire them up.

- [ ] **Step 1: Create the body partial**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner.html` exactly:

```jinja
{# Body of the pending banner. Rendered inside pending_banner_wrapper.html
   only when pending_banner_show is true. The wrapper owns HTMX polling. #}
<div class="pending-banner-row" style="
    background:#fffbe6;
    border:1px solid #ffe58f;
    border-radius:4px;
    padding:0.5rem 1rem;
    margin-bottom:1rem;
    display:flex;
    gap:1.5rem;
    align-items:center;
">
  <span class="pending-banner-count">
    Registered: <strong>{{ pending_registered_count }}</strong>
    / {{ pending_planned_count }}
  </span>
  <span class="js-countdown"
        data-deadline-iso="{{ pending_deadline_iso }}">—:—</span>
</div>
```

- [ ] **Step 2: Create the wrapper partial**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner_wrapper.html`:

```jinja
{# Wrapper used both at initial page render and as the HTMX swap target.
   When pending_banner_show is true, the wrapper carries hx-get/-trigger
   so it polls itself every 10 s. When false, the wrapper renders empty
   AND without hx-trigger, which gracefully stops HTMX polling once the
   tournament transitions out of pending. #}
<div id="pending-banner"
     {% if pending_banner_show %}
     hx-get="/ui/tournaments/{{ tournament_id }}?partial=pending-banner"
     hx-trigger="every 10s"
     hx-swap="outerHTML"
     {% endif %}
>
  {% if pending_banner_show %}
    {% include "ui/partials/pending_banner.html" %}
  {% endif %}
</div>
```

- [ ] **Step 3: Verify the templates parse**

Run a syntax sanity check by loading them through Jinja:

```bash
uv run python -c "
import jinja2
loader = jinja2.FileSystemLoader('packages/atp-dashboard/atp/dashboard/v2/templates')
env = jinja2.Environment(loader=loader, autoescape=True)
env.get_template('ui/partials/pending_banner.html')
env.get_template('ui/partials/pending_banner_wrapper.html')
print('ok')
"
```

Expected output: `ok`.

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner.html packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/pending_banner_wrapper.html
git commit -m "feat(banner): pending banner wrapper + body partials"
```

---

## Task 3: Wire helper into `ui_tournament_detail` + new partial branch

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
- Create: `tests/integration/dashboard/test_pending_banner.py`

**Goal:** Render the banner on `/ui/tournaments/{id}` and add the new `?partial=pending-banner` branch on the same route. Detail-only — live-route wiring lands in Task 4.

- [ ] **Step 1: Add tournament_id + banner context to detail route**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, find the `context = {...}` dict in `ui_tournament_detail` (around line 1368). Add two new keys:

```python
    context = {
        "active_page": "tournaments",
        "tournament": tournament,
        "tournament_id": tournament_id,
        "creator_name": creator_name,
        "is_admin": is_admin,
        "cancelled_by_name": cancelled_by_name,
        "sorted_rounds": sorted_rounds,
        "sorted_participants": sorted_participants,
        "participant_map": participant_map,
        "completed_rounds": completed_rounds,
        "timeline": timeline,
        "user": user,
        "visible_reasoning_action_ids": visible_reasoning_action_ids,
        "cards_match_id": cards_match_id,
        **_pending_banner_context(tournament),
    }
```

- [ ] **Step 2: Add the `?partial=pending-banner` branch**

In the same handler, find the existing `partial = request.query_params.get("partial")` block (around line 1384) and the `if partial == "live":` branch. Add a new branch before that block:

```python
    partial = request.query_params.get("partial")
    if partial == "pending-banner":
        # Banner-only partial. The wrapper template needs only the path
        # id; we deliberately do NOT pass the full Tournament row, so
        # the contract stays narrow and grep-friendly.
        banner_ctx = {
            "tournament_id": tournament_id,
            **_pending_banner_context(tournament),
        }
        response = _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/pending_banner_wrapper.html",
            context=banner_ctx,
        )
        # Counter must NOT be served from BFCache or any intermediary.
        response.headers["Cache-Control"] = "no-store"
        return response
    if partial == "live":
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/tournament_live.html",
            context=context,
        )
```

- [ ] **Step 3: Add the include to `tournament_detail.html`**

Open `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`. Find the `{% block content %}` line (line 11). Immediately after it, add the include:

```jinja
{% block content %}
{% if pending_banner_show is defined %}
  {% include "ui/partials/pending_banner_wrapper.html" %}
{% endif %}
<h2>{{ name }}{% if show_id_badge %} <small style="color:#888;font-weight:normal;font-size:0.65em">#{{ t.id }}</small>{% endif %}</h2>
```

- [ ] **Step 4: Write the first integration test (initial page render)**

Create `tests/integration/dashboard/test_pending_banner.py`:

```python
"""HTTP-level tests for the pending tournament banner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, User
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def client(test_database: Database) -> AsyncIterator[AsyncClient]:
    app = create_test_app()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_pending_tournament(
    session: AsyncSession,
    *,
    tenant_id: str = DEFAULT_TENANT_ID,
    num_players: int = 4,
    registered: int = 0,
    deadline_minutes: int = 15,
) -> int:
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=tenant_id,
        game_type="el_farol",
        config={"name": "T"},
        status=TournamentStatus.PENDING,
        starts_at=starts,
        ends_at=starts + timedelta(minutes=10),
        num_players=num_players,
        total_rounds=5,
        round_deadline_s=30,
        join_token=None,
        pending_deadline=starts + timedelta(minutes=deadline_minutes),
    )
    session.add(t)
    await session.flush()
    for i in range(registered):
        u = User(
            username=f"u{i}",
            email=f"u{i}@e.com",
            hashed_password="x",
            is_admin=False,
            is_active=True,
        )
        session.add(u)
        await session.flush()
        a = Agent(
            tenant_id=DEFAULT_TENANT_ID,
            name=f"agent{i}",
            agent_type="tournament",
            owner_id=u.id,
            purpose="tournament",
        )
        session.add(a)
        await session.flush()
        session.add(
            Participant(
                tournament_id=t.id,
                user_id=u.id,
                agent_id=a.id,
                agent_name=f"agent{i}",
            )
        )
    await session.commit()
    return t.id


@pytest.mark.anyio
async def test_detail_page_renders_banner_for_pending_tournament(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=2)
    r = await client.get(f"/ui/tournaments/{tid}")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert 'hx-trigger="every 10s"' in r.text
    assert "Registered:" in r.text
    assert "<strong>2</strong>" in r.text
    assert "/ 4" in r.text
    # Wrapper URL must contain the tournament id, not be empty.
    assert f'/ui/tournaments/{tid}?partial=pending-banner' in r.text
    # ISO string must carry a UTC marker.
    assert ("+00:00" in r.text) or ("Z\"" in r.text)
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/integration/dashboard/test_pending_banner.py::test_detail_page_renders_banner_for_pending_tournament -v`
Expected: PASS.

- [ ] **Step 6: Add the partial-endpoint test**

Append to `tests/integration/dashboard/test_pending_banner.py`:

```python
@pytest.mark.anyio
async def test_partial_endpoint_returns_wrapper_with_hx_trigger_and_no_store(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=1)
    r = await client.get(f"/ui/tournaments/{tid}?partial=pending-banner")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert 'hx-trigger="every 10s"' in r.text
    assert "<strong>1</strong>" in r.text
    assert r.headers["Cache-Control"] == "no-store"


@pytest.mark.anyio
async def test_partial_endpoint_after_status_flip_drops_hx_trigger(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """Race window: pending → active between two HTMX pulls."""
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=2)

    r1 = await client.get(f"/ui/tournaments/{tid}?partial=pending-banner")
    assert 'hx-trigger="every 10s"' in r1.text

    # Flip status; the next pull must return an empty wrapper (no
    # hx-trigger, no inner content) — this is the gracefully-retire
    # swap the design relies on.
    from sqlalchemy import update

    async with test_database.session() as s:
        await s.execute(
            update(Tournament)
            .where(Tournament.id == tid)
            .values(status=TournamentStatus.ACTIVE)
        )
        await s.commit()

    r2 = await client.get(f"/ui/tournaments/{tid}?partial=pending-banner")
    assert r2.status_code == 200
    assert 'id="pending-banner"' in r2.text
    assert "hx-trigger" not in r2.text
    assert "Registered:" not in r2.text


@pytest.mark.anyio
async def test_detail_page_non_default_tenant_renders_empty_wrapper(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(
        db_session, tenant_id="other-tenant", num_players=4
    )
    r = await client.get(f"/ui/tournaments/{tid}")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert "hx-trigger" not in r.text
    assert "Registered:" not in r.text
```

- [ ] **Step 7: Run all integration tests**

Run: `uv run pytest tests/integration/dashboard/test_pending_banner.py -v`
Expected: 4 PASS.

- [ ] **Step 8: Sanity — existing tournament_detail tests still green**

Run: `uv run pytest tests/integration/dashboard/test_tournament_detail_ui.py -q`
Expected: existing tests pass.

- [ ] **Step 9: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py tests/integration/dashboard/test_pending_banner.py`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py tests/integration/dashboard/test_pending_banner.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html tests/integration/dashboard/test_pending_banner.py
git commit -m "feat(banner): wire banner into tournament detail page + partial endpoint"
```

---

## Task 4: Live route eager-load fix + banner include in `match_detail.html`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/match_detail.html`
- Modify: `tests/integration/dashboard/test_pending_banner.py`

**Goal:** Convert `ui_tournament_live`'s `session.get(...)` to a `select(...).options(selectinload(Tournament.participants))` so `tournament.participants` access doesn't raise `MissingGreenlet`. Then spread banner context, add the conditional include, and add a render-parity test that asserts both pages contain the same wrapper URL.

- [ ] **Step 1: Eager-load participants in `ui_tournament_live`**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, find `ui_tournament_live` (around line 771). Replace the line:

```python
    tournament = await session.get(Tournament, tournament_id)
```

with:

```python
    result = await session.execute(
        select(Tournament)
        .options(selectinload(Tournament.participants))
        .where(Tournament.id == tournament_id)
    )
    tournament = result.scalar_one_or_none()
```

`select` and `selectinload` are already imported at the top of the file (lines 18-20).

- [ ] **Step 2: Spread banner context into the live route's context dict**

In the same handler, find the `context: dict[str, Any] = { ... }` initialization (around line 809). Below it, after the visibility / not_found / non-el-farol short-circuits, find the place where the El Farol payload is built and the final `TemplateResponse` is returned. Add the banner spread when the tournament is loaded and visible. The simplest place: just before the final `return _templates(request).TemplateResponse(...)` for the live page, mutate `context` in place:

```python
    context.update(_pending_banner_context(tournament))
```

Insert this line right after `current_round_deadline_ms` is computed (around line 888) and before the final `TemplateResponse`. Search for the final `return _templates(...)` at the end of the function and place it right before that.

If the function has multiple return paths (e.g. not_found placeholder branches that return earlier), the early-return paths do NOT get banner context — that's intentional, and the template's `{% if pending_banner_show is defined %}` guard handles it.

- [ ] **Step 3: Add the conditional include to `match_detail.html`**

Open `packages/atp-dashboard/atp/dashboard/v2/templates/ui/match_detail.html`. Find `{% block content %}` (line 5). Immediately after it (and after the existing `<link rel="stylesheet" href="/static/v2/css/el_farol.css">` line), add the conditional include:

```jinja
{% block content %}
<link rel="stylesheet" href="/static/v2/css/el_farol.css">

{% if pending_banner_show is defined %}
  {% include "ui/partials/pending_banner_wrapper.html" %}
{% endif %}

<div style="padding:8px 16px"><a href="{{ back_link_href }}">← {{ back_link_label }}</a></div>
```

The `is defined` guard is critical: the same template is used by `/ui/matches/{match_id}` (the post-completion replay route) which has no tournament context. Without the guard the template would render an empty wrapper for replay pages.

- [ ] **Step 4: Add the render-parity test**

Append to `tests/integration/dashboard/test_pending_banner.py`:

```python
@pytest.mark.anyio
async def test_live_page_renders_banner_with_correct_wrapper_url(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """Regression: wrapper template references tournament_id (not
    tournament.id from a non-existent ORM-row context). If the live
    route forgets to put tournament_id in the template context, the
    URL would render as '/ui/tournaments/?partial=pending-banner' and
    HTMX would 404-loop."""
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=2)
    r = await client.get(f"/ui/tournaments/{tid}/live")
    assert r.status_code == 200
    assert 'id="pending-banner"' in r.text
    assert f'/ui/tournaments/{tid}?partial=pending-banner' in r.text
    assert "/ui/tournaments/?partial=pending-banner" not in r.text


@pytest.mark.anyio
async def test_detail_and_live_pages_have_identical_wrapper_url(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """Both pages must point HTMX at the same partial URL — that's the
    contract for 'one partial endpoint serves both hosts'."""
    tid = await _seed_pending_tournament(db_session, num_players=4, registered=1)
    r_detail = await client.get(f"/ui/tournaments/{tid}")
    r_live = await client.get(f"/ui/tournaments/{tid}/live")
    expected_url = f'/ui/tournaments/{tid}?partial=pending-banner'
    assert expected_url in r_detail.text
    assert expected_url in r_live.text


@pytest.mark.anyio
async def test_match_detail_replay_does_not_render_banner(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    """The same match_detail.html template serves /ui/matches/{id} for
    replay. It must NOT render the banner wrapper there because the
    replay route has no tournament context. The {% if ... is defined %}
    guard is what protects this."""
    r = await client.get("/ui/matches/nonexistent-match")
    assert r.status_code == 200  # placeholder page
    assert 'id="pending-banner"' not in r.text
```

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest tests/integration/dashboard/test_pending_banner.py -v`
Expected: 7 PASS (4 from Task 3 + 3 new).

- [ ] **Step 6: Sanity — existing live & match tests still green**

Run: `uv run pytest tests/integration/dashboard/test_match_detail_ui.py tests/integration/dashboard/tournament/test_live_dashboard_ui.py -q`
Expected: existing tests pass — the eager-load change must not break them.

- [ ] **Step 7: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/match_detail.html tests/integration/dashboard/test_pending_banner.py
git commit -m "feat(banner): wire banner into live page + eager-load participants"
```

---

## Task 5: JS countdown + script tag

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/static/js/pending_banner.js`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`
- Modify: `tests/integration/dashboard/test_pending_banner.py`

**Goal:** Ship the 1 s JS-tick countdown and load it from every page via `base_ui.html`. One global `setInterval` queries the DOM each tick — no per-element timers, no leaks across HTMX swaps.

- [ ] **Step 1: Create the JS file**

Create `packages/atp-dashboard/atp/dashboard/v2/static/js/pending_banner.js` (filesystem path; the URL it serves at is `/static/v2/js/pending_banner.js`):

```javascript
// Pending tournament banner countdown.
//
// ONE global setInterval polls the DOM every second and updates every
// .js-countdown[data-deadline-iso] element. This avoids the
// per-element interval-leak pitfall: with hx-swap="outerHTML" the old
// span detaches from the document, but a per-element setInterval keeps
// firing on the detached node — leaking +1 timer every 10 s. A single
// global interval is unaffected by swaps; querySelectorAll simply
// returns the current set of elements after each swap.
(function () {
  function tickAll() {
    var els = document.querySelectorAll(
      ".js-countdown[data-deadline-iso]"
    );
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var deadlineMs = new Date(el.dataset.deadlineIso).getTime();
      var remainingMs = Math.max(0, deadlineMs - Date.now());
      var totalSec = Math.floor(remainingMs / 1000);
      var h = Math.floor(totalSec / 3600);
      var m = Math.floor((totalSec % 3600) / 60);
      var s = totalSec % 60;
      // Multi-hour deadlines render as "Hh Mm Ss"; sub-hour as "M:SS"
      // so a 5-minute window stays visually compact.
      el.textContent = h > 0
        ? h + "h " + m + "m " + s + "s"
        : m + ":" + String(s).padStart(2, "0");
    }
  }
  document.addEventListener("DOMContentLoaded", function () {
    tickAll();
    setInterval(tickAll, 1000);
  });
})();
```

The IIFE keeps helpers off the global namespace. Plain ES6 (no build step). Tested in modern browsers — `padStart` is standard since ES2017, supported everywhere ATP runs.

- [ ] **Step 2: Add the script tag to `base_ui.html`**

Open `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`. Find the `<head>` section (lines 3-10) — there's already a `<script src="https://unpkg.com/htmx.org@2.0.4"></script>` line. Add the new script tag immediately after it:

```html
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="/static/v2/js/pending_banner.js" defer></script>
</head>
```

`defer` ensures the script doesn't block parsing and runs after the DOM is ready.

- [ ] **Step 3: Add a test that the script tag is present on a banner page**

Append to `tests/integration/dashboard/test_pending_banner.py`:

```python
@pytest.mark.anyio
async def test_pending_banner_js_loaded_on_detail_page(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    client: AsyncClient,
):
    tid = await _seed_pending_tournament(db_session, num_players=4)
    r = await client.get(f"/ui/tournaments/{tid}")
    assert r.status_code == 200
    assert "/static/v2/js/pending_banner.js" in r.text
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/integration/dashboard/test_pending_banner.py::test_pending_banner_js_loaded_on_detail_page -v`
Expected: PASS.

- [ ] **Step 5: Sanity — full integration suite**

Run: `uv run pytest tests/integration/dashboard/test_pending_banner.py -v`
Expected: 8 PASS.

- [ ] **Step 6: Type-check + lint**

Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html tests/integration/dashboard/test_pending_banner.py`
Expected: clean. (No pyrefly/ruff for the JS file — the project has no JS lint stack.)

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/static/js/pending_banner.js packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html tests/integration/dashboard/test_pending_banner.py
git commit -m "feat(banner): JS countdown via single global interval"
```

---

## Final verification

- [ ] **Step 1: Run the entire banner test surface**

Run: `uv run pytest tests/unit/dashboard/test_pending_banner_context.py tests/integration/dashboard/test_pending_banner.py -v`
Expected: 6 unit + 8 integration = 14 PASS.

- [ ] **Step 2: Sanity across neighboring suites**

Run: `uv run pytest tests/integration/dashboard/test_tournament_detail_ui.py tests/integration/dashboard/test_match_detail_ui.py tests/integration/dashboard/tournament/test_live_dashboard_ui.py -q`
Expected: existing tests pass (no regressions from the eager-load change or new context keys).

- [ ] **Step 3: Type-check + lint everything**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py tests/unit/dashboard/test_pending_banner_context.py tests/integration/dashboard/test_pending_banner.py`
Run: `uv run ruff format .`
Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Coverage check on new code**

Run: `uv run pytest tests/unit/dashboard/test_pending_banner_context.py tests/integration/dashboard/test_pending_banner.py --cov=atp.dashboard.v2.routes.ui --cov-report=term-missing | grep -E "pending_banner|TOTAL"`
Confirm the helper and partial branch both show ≥80 % coverage on their lines.

- [ ] **Step 5: Local browser smoke (optional)**

Start the dashboard locally (`uv run atp dashboard`), open `/ui/tournaments/new`, create a tournament with `num_players=2` (so it stays in pending until a second bot joins or the deadline hits), and visit `/ui/tournaments/{id}` in the browser. Verify:

- The yellow banner appears at the top.
- The countdown ticks every second.
- The page polls every 10 s (DevTools Network tab shows requests to `?partial=pending-banner`).
- After joining a second bot or letting the deadline trigger the shrink, the next pull returns an empty wrapper and the banner disappears.

- [ ] **Step 6: Push the branch**

```bash
git push -u origin feat/pending-banner
```
