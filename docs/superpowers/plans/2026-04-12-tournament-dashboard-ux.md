# Tournament Dashboard UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/ui/tournaments` list and `/ui/tournaments/{id}` detail pages with scoreboard, round history, cancel reasons, admin event timeline, and HTMX polling.

**Architecture:** New routes in `ui.py` query Tournament/Participant/Round/Action ORM models with `selectinload`. Jinja2 templates extend `base_ui.html`. HTMX `?partial=live` pattern for active tournament polling. Admin timeline reconstructed from DB timestamps.

**Tech Stack:** FastAPI, SQLAlchemy (async), Jinja2, HTMX 2.0.4, Pico CSS 2

**Spec:** `docs/superpowers/specs/2026-04-12-tournament-dashboard-ux-design.md`

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` | Modify | Add `ui_tournaments()` + `ui_tournament_detail()` routes |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html` | Modify | Add "Tournaments" nav item |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/games.html` | Modify | Remove tournament table, add link |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournaments.html` | Create | Tournament list page |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html` | Create | Tournament detail page |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_list_table.html` | Create | HTMX partial for list pagination |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_live.html` | Create | Combined HTMX partial for active poll |
| `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css` | Modify | Tournament-specific styles |
| `tests/unit/dashboard/ui/test_tournament_ui.py` | Create | Route tests |

---

### Task 1: CSS — tournament-specific styles

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css`

- [ ] **Step 1: Add tournament CSS classes**

Append to the end of `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css`:

```css
/* Tournament styles */
.tournament-name {
    font-weight: 500;
    color: #7c3aed;
    text-decoration: none;
}

.tournament-name:hover {
    text-decoration: underline;
}

.game-type {
    font-size: 0.8rem;
    color: #888;
}

.score-pair {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.9rem;
    font-weight: 600;
}

.cooperate {
    color: #28a745;
    font-weight: 600;
}

.defect {
    color: #dc3545;
    font-weight: 600;
}

.cancel-reason {
    font-size: 0.8rem;
    color: #999;
    margin-top: 0.15rem;
}

.cancel-box {
    background: #fff5f5;
    border: 1px solid #fed7d7;
    border-radius: 0.5rem;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
}

.cancel-box .reason {
    font-weight: 600;
    color: #c53030;
}

.cancel-box .detail {
    color: #666;
    font-size: 0.9rem;
    margin-top: 0.35rem;
}

.tournament-meta {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
}

.admin-badge {
    font-size: 0.75rem;
    color: #7c3aed;
    background: #f3eefe;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
}

/* Event timeline */
.timeline {
    position: relative;
    padding-left: 24px;
}

.timeline::before {
    content: '';
    position: absolute;
    left: 7px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: #e0e0e0;
}

.timeline-item {
    position: relative;
    padding: 8px 0;
    font-size: 0.85rem;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -20px;
    top: 14px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid white;
}

.timeline-item .event-name {
    font-weight: 600;
    color: #333;
}

.timeline-item .event-time {
    color: #999;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}

.timeline-item .event-detail {
    color: #666;
    font-size: 0.8rem;
    margin-top: 0.15rem;
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css
git commit -m "feat(dashboard): add tournament CSS styles"
```

---

### Task 2: Nav and Games page updates

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/games.html`

- [ ] **Step 1: Add Tournaments nav item in base_ui.html**

In `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`, find:

```html
                <li><a href="/ui/games" class="{% if active_page == 'games' %}active{% endif %}">Games</a></li>
```

Add after it:

```html
                <li><a href="/ui/tournaments" class="{% if active_page == 'tournaments' %}active{% endif %}">Tournaments</a></li>
```

- [ ] **Step 2: Replace tournament table in games.html**

Replace the entire second `<section>` block in `packages/atp-dashboard/atp/dashboard/v2/templates/ui/games.html` (lines 39-68, the Tournaments section) with:

```html
<section>
    <h3>Tournaments</h3>
    <p><a href="/ui/tournaments">View Tournaments →</a></p>
</section>
```

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/games.html
git commit -m "feat(dashboard): add Tournaments nav item, link from Games page"
```

---

### Task 3: Tournament list route + template

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournaments.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_list_table.html`

- [ ] **Step 1: Write failing test**

Create `tests/unit/dashboard/ui/__init__.py` (empty file) and `tests/unit/dashboard/ui/test_tournament_ui.py`:

```python
"""Tests for tournament UI routes."""

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
async def test_tournament_list_returns_200(client: AsyncClient):
    resp = await client.get("/ui/tournaments")
    assert resp.status_code == 200
    assert "Tournaments" in resp.text


@pytest.mark.anyio
async def test_tournament_list_partial_returns_200(client: AsyncClient):
    resp = await client.get("/ui/tournaments?partial=1")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_tournament_detail_404_for_missing(client: AsyncClient):
    resp = await client.get("/ui/tournaments/99999")
    assert resp.status_code == 404
    assert "Not Found" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/dashboard/ui/test_tournament_ui.py -v -x`
Expected: FAIL (routes don't exist yet — 404 or starlette routing error)

- [ ] **Step 3: Add tournament list route to ui.py**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, add these imports at the top (after existing imports):

```python
from sqlalchemy.orm import selectinload
```

Then add the following route after the `ui_games` function (after line 239):

```python
@router.get("/tournaments", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tournaments(
    request: Request,
    session: DBSession,
    page: int = 1,
) -> HTMLResponse:
    """Render tournament list page."""
    from atp.dashboard.models import User
    from atp.dashboard.tournament.models import Tournament

    per_page = 50
    offset = (page - 1) * per_page

    result = await session.execute(
        select(func.count(Tournament.id))
    )
    total = result.scalar() or 0

    result = await session.execute(
        select(Tournament)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds),
        )
        .order_by(Tournament.id.desc())
        .limit(per_page)
        .offset(offset)
    )
    tournaments = result.scalars().all()

    # Batch-load creator usernames
    creator_ids = {t.created_by for t in tournaments if t.created_by}
    creators: dict[int, str] = {}
    if creator_ids:
        user_result = await session.execute(
            select(User).where(User.id.in_(creator_ids))
        )
        creators = {u.id: u.username for u in user_result.scalars()}

    total_pages = (total + per_page - 1) // per_page

    template_name = "ui/tournaments.html"
    partial = request.query_params.get("partial")
    if partial:
        template_name = "ui/partials/tournament_list_table.html"

    return _templates(request).TemplateResponse(
        request=request,
        name=template_name,
        context={
            "active_page": "tournaments",
            "tournaments": tournaments,
            "creators": creators,
            "page": page,
            "total_pages": total_pages,
            "total": total,
        },
    )
```

- [ ] **Step 4: Create tournament list partial template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_list_table.html`:

```html
{% for t in tournaments %}
{% set name = (t.config or {}).get("name", "Tournament #" ~ t.id) %}
{% set active_players = t.participants | selectattr("released_at", "none") | list | length %}
{% set total_players = t.participants | length %}
{% set completed_rounds = t.rounds | selectattr("status", "equalto", "completed") | list | length %}
{% set scores = t.participants | sort(attribute="id") %}
<tr>
    <td>
        <a href="/ui/tournaments/{{ t.id }}" class="tournament-name">{{ name }}</a>
        <div class="game-type">{{ t.game_type }}</div>
    </td>
    <td>
        {% if t.status in ("pending", "active") %}
        {{ active_players }} / {{ t.num_players }}
        {% else %}
        {{ total_players }} / {{ t.num_players }}
        {% endif %}
    </td>
    <td>
        {% if completed_rounds > 0 %}
        {{ completed_rounds }} / {{ t.total_rounds }}
        {% else %}
        — / {{ t.total_rounds }}
        {% endif %}
    </td>
    <td class="score-pair">
        {% if scores | length == 2 %}
            {% set s0 = scores[0].total_score %}
            {% set s1 = scores[1].total_score %}
            {% if s0 is not none or s1 is not none %}
                {{ s0 if s0 is not none else "—" }} : {{ s1 if s1 is not none else "—" }}
            {% else %}
            —
            {% endif %}
        {% elif scores | length > 2 %}
            {% set best = scores | map(attribute="total_score") | reject("none") | sort(reverse=true) | first %}
            {% if best is defined %}{{ best }} (best){% else %}—{% endif %}
        {% else %}
        —
        {% endif %}
    </td>
    <td>
        {% if t.status == "completed" %}
        <span class="status-badge" style="background:#28a745">completed</span>
        {% elif t.status == "active" %}
        <span class="status-badge" style="background:#007bff">active</span>
        {% elif t.status == "pending" %}
        <span class="status-badge" style="background:#ffc107;color:#333">pending</span>
        {% elif t.status == "cancelled" %}
        <span class="status-badge" style="background:#dc3545">cancelled</span>
        {% if t.cancelled_reason %}
        <div class="cancel-reason">
            {% if t.cancelled_reason.value == "pending_timeout" %}Expired before full roster
            {% elif t.cancelled_reason.value == "admin_action" %}Cancelled by admin
            {% elif t.cancelled_reason.value == "abandoned" %}All participants left
            {% endif %}
        </div>
        {% endif %}
        {% else %}
        <span class="status-badge" style="background:#6c757d">{{ t.status }}</span>
        {% endif %}
    </td>
    <td style="font-size:0.85rem;color:#666">{{ creators.get(t.created_by, "—") }}</td>
    <td style="font-size:0.85rem;color:#666">
        {{ t.created_at.strftime("%b %d, %H:%M") if t.created_at else "—" }}
    </td>
</tr>
{% endfor %}
```

- [ ] **Step 5: Create tournament list page template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournaments.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}Tournaments - ATP Platform{% endblock %}

{% block content %}
<h2>Tournaments</h2>
<p>{{ total }} tournament{{ "s" if total != 1 else "" }}</p>

<table>
    <thead>
        <tr>
            <th>Tournament</th>
            <th>Players</th>
            <th>Rounds</th>
            <th>Scores</th>
            <th>Status</th>
            <th>Created by</th>
            <th>Created</th>
        </tr>
    </thead>
    <tbody id="tournament-table-body"
           hx-get="/ui/tournaments?partial=1&page={{ page }}"
           hx-trigger="none">
        {% include "ui/partials/tournament_list_table.html" %}
    </tbody>
</table>

{% if total_pages > 1 %}
<nav style="margin-top:1rem; display:flex; gap:0.5rem; justify-content:center;">
    {% for p in range(1, total_pages + 1) %}
    <a href="/ui/tournaments?page={{ p }}"
       hx-get="/ui/tournaments?partial=1&page={{ p }}"
       hx-target="#tournament-table-body"
       hx-swap="innerHTML"
       {% if p == page %}style="font-weight:bold"{% endif %}>{{ p }}</a>
    {% endfor %}
</nav>
{% endif %}
{% endblock %}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/dashboard/ui/test_tournament_ui.py -v -x`
Expected: 3 tests PASS (list 200, partial 200, detail 404)

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournaments.html
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_list_table.html
git add tests/unit/dashboard/ui/
git commit -m "feat(dashboard): tournament list page with pagination"
```

---

### Task 4: Tournament detail route + template

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_live.html`
- Modify: `tests/unit/dashboard/ui/test_tournament_ui.py`

- [ ] **Step 1: Write failing test**

Add to `tests/unit/dashboard/ui/test_tournament_ui.py`:

```python
from atp.dashboard.tournament.models import (
    Action,
    ActionSource,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.models import User
from datetime import datetime, timedelta


async def _seed_tournament(client: AsyncClient):
    """Create a completed 2-player, 2-round PD tournament in the test DB."""
    app = client._transport.app  # type: ignore[attr-defined]
    from atp.dashboard.database import async_session_factory

    async with async_session_factory()() as session:
        user = User(username="test_admin", email="a@test.com", is_admin=True)
        session.add(user)
        await session.flush()

        now = datetime.utcnow()
        t = Tournament(
            game_type="prisoners_dilemma",
            config={"name": "Test PD"},
            status=TournamentStatus.COMPLETED,
            num_players=2,
            total_rounds=2,
            round_deadline_s=30,
            created_by=user.id,
            created_at=now - timedelta(minutes=5),
            starts_at=now - timedelta(minutes=4),
            ends_at=now,
            pending_deadline=now,
        )
        session.add(t)
        await session.flush()

        p1 = Participant(
            tournament_id=t.id,
            user_id=user.id,
            agent_name="alice",
            total_score=6.0,
        )
        p2_user = User(username="bot_bob", email="b@test.com")
        session.add(p2_user)
        await session.flush()
        p2 = Participant(
            tournament_id=t.id,
            user_id=p2_user.id,
            agent_name="bob",
            total_score=6.0,
        )
        session.add_all([p1, p2])
        await session.flush()

        for rn in (1, 2):
            r = Round(
                tournament_id=t.id,
                round_number=rn,
                status=RoundStatus.COMPLETED,
                started_at=now - timedelta(minutes=4 - rn),
            )
            session.add(r)
            await session.flush()
            for p in (p1, p2):
                session.add(
                    Action(
                        round_id=r.id,
                        participant_id=p.id,
                        action_data={"choice": "cooperate"},
                        submitted_at=now,
                        payoff=3.0,
                        source=ActionSource.SUBMITTED,
                    )
                )
        await session.commit()
        return t.id


@pytest.mark.anyio
async def test_tournament_detail_returns_200(client: AsyncClient):
    tid = await _seed_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    assert "Test PD" in resp.text
    assert "alice" in resp.text
    assert "bob" in resp.text


@pytest.mark.anyio
async def test_tournament_detail_shows_round_history(client: AsyncClient):
    tid = await _seed_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    assert "cooperate" in resp.text
    assert "3.0" in resp.text or "3" in resp.text


@pytest.mark.anyio
async def test_tournament_detail_partial_live(client: AsyncClient):
    tid = await _seed_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}?partial=live")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/dashboard/ui/test_tournament_ui.py::test_tournament_detail_returns_200 -v -x`
Expected: FAIL (route doesn't exist — 404 or Method Not Allowed)

- [ ] **Step 3: Add tournament detail route to ui.py**

Add this route after the `ui_tournaments` function in `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`:

```python
@router.get("/tournaments/{tournament_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tournament_detail(
    request: Request,
    tournament_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Render tournament detail page or HTMX live partial."""
    from atp.dashboard.models import User
    from atp.dashboard.tournament.models import (
        Action,
        Participant,
        Round,
        Tournament,
    )

    result = await session.execute(
        select(Tournament)
        .where(Tournament.id == tournament_id)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds).selectinload(Round.actions),
        )
    )
    tournament = result.scalar_one_or_none()

    if tournament is None:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/error.html",
            context={
                "error_title": "Not Found",
                "error_message": f"Tournament #{tournament_id} not found.",
            },
            status_code=404,
        )

    # Creator username
    creator_name = "—"
    if tournament.created_by:
        creator = await session.get(User, tournament.created_by)
        if creator:
            creator_name = creator.username

    # Admin check
    is_admin = False
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        user = await session.get(User, user_id)
        if user and user.is_admin:
            is_admin = True

    # Cancelled-by username
    cancelled_by_name = None
    if tournament.cancelled_by:
        cb_user = await session.get(User, tournament.cancelled_by)
        if cb_user:
            cancelled_by_name = cb_user.username

    # Sort rounds newest-first
    sorted_rounds = sorted(
        tournament.rounds,
        key=lambda r: r.round_number,
        reverse=True,
    )

    # Sort participants by id for consistent score display
    sorted_participants = sorted(
        tournament.participants,
        key=lambda p: p.id,
    )

    # Build participant id->name map for round history columns
    participant_map = {p.id: p.agent_name for p in sorted_participants}

    # Completed round count
    completed_rounds = sum(
        1 for r in tournament.rounds if r.status == "completed"
    )

    # Build event timeline (admin only)
    timeline = []
    if is_admin:
        if tournament.created_at:
            timeline.append(
                ("tournament_created", tournament.created_at,
                 f"{tournament.game_type}, {tournament.num_players} players, "
                 f"{tournament.total_rounds} rounds")
            )
        for p in sorted(tournament.participants, key=lambda p: p.joined_at or datetime.min):
            if p.joined_at:
                timeline.append(
                    ("participant_joined", p.joined_at, p.agent_name)
                )
        for r in sorted(tournament.rounds, key=lambda r: r.round_number):
            if r.started_at:
                timeline.append(
                    ("round_started", r.started_at,
                     f"Round {r.round_number} of {tournament.total_rounds}")
                )
        if tournament.ends_at and tournament.status == "completed":
            timeline.append(
                ("tournament_completed", tournament.ends_at, "")
            )
        if tournament.cancelled_at and tournament.status == "cancelled":
            reason_text = ""
            if tournament.cancelled_reason:
                reason_text = tournament.cancelled_reason.value
            timeline.append(
                ("tournament_cancelled", tournament.cancelled_at, reason_text)
            )
        # Newest first
        timeline.sort(key=lambda e: e[1], reverse=True)

    context = {
        "active_page": "tournaments",
        "tournament": tournament,
        "creator_name": creator_name,
        "is_admin": is_admin,
        "cancelled_by_name": cancelled_by_name,
        "sorted_rounds": sorted_rounds,
        "sorted_participants": sorted_participants,
        "participant_map": participant_map,
        "completed_rounds": completed_rounds,
        "timeline": timeline,
    }

    partial = request.query_params.get("partial")
    if partial == "live":
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/tournament_live.html",
            context=context,
        )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_detail.html",
        context=context,
    )
```

Also add `getattr` to imports if not already present (it's a builtin, no import needed).

- [ ] **Step 4: Create tournament detail template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`:

```html
{% extends "ui/base_ui.html" %}

{% set t = tournament %}
{% set name = (t.config or {}).get("name", "Tournament #" ~ t.id) %}

{% block title %}{{ name }} - ATP Platform{% endblock %}

{% block content %}
<h2>{{ name }}</h2>
<div class="tournament-meta">
    {% if t.status == "completed" %}
    <span class="status-badge" style="background:#28a745">completed</span>
    {% elif t.status == "active" %}
    <span class="status-badge" style="background:#007bff">active</span>
    {% elif t.status == "pending" %}
    <span class="status-badge" style="background:#ffc107;color:#333">pending</span>
    {% elif t.status == "cancelled" %}
    <span class="status-badge" style="background:#dc3545">cancelled</span>
    {% endif %}
    <span>{{ t.game_type }}</span>
    <span>by {{ creator_name }}</span>
    <span>
        {% if t.status == "completed" and t.starts_at and t.ends_at %}
        {{ t.starts_at.strftime("%b %d, %H:%M") }} — {{ t.ends_at.strftime("%H:%M") }}
        {% elif t.status == "active" and t.starts_at %}
        Started {{ t.starts_at.strftime("%b %d, %H:%M") }}
        {% else %}
        Created {{ t.created_at.strftime("%b %d, %H:%M") if t.created_at else "—" }}
        {% endif %}
    </span>
</div>

{% if t.status == "cancelled" %}
<div class="cancel-box">
    <div class="reason">
        {% if t.cancelled_reason and t.cancelled_reason.value == "pending_timeout" %}
        Expired before full roster
        {% elif t.cancelled_reason and t.cancelled_reason.value == "admin_action" %}
        Cancelled by {{ cancelled_by_name or "admin" }}
        {% elif t.cancelled_reason and t.cancelled_reason.value == "abandoned" %}
        All participants left
        {% else %}
        Cancelled
        {% endif %}
    </div>
    <div class="detail">
        {% if t.cancelled_reason and t.cancelled_reason.value == "pending_timeout" %}
        Tournament did not reach the required {{ t.num_players }} players.
        Auto-cancelled at {{ t.cancelled_at.strftime("%b %d, %H:%M") if t.cancelled_at else "—" }}.
        {% elif t.cancelled_reason and t.cancelled_reason.value == "admin_action" %}
        {{ t.cancelled_reason_detail or "" }}
        Cancelled at {{ t.cancelled_at.strftime("%b %d, %H:%M") if t.cancelled_at else "—" }}.
        {% elif t.cancelled_reason and t.cancelled_reason.value == "abandoned" %}
        {% set last_release = sorted_participants | map(attribute="released_at") | reject("none") | sort(reverse=true) | first %}
        Last departure at {{ last_release.strftime("%b %d, %H:%M") if last_release else "—" }}.
        {% endif %}
    </div>
</div>
{% endif %}

<div id="live-content"
     {% if t.status == "active" %}
     hx-get="/ui/tournaments/{{ t.id }}?partial=live"
     hx-trigger="every 10s"
     hx-swap="innerHTML"
     {% endif %}>
    {% include "ui/partials/tournament_live.html" %}
</div>

{% if is_admin and timeline %}
<hr style="margin: 2rem 0;">
<h3>Event Timeline <span class="admin-badge">admin only</span></h3>
<div class="timeline" style="padding: 1rem; background: white; border: 1px solid #e0e0e0; border-radius: 0.5rem;">
    {% if timeline | length <= 10 %}
        {% for event_name, event_time, event_detail in timeline %}
        <div class="timeline-item">
            <span class="event-name">{{ event_name }}</span>
            <span class="event-time">{{ event_time.strftime("%H:%M:%S") }}</span>
            {% if event_detail %}
            <div class="event-detail">{{ event_detail }}</div>
            {% endif %}
        </div>
        {% endfor %}
    {% else %}
        {% for event_name, event_time, event_detail in timeline[:5] %}
        <div class="timeline-item">
            <span class="event-name">{{ event_name }}</span>
            <span class="event-time">{{ event_time.strftime("%H:%M:%S") }}</span>
            {% if event_detail %}
            <div class="event-detail">{{ event_detail }}</div>
            {% endif %}
        </div>
        {% endfor %}
        <div class="timeline-item" style="color:#999">
            <span class="event-name" style="color:#999">... {{ timeline | length - 10 }} more events ...</span>
        </div>
        {% for event_name, event_time, event_detail in timeline[-5:] %}
        <div class="timeline-item">
            <span class="event-name">{{ event_name }}</span>
            <span class="event-time">{{ event_time.strftime("%H:%M:%S") }}</span>
            {% if event_detail %}
            <div class="event-detail">{{ event_detail }}</div>
            {% endif %}
        </div>
        {% endfor %}
    {% endif %}
</div>
{% endif %}

<p style="margin-top:1.5rem">
    <a href="/api/v1/tournaments/{{ t.id }}" style="color:#7c3aed">View raw JSON →</a>
</p>
{% endblock %}
```

- [ ] **Step 5: Create live partial template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_live.html`:

```html
{% set t = tournament %}
{% set active_players = t.participants | selectattr("released_at", "none") | list | length %}

{# Stat cards #}
<div class="stat-cards">
    <div class="stat-card">
        <div class="value">{{ active_players }} / {{ t.num_players }}</div>
        <div class="label">Players</div>
    </div>
    <div class="stat-card">
        <div class="value">{{ completed_rounds }} / {{ t.total_rounds }}</div>
        <div class="label">Rounds</div>
    </div>
    <div class="stat-card">
        <div class="value">
            {% if t.status == "completed" and t.starts_at and t.ends_at %}
                {% set dur = (t.ends_at - t.starts_at).total_seconds() | int %}
                {{ dur // 60 }}m {{ dur % 60 }}s
            {% elif t.status == "active" %}
                In progress
            {% else %}
                —
            {% endif %}
        </div>
        <div class="label">Duration</div>
    </div>
    <div class="stat-card">
        <div class="value">{{ t.round_deadline_s }}s</div>
        <div class="label">Round Deadline</div>
    </div>
</div>

{# Scoreboard #}
{% if sorted_participants and completed_rounds > 0 %}
<h3>Scoreboard</h3>
<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Agent</th>
            <th>Score</th>
            <th>Avg / Round</th>
        </tr>
    </thead>
    <tbody>
        {% set ranked = sorted_participants | sort(attribute="total_score", reverse=true) %}
        {% set ns = namespace(rank=0, prev_score=none) %}
        {% for p in ranked %}
        {% if p.total_score != ns.prev_score %}
            {% set ns.rank = loop.index %}
        {% endif %}
        {% set ns.prev_score = p.total_score %}
        <tr{% if ns.rank <= 3 %} style="background:#f8f5ff"{% endif %}>
            <td>
                {% if ns.rank == 1 %}🥇{% elif ns.rank == 2 %}🥈{% elif ns.rank == 3 %}🥉{% endif %}
                {{ ns.rank }}
            </td>
            <td>{{ p.agent_name }}</td>
            <td class="score-pair" style="color:#7c3aed">
                {{ p.total_score if p.total_score is not none else "—" }}
            </td>
            <td class="score-pair">
                {% if p.total_score is not none and completed_rounds > 0 %}
                {{ "%.2f" | format(p.total_score / completed_rounds) }}
                {% else %}—{% endif %}
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endif %}

{# Round history #}
{% if sorted_rounds and completed_rounds > 0 %}
<h3>Round History</h3>
<table>
    <thead>
        <tr>
            <th>Round</th>
            {% for p in sorted_participants %}
            <th>{{ p.agent_name }}</th>
            {% endfor %}
            <th>Payoff</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        {% for r in sorted_rounds %}
        {% if r.status == "completed" %}
        <tr>
            <td class="score-pair">{{ r.round_number }}</td>
            {% for p in sorted_participants %}
                {% set action = r.actions | selectattr("participant_id", "equalto", p.id) | first %}
                {% if action %}
                    {% set choice = (action.action_data or {}).get("choice", "—") %}
                    <td>
                        <span class="{{ 'cooperate' if choice == 'cooperate' else 'defect' }}">{{ choice }}</span>
                        {% if is_admin and action.source == "TIMEOUT_DEFAULT" %}
                        <span style="font-size:0.75rem;color:#999">(timeout)</span>
                        {% endif %}
                    </td>
                {% else %}
                    <td>—</td>
                {% endif %}
            {% endfor %}
            <td class="score-pair">
                {% set ns = namespace(parts=[]) %}
                {% for p in sorted_participants %}
                    {% set action = r.actions | selectattr("participant_id", "equalto", p.id) | first %}
                    {% if action and action.payoff is not none %}
                        {% set ns.parts = ns.parts + [action.payoff | string] %}
                    {% else %}
                        {% set ns.parts = ns.parts + ["—"] %}
                    {% endif %}
                {% endfor %}
                {{ ns.parts | join(" : ") }}
            </td>
            <td style="font-size:0.8rem;color:#999">{{ r.status }}</td>
        </tr>
        {% endif %}
        {% endfor %}
    </tbody>
</table>
{% endif %}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/dashboard/ui/test_tournament_ui.py -v -x`
Expected: All 6 tests PASS

- [ ] **Step 7: Run ruff format and check**

Run: `uv run ruff format packages/atp-dashboard/atp/dashboard/v2/routes/ui.py && uv run ruff check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_live.html
git add tests/unit/dashboard/ui/test_tournament_ui.py
git commit -m "feat(dashboard): tournament detail page with scoreboard, rounds, timeline"
```

---

### Task 5: Manual smoke test

**Files:** None (verification only)

- [ ] **Step 1: Start the dashboard server**

```bash
ATP_DISABLE_AUTH=true ATP_SECRET_KEY=dev uv run uvicorn atp.dashboard.v2.factory:app --host 127.0.0.1 --port 8080
```

- [ ] **Step 2: Open in browser and verify**

1. Navigate to `http://localhost:8080/ui/` — verify "Tournaments" appears in sidebar nav
2. Click "Tournaments" — verify list page loads (may be empty if no local data)
3. Click "Games" — verify game registry shows, tournament table replaced with link
4. If tournaments exist, click one → verify detail page loads with scoreboard, rounds, etc.
5. Navigate to `http://localhost:8080/ui/tournaments/99999` — verify 404 page

- [ ] **Step 3: Run full tournament test suite**

```bash
uv run python -m pytest tests/unit/dashboard -v -x
```

- [ ] **Step 4: Run ruff + pyrefly**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
```

- [ ] **Step 5: Final commit (if any formatting fixes needed)**

```bash
git add -A && git commit -m "style: formatting fixes for tournament dashboard"
```
