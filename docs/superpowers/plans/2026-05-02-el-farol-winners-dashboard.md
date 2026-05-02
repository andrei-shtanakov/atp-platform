# El Farol Winners Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-tournament winners poster (`/ui/tournaments/{id}/winners`), cross-tournament Hall of Fame UI (`/ui/leaderboard/el-farol`), and a JSON contract (`/api/public/leaderboard/el-farol`) for completed public El Farol tournaments.

**Architecture:** Two routers (`winners_ui.py` mounted alongside `ui_router`, `winners_api.py` mounted under `/api`) share Pydantic models and SQL helpers from `services/winners.py`. Reads only — no schema changes. Caching via the existing `QueryCache` (60 s TTL) plus `Cache-Control: public, s-maxage=60` headers. Filter is `tenant_id = DEFAULT_TENANT_ID AND game_type = "el_farol" AND status = "completed" AND join_token IS NULL`. Hall of Fame aggregates by *logical agent* `(tenant_id, owner_id, name)`, builtin participants excluded.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x async, Pydantic v2, Jinja2 + HTMX + Pico CSS, pytest + anyio.

**Spec:** `docs/superpowers/specs/2026-05-02-el-farol-winners-dashboard-design.md`

---

## File Structure

### Created files

| Path | Responsibility |
|---|---|
| `packages/atp-dashboard/atp/dashboard/v2/services/winners.py` | Pydantic models (`WinnerEntry`, `HallEntry`, `LeaderboardPayload`), query helpers (`_winners_query`, `_hall_of_fame_query`), `QueryCache` instances |
| `packages/atp-dashboard/atp/dashboard/v2/routes/winners_api.py` | JSON router at `/public/leaderboard/el-farol` (mounted under `/api`) |
| `packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py` | HTML routes at `/ui/tournaments/{id}/winners` and `/ui/leaderboard/el-farol` |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_tournament.html` | Per-tournament poster page |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_hall_of_fame.html` | Hall of Fame page with HTMX pagination |
| `tests/unit/dashboard/winners/__init__.py` | Empty marker |
| `tests/unit/dashboard/winners/conftest.py` | In-memory session fixture (mirrors `tests/unit/dashboard/tournament/conftest.py`) |
| `tests/unit/dashboard/winners/test_aggregation.py` | Unit tests for `_winners_query` and `_hall_of_fame_query` |
| `tests/integration/dashboard/winners/__init__.py` | Empty marker |
| `tests/integration/dashboard/winners/test_winners_tournament_ui.py` | Per-tournament HTML route tests + access gates |
| `tests/integration/dashboard/winners/test_hall_of_fame_ui.py` | Hall of Fame HTML route tests + pagination |
| `tests/integration/dashboard/winners/test_hall_of_fame_api.py` | Hall of Fame JSON route tests + cache headers |

### Modified files

| Path | Why |
|---|---|
| `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` | Register `winners_api_router` (lands under `/api`) |
| `packages/atp-dashboard/atp/dashboard/v2/factory.py` | Include `winners_ui_router` alongside `ui_router` |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html` | Add "Winners →" link for completed/public/el_farol tournaments |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html` | Add "El Farol Hall of Fame" sidebar link |
| `scripts/smoke_el_farol_prod.py` | Fetch the winners page after tournament completes |

---

## Task 1: Scaffold `services/winners.py` with models and cache

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/services/winners.py`

**Goal:** Pure scaffolding — no SQL yet. Establishes the Pydantic contract and cache instances that downstream tasks plug into.

- [ ] **Step 1: Create the file**

```python
# packages/atp-dashboard/atp/dashboard/v2/services/winners.py
"""Shared models and query helpers for the El Farol winners dashboard.

The two query helpers (``_winners_query`` and ``_hall_of_fame_query``)
are the single source of truth for both the JSON API
(``routes/winners_api.py``) and the HTML pages
(``routes/winners_ui.py``). Caching uses ``QueryCache`` with semantic
keys built via ``QueryCache._make_key``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from atp.dashboard.query_cache import QueryCache
from atp.dashboard.v2.routes.el_farol_from_tournament import _CAPACITY_RATIO

# Pin the schema version on the JSON contract. Bump on any
# breaking change to ``LeaderboardPayload``.
SCHEMA_VERSION = 1


class WinnerEntry(BaseModel):
    """One row on the per-tournament winners poster.

    ``agent_name`` is the historical display name from
    ``Participant.agent_name``; ``agent_description`` and
    ``owner_username`` are read from the live ``Agent`` and ``User`` rows
    (i.e. "as of now"). Builtin participants render with
    ``owner_username='system'`` and a fixed description.
    """

    rank: int
    agent_name: str
    agent_description: str | None
    owner_username: str
    score: float | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    model_id: str | None  # "mixed" if the participant emitted multiple distinct model ids


class HallEntry(BaseModel):
    """One row on the Hall of Fame, keyed by logical agent.

    Logical agent identity = ``(tenant_id, owner_id, agent.name)``.
    Versions of the same agent (different ``Agent.version`` rows under
    the same ``(owner_id, name)``) are aggregated together.
    """

    rank: int
    owner_username: str
    agent_name: str
    agent_description: str | None
    total_score: float
    tournaments_count: int


class LeaderboardPayload(BaseModel):
    """Public JSON contract for the Hall of Fame.

    ``schema_version`` lets external consumers refuse mismatched shapes.
    ``generated_at`` is the server time when the response was assembled
    — useful for debugging caches and for clients that want to display
    "as of HH:MM".
    """

    schema_version: int = Field(default=SCHEMA_VERSION)
    generated_at: datetime
    total: int
    limit: int
    offset: int
    entries: list[HallEntry]


# 60 s TTL is deliberate. Per-tournament rows are immutable post-
# ``completed`` and could safely use a much longer TTL, but keeping
# both caches at 60 s makes invalidation trivial if we ever add a
# mutable post-completion field (e.g. moderator-added annotations).
_WINNERS_CACHE_TTL_S = 60
_WINNERS_CACHE_MAX = 256

_winners_cache: QueryCache[list[WinnerEntry]] | None = None
_hof_cache: QueryCache[tuple[int, list[HallEntry]]] | None = None


def get_winners_cache() -> QueryCache[list[WinnerEntry]]:
    """Lazy singleton for the per-tournament winners cache."""
    global _winners_cache
    if _winners_cache is None:
        _winners_cache = QueryCache(
            max_size=_WINNERS_CACHE_MAX,
            ttl_seconds=_WINNERS_CACHE_TTL_S,
        )
    return _winners_cache


def get_hof_cache() -> QueryCache[tuple[int, list[HallEntry]]]:
    """Lazy singleton for the Hall of Fame paginated cache.

    Cache value is ``(total_count, page_entries)`` so a single fetch
    yields both pieces the route needs.
    """
    global _hof_cache
    if _hof_cache is None:
        _hof_cache = QueryCache(
            max_size=_WINNERS_CACHE_MAX,
            ttl_seconds=_WINNERS_CACHE_TTL_S,
        )
    return _hof_cache


def reset_caches_for_tests() -> None:
    """Drop both caches. Tests call this to keep TTL state isolated."""
    global _winners_cache, _hof_cache
    _winners_cache = None
    _hof_cache = None


# Re-export for ergonomic import in routes.
CAPACITY_RATIO = _CAPACITY_RATIO
```

- [ ] **Step 2: Confirm the module imports cleanly**

Run: `uv run python -c "from atp.dashboard.v2.services import winners; print(winners.SCHEMA_VERSION, winners.CAPACITY_RATIO)"`
Expected output: `1 0.6`

- [ ] **Step 3: Type-check**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/services/winners.py`
Expected: no errors. Fix any reported issues before moving on.

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/winners.py
git commit -m "feat(winners): scaffold services/winners.py with Pydantic models + cache"
```

---

## Task 2: Implement `_winners_query` (per-tournament aggregator)

**Files:**
- Create: `tests/unit/dashboard/winners/__init__.py`
- Create: `tests/unit/dashboard/winners/conftest.py`
- Create: `tests/unit/dashboard/winners/test_aggregation.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/winners.py`

**Goal:** Add the per-tournament SQL helper plus a unit-test suite that locks in dense ranking, builtin handling, archived agents, telemetry aggregation, and tiebreakers.

- [ ] **Step 1: Create the test package markers**

```bash
mkdir -p tests/unit/dashboard/winners
```

Create `tests/unit/dashboard/winners/__init__.py` as an empty file.

- [ ] **Step 2: Create `tests/unit/dashboard/winners/conftest.py`**

```python
"""Shared fixtures for winners aggregation tests.

In-memory SQLite + ``Base.metadata.create_all``, mirroring
``tests/unit/dashboard/tournament/conftest.py`` but without the
tournament-engine cache hooks (we don't run any game logic here, only
read SQL).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import atp.dashboard.tournament.models  # noqa: F401  (register tournament tables)
from atp.dashboard.models import Base
from atp.dashboard.v2.services.winners import reset_caches_for_tests


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


@pytest.fixture(autouse=True)
def _isolate_caches() -> None:
    """Drop QueryCache singletons between tests."""
    reset_caches_for_tests()
    yield
    reset_caches_for_tests()
```

- [ ] **Step 3: Write the first failing test (empty tournament)**

Create `tests/unit/dashboard/winners/test_aggregation.py`:

```python
"""Unit tests for the winners aggregation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, User
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.services.winners import (
    _hall_of_fame_query,
    _winners_query,
)


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
    session: AsyncSession,
    *,
    owner: User,
    name: str,
    description: str | None = None,
    version: str = "1",
    deleted_at: datetime | None = None,
) -> Agent:
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name=name,
        agent_type="tournament",
        owner_id=owner.id,
        description=description,
        version=version,
        deleted_at=deleted_at,
        purpose="tournament",
    )
    session.add(a)
    await session.flush()
    return a


async def _make_tournament(
    session: AsyncSession,
    *,
    game_type: str = "el_farol",
    status: TournamentStatus = TournamentStatus.COMPLETED,
    join_token: str | None = None,
    name: str = "T",
    num_players: int = 2,
    total_rounds: int = 5,
    starts_at: datetime | None = None,
    ends_at: datetime | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> Tournament:
    starts_at = starts_at or datetime(2026, 5, 1, 12, 0, 0)
    ends_at = ends_at or starts_at + timedelta(minutes=10)
    t = Tournament(
        tenant_id=tenant_id,
        game_type=game_type,
        config={"name": name},
        status=status,
        starts_at=starts_at,
        ends_at=ends_at,
        num_players=num_players,
        total_rounds=total_rounds,
        round_deadline_s=30,
        join_token=join_token,
        pending_deadline=starts_at,
    )
    session.add(t)
    await session.flush()
    return t


async def _make_participant(
    session: AsyncSession,
    *,
    tournament: Tournament,
    user: User | None = None,
    agent: Agent | None = None,
    builtin_strategy: str | None = None,
    agent_name: str = "agent",
    total_score: float | None = None,
) -> Participant:
    p = Participant(
        tournament_id=tournament.id,
        user_id=user.id if user else None,
        agent_id=agent.id if agent else None,
        builtin_strategy=builtin_strategy,
        agent_name=agent_name,
        total_score=total_score,
    )
    session.add(p)
    await session.flush()
    return p


@pytest.mark.anyio
async def test_winners_query_returns_empty_for_no_participants(
    session: AsyncSession,
):
    t = await _make_tournament(session)
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert rows == []
```

- [ ] **Step 4: Verify the test fails because `_winners_query` does not exist yet**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py -v`
Expected: ImportError on `_winners_query` — failure inside the import line.

- [ ] **Step 5: Add `_winners_query` skeleton to `services/winners.py`**

Append to `packages/atp-dashboard/atp/dashboard/v2/services/winners.py`:

```python
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import Agent, User
from atp.dashboard.tournament.models import Action, Participant


async def _winners_query(
    session: AsyncSession, tournament_id: int
) -> list[WinnerEntry]:
    """Aggregate winners for one tournament.

    Pulls per-participant totals together with optional LLM telemetry
    summed across the participant's actions. The display name comes
    from ``Participant.agent_name`` (historical at join time);
    description and owner are looked up live from ``Agent`` / ``User``.
    """
    stmt = (
        select(
            Participant.id.label("participant_id"),
            Participant.agent_name.label("display_name"),
            Participant.agent_id.label("agent_id"),
            Participant.builtin_strategy.label("builtin_strategy"),
            Participant.total_score.label("total_score"),
            Agent.description.label("agent_description"),
            Agent.deleted_at.label("agent_deleted_at"),
            User.username.label("owner_username"),
            func.sum(Action.tokens_in).label("tokens_in"),
            func.sum(Action.tokens_out).label("tokens_out"),
            func.sum(Action.cost_usd).label("cost_usd"),
            func.min(Action.model_id).label("sample_model"),
            func.count(func.distinct(Action.model_id)).label("distinct_models"),
        )
        .join(Agent, Agent.id == Participant.agent_id, isouter=True)
        .join(User, User.id == Agent.owner_id, isouter=True)
        .join(Action, Action.participant_id == Participant.id, isouter=True)
        .where(Participant.tournament_id == tournament_id)
        .group_by(Participant.id, Agent.id, User.id)
        .order_by(
            Participant.total_score.desc().nulls_last(),
            Participant.id.asc(),
        )
    )
    result = await session.execute(stmt)
    rows = result.all()

    # Dense ranking — ties share a rank, the next non-tied score jumps
    # to len(seen_so_far) + 1. Pure post-processing keeps the SQL
    # portable across SQLite (test DB) and Postgres (prod DB).
    entries: list[WinnerEntry] = []
    rank = 0
    prev_score: float | None = None
    seen = 0
    for row in rows:
        seen += 1
        if row.total_score != prev_score:
            rank = seen
            prev_score = row.total_score

        if row.agent_id is None:
            owner_username = "system"
            description = "built-in strategy"
            display_name = row.display_name
        else:
            owner_username = row.owner_username or "—"
            description = row.agent_description
            display_name = row.display_name
            if row.agent_deleted_at is not None:
                display_name = f"{display_name} (archived)"

        if row.distinct_models is None or row.distinct_models == 0:
            model_id: str | None = None
        elif row.distinct_models == 1:
            model_id = row.sample_model
        else:
            model_id = "mixed"

        entries.append(
            WinnerEntry(
                rank=rank,
                agent_name=display_name,
                agent_description=description,
                owner_username=owner_username,
                score=row.total_score,
                tokens_in=row.tokens_in,
                tokens_out=row.tokens_out,
                cost_usd=row.cost_usd,
                model_id=model_id,
            )
        )

    return entries
```

- [ ] **Step 6: Run the empty-tournament test to confirm it passes**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py::test_winners_query_returns_empty_for_no_participants -v`
Expected: PASS.

- [ ] **Step 7: Add the happy-path test (single participant, with telemetry)**

Append to `tests/unit/dashboard/winners/test_aggregation.py`:

```python
async def _make_round_with_action(
    session: AsyncSession,
    *,
    participant: Participant,
    round_number: int,
    payoff: float | None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
    model_id: str | None = None,
) -> Action:
    rnd = Round(
        tournament_id=participant.tournament_id,
        round_number=round_number,
        status=RoundStatus.COMPLETED,
        deadline_at=datetime(2026, 5, 1, 12, 0, 0),
    )
    session.add(rnd)
    await session.flush()
    act = Action(
        round_id=rnd.id,
        participant_id=participant.id,
        action_data={},
        payoff=payoff,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model_id=model_id,
    )
    session.add(act)
    await session.flush()
    return act


@pytest.mark.anyio
async def test_winners_query_single_participant_with_telemetry(
    session: AsyncSession,
):
    t = await _make_tournament(session)
    alice = await _make_user(session, "alice")
    agent = await _make_agent(
        session, owner=alice, name="alfa", description="greedy spammer"
    )
    p = await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=42.0,
    )
    await _make_round_with_action(
        session,
        participant=p,
        round_number=1,
        payoff=42.0,
        tokens_in=100,
        tokens_out=80,
        cost_usd=0.01,
        model_id="gpt-4o-mini",
    )
    await _make_round_with_action(
        session,
        participant=p,
        round_number=2,
        payoff=0.0,
        tokens_in=50,
        tokens_out=40,
        cost_usd=0.005,
        model_id="gpt-4o-mini",
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 1
    e = rows[0]
    assert e.rank == 1
    assert e.agent_name == "alfa"
    assert e.agent_description == "greedy spammer"
    assert e.owner_username == "alice"
    assert e.score == 42.0
    assert e.tokens_in == 150
    assert e.tokens_out == 120
    assert e.cost_usd == pytest.approx(0.015)
    assert e.model_id == "gpt-4o-mini"
```

- [ ] **Step 8: Run the new test**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py::test_winners_query_single_participant_with_telemetry -v`
Expected: PASS.

- [ ] **Step 9: Add tests for builtin, archived, mixed model, dense ranking, no telemetry**

Append:

```python
@pytest.mark.anyio
async def test_winners_query_builtin_owner_is_system(session: AsyncSession):
    t = await _make_tournament(session)
    p = await _make_participant(
        session,
        tournament=t,
        user=None,
        agent=None,
        builtin_strategy="el_farol/random",
        agent_name="el_farol/random",
        total_score=10.0,
    )
    await _make_round_with_action(session, participant=p, round_number=1, payoff=10.0)
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 1
    assert rows[0].owner_username == "system"
    assert rows[0].agent_description == "built-in strategy"
    assert rows[0].model_id is None  # no telemetry recorded
    assert rows[0].tokens_in is None
    assert rows[0].cost_usd is None


@pytest.mark.anyio
async def test_winners_query_archived_agent_gets_suffix(session: AsyncSession):
    t = await _make_tournament(session)
    alice = await _make_user(session, "alice")
    agent = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        deleted_at=datetime(2026, 5, 1, 11, 0, 0),
    )
    await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=5.0,
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 1
    assert rows[0].agent_name == "alfa (archived)"


@pytest.mark.anyio
async def test_winners_query_mixed_model_id(session: AsyncSession):
    t = await _make_tournament(session)
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    p = await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=10.0,
    )
    await _make_round_with_action(
        session, participant=p, round_number=1, payoff=5.0, model_id="gpt-4o-mini"
    )
    await _make_round_with_action(
        session, participant=p, round_number=2, payoff=5.0, model_id="claude-haiku-4-5"
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert rows[0].model_id == "mixed"


@pytest.mark.anyio
async def test_winners_query_dense_rank_with_ties(session: AsyncSession):
    t = await _make_tournament(session, num_players=3)
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    carol = await _make_user(session, "carol")
    a = await _make_agent(session, owner=alice, name="a")
    b = await _make_agent(session, owner=bob, name="b")
    c = await _make_agent(session, owner=carol, name="c")
    await _make_participant(
        session, tournament=t, user=alice, agent=a, agent_name="a", total_score=100.0
    )
    await _make_participant(
        session, tournament=t, user=bob, agent=b, agent_name="b", total_score=100.0
    )
    await _make_participant(
        session, tournament=t, user=carol, agent=c, agent_name="c", total_score=90.0
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert [r.rank for r in rows] == [1, 1, 3]


@pytest.mark.anyio
async def test_winners_query_null_score_sorted_last(session: AsyncSession):
    t = await _make_tournament(session, num_players=2)
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    a = await _make_agent(session, owner=alice, name="a")
    b = await _make_agent(session, owner=bob, name="b")
    await _make_participant(
        session, tournament=t, user=alice, agent=a, agent_name="a", total_score=None
    )
    await _make_participant(
        session, tournament=t, user=bob, agent=b, agent_name="b", total_score=10.0
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert [r.agent_name for r in rows] == ["b", "a"]
    assert rows[1].score is None
```

- [ ] **Step 10: Run the full Task-2 test set**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py -v`
Expected: 6 PASS (empty + happy path + builtin + archived + mixed + dense rank + null score = 7 actually). All green.

- [ ] **Step 11: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/services/winners.py tests/unit/dashboard/winners/`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/services/winners.py tests/unit/dashboard/winners/`
Expected: clean. Fix any issues.

- [ ] **Step 12: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/winners.py tests/unit/dashboard/winners/
git commit -m "feat(winners): per-tournament winners aggregator + unit tests"
```

---

## Task 3: Implement `_hall_of_fame_query` (cross-tournament aggregator)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/winners.py`
- Modify: `tests/unit/dashboard/winners/test_aggregation.py`

**Goal:** Add the Hall of Fame helper. SUM by `(tenant_id, owner_id, name)` so versions of the same agent are aggregated; emit a `has_extant` signal so the post-processor can distinguish "all versions soft-deleted" from "no description".

- [ ] **Step 1: Write a failing test for empty result**

Append to `tests/unit/dashboard/winners/test_aggregation.py`:

```python
@pytest.mark.anyio
async def test_hall_of_fame_query_empty_when_no_tournaments(
    session: AsyncSession,
):
    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 0
    assert entries == []
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py::test_hall_of_fame_query_empty_when_no_tournaments -v`
Expected: ImportError or AttributeError on `_hall_of_fame_query`.

- [ ] **Step 3: Implement `_hall_of_fame_query` in `services/winners.py`**

Append to the module:

```python
from datetime import datetime, timezone

from sqlalchemy import and_, exists, literal


async def _hall_of_fame_query(
    session: AsyncSession,
    limit: int,
    offset: int,
) -> tuple[int, list[HallEntry]]:
    """Return (total_count, page) for the El Farol Hall of Fame.

    Identity: ``(tenant_id, owner_id, agent.name)``. Versions of the
    same agent are aggregated. Builtin participants and rows with NULL
    ``total_score`` are excluded.
    """
    from atp.dashboard.models import DEFAULT_TENANT_ID

    base_filters = and_(
        Tournament.tenant_id == DEFAULT_TENANT_ID,
        Agent.tenant_id == DEFAULT_TENANT_ID,
        Tournament.game_type == "el_farol",
        Tournament.status == TournamentStatus.COMPLETED,
        Tournament.join_token.is_(None),
        Participant.agent_id.is_not(None),
        Participant.total_score.is_not(None),
    )

    # Aggregation grouped by logical agent.
    agg_stmt = (
        select(
            Agent.tenant_id.label("tenant_id"),
            Agent.owner_id.label("owner_id"),
            Agent.name.label("name"),
            func.sum(Participant.total_score).label("total_score"),
            func.count(func.distinct(Participant.tournament_id)).label(
                "tournaments_count"
            ),
        )
        .join(Tournament, Tournament.id == Participant.tournament_id)
        .join(Agent, Agent.id == Participant.agent_id)
        .where(base_filters)
        .group_by(Agent.tenant_id, Agent.owner_id, Agent.name)
    ).subquery("agg")

    # Latest non-deleted description for the (owner, name) pair.
    latest_desc = (
        select(Agent.description)
        .where(
            Agent.tenant_id == agg_stmt.c.tenant_id,
            Agent.owner_id == agg_stmt.c.owner_id,
            Agent.name == agg_stmt.c.name,
            Agent.deleted_at.is_(None),
        )
        .order_by(Agent.updated_at.desc())
        .limit(1)
        .correlate(agg_stmt)
        .scalar_subquery()
    )

    has_extant = (
        exists()
        .where(
            Agent.tenant_id == agg_stmt.c.tenant_id,
            Agent.owner_id == agg_stmt.c.owner_id,
            Agent.name == agg_stmt.c.name,
            Agent.deleted_at.is_(None),
        )
        .correlate(agg_stmt)
    )

    page_stmt = (
        select(
            agg_stmt.c.owner_id,
            agg_stmt.c.name,
            agg_stmt.c.total_score,
            agg_stmt.c.tournaments_count,
            User.username.label("owner_username"),
            latest_desc.label("description"),
            has_extant.label("has_extant"),
        )
        .join(User, User.id == agg_stmt.c.owner_id, isouter=True)
        .order_by(
            agg_stmt.c.total_score.desc(),
            agg_stmt.c.owner_id.asc(),
            agg_stmt.c.name.asc(),
        )
        .limit(limit)
        .offset(offset)
    )

    count_stmt = select(func.count()).select_from(agg_stmt)
    total = (await session.execute(count_stmt)).scalar_one() or 0

    result = await session.execute(page_stmt)
    rows = result.all()

    entries: list[HallEntry] = []
    for i, row in enumerate(rows):
        rank = offset + i + 1
        if row.has_extant:
            agent_name = row.name
            description = row.description
        else:
            agent_name = f"{row.name} (archived)"
            description = None

        entries.append(
            HallEntry(
                rank=rank,
                owner_username=row.owner_username or "—",
                agent_name=agent_name,
                agent_description=description,
                total_score=float(row.total_score),
                tournaments_count=int(row.tournaments_count),
            )
        )

    return int(total), entries


def utc_now() -> datetime:
    """Server time used for ``LeaderboardPayload.generated_at``."""
    return datetime.now(tz=timezone.utc)
```

- [ ] **Step 4: Run the empty test**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py::test_hall_of_fame_query_empty_when_no_tournaments -v`
Expected: PASS.

- [ ] **Step 5: Add tests for SUM-across-tournaments, version aggregation, builtin exclusion, archived lineage, ordering, tenant filter, pagination**

Append:

```python
@pytest.mark.anyio
async def test_hall_of_fame_aggregates_two_tournaments(session: AsyncSession):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    t1 = await _make_tournament(session, name="T1")
    t2 = await _make_tournament(session, name="T2")
    await _make_participant(
        session, tournament=t1, user=alice, agent=agent, agent_name="alfa",
        total_score=10.0,
    )
    await _make_participant(
        session, tournament=t2, user=alice, agent=agent, agent_name="alfa",
        total_score=15.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].total_score == 25.0
    assert entries[0].tournaments_count == 2
    assert entries[0].agent_name == "alfa"
    assert entries[0].owner_username == "alice"


@pytest.mark.anyio
async def test_hall_of_fame_aggregates_versions_of_same_agent(
    session: AsyncSession,
):
    alice = await _make_user(session, "alice")
    v1 = await _make_agent(session, owner=alice, name="alfa", version="1")
    v2 = await _make_agent(session, owner=alice, name="alfa", version="2")
    t1 = await _make_tournament(session, name="T1")
    t2 = await _make_tournament(session, name="T2")
    await _make_participant(
        session, tournament=t1, user=alice, agent=v1, agent_name="alfa",
        total_score=10.0,
    )
    await _make_participant(
        session, tournament=t2, user=alice, agent=v2, agent_name="alfa",
        total_score=20.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1  # one logical agent, two versions collapsed
    assert entries[0].total_score == 30.0


@pytest.mark.anyio
async def test_hall_of_fame_excludes_builtins(session: AsyncSession):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    t = await _make_tournament(session)
    await _make_participant(
        session, tournament=t, user=alice, agent=agent, agent_name="alfa",
        total_score=10.0,
    )
    await _make_participant(
        session, tournament=t, user=None, agent=None,
        builtin_strategy="el_farol/random", agent_name="el_farol/random",
        total_score=8.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].agent_name == "alfa"


@pytest.mark.anyio
async def test_hall_of_fame_excludes_private_tournaments(
    session: AsyncSession,
):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    pub = await _make_tournament(session, name="public")
    priv = await _make_tournament(session, name="private", join_token="secret")
    await _make_participant(
        session, tournament=pub, user=alice, agent=agent, agent_name="alfa",
        total_score=10.0,
    )
    await _make_participant(
        session, tournament=priv, user=alice, agent=agent, agent_name="alfa",
        total_score=99.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].total_score == 10.0  # private tournament excluded


@pytest.mark.anyio
async def test_hall_of_fame_excludes_other_tenants(session: AsyncSession):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    t = await _make_tournament(session, tenant_id="other-tenant")
    await _make_participant(
        session, tournament=t, user=alice, agent=agent, agent_name="alfa",
        total_score=42.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 0


@pytest.mark.anyio
async def test_hall_of_fame_archived_lineage(session: AsyncSession):
    alice = await _make_user(session, "alice")
    deleted = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="1",
        deleted_at=datetime(2026, 5, 1, 0, 0, 0),
    )
    t = await _make_tournament(session)
    await _make_participant(
        session, tournament=t, user=alice, agent=deleted, agent_name="alfa",
        total_score=10.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].agent_name == "alfa (archived)"
    assert entries[0].agent_description is None


@pytest.mark.anyio
async def test_hall_of_fame_ordering_and_tiebreaker(session: AsyncSession):
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    a = await _make_agent(session, owner=alice, name="z")
    b = await _make_agent(session, owner=bob, name="a")
    t = await _make_tournament(session, num_players=2)
    # Tie on score — tiebreaker is owner_id ASC, then name ASC.
    await _make_participant(
        session, tournament=t, user=alice, agent=a, agent_name="z",
        total_score=10.0,
    )
    await _make_participant(
        session, tournament=t, user=bob, agent=b, agent_name="a",
        total_score=10.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 2
    # alice has lower id (created first), so she wins the tiebreaker.
    assert entries[0].owner_username == "alice"
    assert entries[1].owner_username == "bob"


@pytest.mark.anyio
async def test_hall_of_fame_pagination(session: AsyncSession):
    t = await _make_tournament(session, num_players=4)
    for i, score in enumerate([40, 30, 20, 10], start=1):
        u = await _make_user(session, f"u{i}")
        ag = await _make_agent(session, owner=u, name=f"agent{i}")
        await _make_participant(
            session, tournament=t, user=u, agent=ag, agent_name=f"agent{i}",
            total_score=float(score),
        )
    await session.commit()

    total, page1 = await _hall_of_fame_query(session, limit=2, offset=0)
    total2, page2 = await _hall_of_fame_query(session, limit=2, offset=2)
    assert total == 4 and total2 == 4
    assert [e.rank for e in page1] == [1, 2]
    assert [e.rank for e in page2] == [3, 4]
    assert page1[0].total_score == 40.0
    assert page2[0].total_score == 20.0
```

- [ ] **Step 6: Run the full unit suite**

Run: `uv run pytest tests/unit/dashboard/winners/test_aggregation.py -v`
Expected: all PASS.

- [ ] **Step 7: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/services/winners.py tests/unit/dashboard/winners/`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/services/winners.py tests/unit/dashboard/winners/`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/winners.py tests/unit/dashboard/winners/test_aggregation.py
git commit -m "feat(winners): hall-of-fame aggregator with version coalescing"
```

---

## Task 4: Wire `winners_api.py` JSON endpoint

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/winners_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Create: `tests/integration/dashboard/winners/__init__.py`
- Create: `tests/integration/dashboard/winners/test_hall_of_fame_api.py`

**Goal:** Anonymous JSON endpoint at `/api/public/leaderboard/el-farol` with strict 422 on bounds, exact `Cache-Control: public, s-maxage=60`, and pagination-aware caching.

- [ ] **Step 1: Create `routes/winners_api.py`**

```python
# packages/atp-dashboard/atp/dashboard/v2/routes/winners_api.py
"""JSON API for the El Farol leaderboard.

Mounted under ``/api`` (via ``v2.routes.__init__``), so the absolute
URL is ``/api/public/leaderboard/el-farol``. The ``/public/leaderboard``
prefix lines up with the existing ``public_leaderboard.py`` convention
for anonymous read endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Response

from atp.dashboard.query_cache import QueryCache
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.services.winners import (
    HallEntry,
    LeaderboardPayload,
    _hall_of_fame_query,
    get_hof_cache,
    utc_now,
)

router = APIRouter(prefix="/public/leaderboard", tags=["winners-api"])

# 60 s parity with QueryCache TTL — see services/winners.py for the
# rationale on keeping per-tournament and HoF TTLs aligned.
_CACHE_CONTROL = "public, s-maxage=60"


@router.get("/el-farol", response_model=LeaderboardPayload)
async def get_hall_of_fame_json(
    session: DBSession,
    response: Response,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> LeaderboardPayload:
    """Return the El Farol Hall of Fame as JSON.

    Strict bounds: ``limit`` must be 1..200, ``offset`` >= 0; both are
    422 on violation (no silent clamp).
    """
    cache = get_hof_cache()
    key = QueryCache._make_key("hall_of_fame", limit=limit, offset=offset)
    cached = cache.get(key)
    if cached is not None:
        total, entries = cached
    else:
        total, entries = await _hall_of_fame_query(
            session, limit=limit, offset=offset
        )
        cache.put(key, (total, entries))

    response.headers["Cache-Control"] = _CACHE_CONTROL
    return LeaderboardPayload(
        generated_at=utc_now(),
        total=total,
        limit=limit,
        offset=offset,
        entries=entries,
    )
```

- [ ] **Step 2: Register the router in `routes/__init__.py`**

Edit `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`. Add the import next to the other routers:

```python
from atp.dashboard.v2.routes.winners_api import router as winners_api_router
```

In the include block (alongside `public_leaderboard_router`):

```python
router.include_router(winners_api_router)
```

In `__all__`:

```python
"winners_api_router",
```

- [ ] **Step 3: Create the integration test directory marker**

```bash
mkdir -p tests/integration/dashboard/winners
```

Create `tests/integration/dashboard/winners/__init__.py` as an empty file.

- [ ] **Step 4: Write the JSON contract test**

Create `tests/integration/dashboard/winners/test_hall_of_fame_api.py`:

```python
"""HTTP-level tests for /api/public/leaderboard/el-farol."""

from __future__ import annotations

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
from atp.dashboard.v2.services.winners import (
    SCHEMA_VERSION,
    reset_caches_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_caches_for_tests()
    yield
    reset_caches_for_tests()


async def _seed(session: AsyncSession) -> None:
    alice = User(
        username="alice", email="a@e.com", hashed_password="x",
        is_admin=False, is_active=True,
    )
    session.add(alice)
    await session.flush()
    agent = Agent(
        tenant_id=DEFAULT_TENANT_ID, name="alfa", agent_type="tournament",
        owner_id=alice.id, description="greedy", purpose="tournament",
    )
    session.add(agent)
    await session.flush()
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=DEFAULT_TENANT_ID, game_type="el_farol",
        config={"name": "T1"}, status=TournamentStatus.COMPLETED,
        starts_at=starts, ends_at=starts + timedelta(minutes=10),
        num_players=2, total_rounds=5, round_deadline_s=30,
        join_token=None, pending_deadline=starts,
    )
    session.add(t)
    await session.flush()
    p = Participant(
        tournament_id=t.id, user_id=alice.id, agent_id=agent.id,
        agent_name="alfa", total_score=42.0,
    )
    session.add(p)
    await session.commit()


@pytest.mark.anyio
async def test_hall_of_fame_returns_payload_shape(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
):
    await _seed(db_session)

    app = create_test_app()

    async def _override_session():
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/public/leaderboard/el-farol")
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == SCHEMA_VERSION
    assert "generated_at" in body
    assert body["total"] == 1
    assert body["limit"] == 50
    assert body["offset"] == 0
    entry = body["entries"][0]
    assert entry["agent_name"] == "alfa"
    assert entry["owner_username"] == "alice"
    assert entry["total_score"] == 42.0
    assert entry["tournaments_count"] == 1
    assert r.headers["Cache-Control"] == "public, s-maxage=60"


@pytest.mark.anyio
async def test_hall_of_fame_rejects_out_of_bounds(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
):
    app = create_test_app()

    async def _override_session():
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in (
            "/api/public/leaderboard/el-farol?limit=0",
            "/api/public/leaderboard/el-farol?limit=201",
            "/api/public/leaderboard/el-farol?offset=-1",
        ):
            r = await client.get(path)
            assert r.status_code == 422, path


@pytest.mark.anyio
async def test_hall_of_fame_pagination_keys_are_distinct_in_cache(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    monkeypatch,
):
    """Two consecutive requests with different ``offset`` must NOT share
    a cache hit. Patch ``_hall_of_fame_query`` to count calls."""
    await _seed(db_session)

    from atp.dashboard.v2.services import winners as winners_service

    real_query = winners_service._hall_of_fame_query
    calls: list[tuple[int, int]] = []

    async def _counting(sess, *, limit, offset):
        calls.append((limit, offset))
        return await real_query(sess, limit=limit, offset=offset)

    monkeypatch.setattr(winners_service, "_hall_of_fame_query", _counting)

    app = create_test_app()

    async def _override_session():
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/api/public/leaderboard/el-farol?limit=2&offset=0")
        await client.get("/api/public/leaderboard/el-farol?limit=2&offset=2")
        # Second hit at offset=0 should be cached.
        await client.get("/api/public/leaderboard/el-farol?limit=2&offset=0")

    # Two distinct (limit, offset) keys → two underlying queries; the
    # third request hits the cache and does not append.
    assert calls == [(2, 0), (2, 2)]
```

- [ ] **Step 5: Run the integration tests**

Run: `uv run pytest tests/integration/dashboard/winners/test_hall_of_fame_api.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/winners_api.py tests/integration/dashboard/winners/`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/routes/ tests/integration/dashboard/winners/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/winners_api.py packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py tests/integration/dashboard/winners/
git commit -m "feat(winners): /api/public/leaderboard/el-farol JSON endpoint"
```

---

## Task 5: Per-tournament HTML route + template

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_tournament.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Create: `tests/integration/dashboard/winners/test_winners_tournament_ui.py`

**Goal:** Render the per-tournament poster page; enforce all four access gates with generic 404; carry the same `Cache-Control` header.

- [ ] **Step 1: Create `routes/winners_ui.py`**

```python
# packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py
"""HTML routes for the El Farol winners surfaces."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse

from atp.dashboard.models import DEFAULT_TENANT_ID
from atp.dashboard.query_cache import QueryCache
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.services.winners import (
    CAPACITY_RATIO,
    _hall_of_fame_query,
    _winners_query,
    get_hof_cache,
    get_winners_cache,
)

router = APIRouter(tags=["winners-ui"])

_CACHE_CONTROL = "public, s-maxage=60"
_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
)


def _capacity_for(num_players: int) -> int:
    return max(1, int(CAPACITY_RATIO * num_players))


@router.get("/ui/tournaments/{tournament_id}/winners", response_class=HTMLResponse)
async def get_winners_html(
    request: Request,
    response: Response,
    tournament_id: int,
    session: DBSession,
) -> HTMLResponse:
    t = await session.get(Tournament, tournament_id)
    if t is None:
        raise _NOT_FOUND
    if t.tenant_id != DEFAULT_TENANT_ID:
        raise _NOT_FOUND
    if t.game_type != "el_farol":
        raise _NOT_FOUND
    if t.status != TournamentStatus.COMPLETED:
        raise _NOT_FOUND
    if t.join_token is not None:
        raise _NOT_FOUND

    cache = get_winners_cache()
    key = QueryCache._make_key("winners", tournament_id)
    entries = cache.get(key)
    if entries is None:
        entries = await _winners_query(session, tournament_id)
        cache.put(key, entries)

    name = (t.config or {}).get("name") or f"Tournament #{t.id}"
    duration: str | None = None
    if t.starts_at and t.ends_at:
        secs = int((t.ends_at - t.starts_at).total_seconds())
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        if h:
            duration = f"{h}h {m}m {s}s"
        elif m:
            duration = f"{m}m {s}s"
        else:
            duration = f"{s}s"

    templates = request.app.state.templates
    response.headers["Cache-Control"] = _CACHE_CONTROL
    return templates.TemplateResponse(
        "ui/winners_tournament.html",
        {
            "request": request,
            "active_page": "tournaments",
            "tournament": t,
            "tournament_name": name,
            "header": {
                "players": t.num_players,
                "days": t.total_rounds,
                "capacity": _capacity_for(t.num_players),
                "duration": duration,
            },
            "entries": entries,
        },
        headers={"Cache-Control": _CACHE_CONTROL},
    )
```

- [ ] **Step 2: Create the template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_tournament.html`:

```jinja
{% extends "ui/base_ui.html" %}

{% block title %}{{ tournament_name }} — Winners — ATP Platform{% endblock %}

{% block content %}
<h2>{{ tournament_name }} — Winners</h2>

<div class="tournament-meta">
    <span class="status-badge" style="background:#28a745">completed</span>
    <span>el_farol</span>
    <span>Players: {{ header.players }}</span>
    <span>Days: {{ header.days }}</span>
    <span>Capacity: {{ header.capacity }}</span>
    <span>Duration: {{ header.duration or "—" }}</span>
</div>

{% if entries %}
<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Bot</th>
            <th>Description</th>
            <th>Owner</th>
            <th>Score</th>
            <th>Tokens in</th>
            <th>Tokens out</th>
            <th>Cost</th>
            <th>Model</th>
        </tr>
    </thead>
    <tbody>
        {% for e in entries %}
        <tr{% if e.rank <= 3 %} style="background:#f8f5ff"{% endif %}>
            <td>
                {% if e.rank == 1 %}🥇{% elif e.rank == 2 %}🥈{% elif e.rank == 3 %}🥉{% endif %}
                {{ e.rank }}
            </td>
            <td>{{ e.agent_name }}</td>
            <td>{{ e.agent_description or "—" }}</td>
            <td>{{ e.owner_username }}</td>
            <td class="score-pair" style="color:#7c3aed">
                {{ "%.2f"|format(e.score) if e.score is not none else "—" }}
            </td>
            <td class="score-pair">{{ e.tokens_in if e.tokens_in is not none else "—" }}</td>
            <td class="score-pair">{{ e.tokens_out if e.tokens_out is not none else "—" }}</td>
            <td class="score-pair">{{ "%.4f"|format(e.cost_usd) if e.cost_usd is not none else "—" }}</td>
            <td>{{ e.model_id or "—" }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p style="color:#888">No final scores recorded.</p>
{% endif %}

<p style="margin-top:1.5rem">
    <a href="/ui/tournaments/{{ tournament.id }}">← Back to tournament</a>
</p>
{% endblock %}
```

- [ ] **Step 3: Mount the UI router in `factory.py`**

Edit `packages/atp-dashboard/atp/dashboard/v2/factory.py`. Find the block around line 248 that imports `ui_router` and `admin_ui_router`. Add a third import + include:

```python
from atp.dashboard.v2.routes.winners_ui import router as winners_ui_router
```

```python
app.include_router(winners_ui_router)
```

Place it after `app.include_router(ui_router)` and before `app.include_router(admin_ui_router)`.

- [ ] **Step 4: Write the integration test**

Create `tests/integration/dashboard/winners/test_winners_tournament_ui.py`:

```python
"""HTML route tests for /ui/tournaments/{id}/winners."""

from __future__ import annotations

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
from atp.dashboard.v2.services.winners import reset_caches_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_caches_for_tests()
    yield
    reset_caches_for_tests()


async def _seed_tournament(
    session: AsyncSession,
    *,
    status_: TournamentStatus = TournamentStatus.COMPLETED,
    join_token: str | None = None,
    game_type: str = "el_farol",
    tenant_id: str = DEFAULT_TENANT_ID,
    archived: bool = False,
) -> int:
    alice = User(
        username="alice", email="a@e.com", hashed_password="x",
        is_admin=False, is_active=True,
    )
    session.add(alice)
    await session.flush()
    agent = Agent(
        tenant_id=DEFAULT_TENANT_ID, name="alfa", agent_type="tournament",
        owner_id=alice.id, description="greedy", purpose="tournament",
        deleted_at=datetime(2026, 5, 1, 0, 0, 0) if archived else None,
    )
    session.add(agent)
    await session.flush()
    starts = datetime(2026, 5, 1, 12, 0, 0)
    t = Tournament(
        tenant_id=tenant_id, game_type=game_type, config={"name": "T1"},
        status=status_, starts_at=starts, ends_at=starts + timedelta(minutes=12, seconds=4),
        num_players=2, total_rounds=5, round_deadline_s=30,
        join_token=join_token, pending_deadline=starts,
    )
    session.add(t)
    await session.flush()
    p = Participant(
        tournament_id=t.id, user_id=alice.id, agent_id=agent.id,
        agent_name="alfa", total_score=42.0,
    )
    session.add(p)
    await session.commit()
    return t.id


def _make_client(test_database, app=None):
    if app is None:
        app = create_test_app()

    async def _override_session():
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session
    return ASGITransport(app=app), app


@pytest.mark.anyio
async def test_winners_page_renders_for_completed_public_tournament(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    tid = await _seed_tournament(db_session)
    transport, _app = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/ui/tournaments/{tid}/winners")
    assert r.status_code == 200
    assert "alfa" in r.text
    assert "Players: 2" in r.text
    assert "Days: 5" in r.text
    assert "Capacity: 1" in r.text  # max(1, int(0.6 * 2)) = 1
    assert "12m 4s" in r.text
    assert r.headers["Cache-Control"] == "public, s-maxage=60"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"status_": TournamentStatus.PENDING},
        {"status_": TournamentStatus.ACTIVE},
        {"status_": TournamentStatus.CANCELLED},
        {"join_token": "secret"},
        {"game_type": "prisoners_dilemma"},
        {"tenant_id": "other"},
    ],
)
async def test_winners_page_404_on_gate_violations(
    test_database: Database,
    db_session: AsyncSession,
    disable_dashboard_auth,
    kwargs,
):
    tid = await _seed_tournament(db_session, **kwargs)
    transport, _app = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/ui/tournaments/{tid}/winners")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_winners_page_404_for_missing_tournament(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    transport, _app = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ui/tournaments/9999/winners")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_winners_page_archived_agent_suffix(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    tid = await _seed_tournament(db_session, archived=True)
    transport, _app = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/ui/tournaments/{tid}/winners")
    assert r.status_code == 200
    assert "alfa (archived)" in r.text
```

- [ ] **Step 5: Run the integration tests**

Run: `uv run pytest tests/integration/dashboard/winners/test_winners_tournament_ui.py -v`
Expected: all PASS.

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py packages/atp-dashboard/atp/dashboard/v2/factory.py`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_tournament.html packages/atp-dashboard/atp/dashboard/v2/factory.py tests/integration/dashboard/winners/test_winners_tournament_ui.py
git commit -m "feat(winners): per-tournament winners HTML poster"
```

---

## Task 6: Hall of Fame HTML route + template

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_hall_of_fame.html`
- Create: `tests/integration/dashboard/winners/test_hall_of_fame_ui.py`

**Goal:** Render the public Hall of Fame at `/ui/leaderboard/el-farol` with HTMX-style pagination. Strict 422 on out-of-bounds; same `Cache-Control` header.

- [ ] **Step 1: Add the HoF route to `winners_ui.py`**

Append to `packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py`:

```python
@router.get("/ui/leaderboard/el-farol", response_class=HTMLResponse)
async def get_hall_of_fame_html(
    request: Request,
    response: Response,
    session: DBSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> HTMLResponse:
    cache = get_hof_cache()
    key = QueryCache._make_key("hall_of_fame", limit=limit, offset=offset)
    cached = cache.get(key)
    if cached is None:
        total, entries = await _hall_of_fame_query(
            session, limit=limit, offset=offset
        )
        cache.put(key, (total, entries))
    else:
        total, entries = cached

    page = (offset // limit) + 1
    total_pages = max(1, (total + limit - 1) // limit)

    templates = request.app.state.templates
    response.headers["Cache-Control"] = _CACHE_CONTROL
    return templates.TemplateResponse(
        "ui/winners_hall_of_fame.html",
        {
            "request": request,
            "active_page": "el_farol_hall_of_fame",
            "entries": entries,
            "total": total,
            "limit": limit,
            "offset": offset,
            "page": page,
            "total_pages": total_pages,
        },
        headers={"Cache-Control": _CACHE_CONTROL},
    )
```

- [ ] **Step 2: Create the HoF template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_hall_of_fame.html`:

```jinja
{% extends "ui/base_ui.html" %}

{% block title %}El Farol Hall of Fame — ATP Platform{% endblock %}

{% block content %}
<h2>El Farol Hall of Fame</h2>
<p>
    Top user-built agents by total score across all completed public
    El Farol tournaments.
</p>
<p style="color:#888;font-size:0.85em">
    Note: rank reflects pagination order; ties may split across pages.
</p>

{% if entries %}
<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Bot</th>
            <th>Description</th>
            <th>Owner</th>
            <th>Total score</th>
            <th>Tournaments</th>
        </tr>
    </thead>
    <tbody>
        {% for e in entries %}
        <tr>
            <td>{{ e.rank }}</td>
            <td>{{ e.agent_name }}</td>
            <td>{{ e.agent_description or "—" }}</td>
            <td>{{ e.owner_username }}</td>
            <td class="score-pair">{{ "%.2f"|format(e.total_score) }}</td>
            <td class="score-pair">{{ e.tournaments_count }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

{% if total_pages > 1 %}
<nav style="margin-top:1rem; display:flex; gap:0.5rem; justify-content:center;">
    {% for p in range(1, total_pages + 1) %}
    {% set p_offset = (p - 1) * limit %}
    <a href="/ui/leaderboard/el-farol?limit={{ limit }}&offset={{ p_offset }}"
       {% if p == page %}style="font-weight:bold"{% endif %}>{{ p }}</a>
    {% endfor %}
</nav>
{% endif %}
{% else %}
<p style="color:#888">No completed El Farol tournaments yet.</p>
{% endif %}

<p style="margin-top:1.5rem">
    <a href="/api/public/leaderboard/el-farol?limit={{ limit }}&offset={{ offset }}"
       style="color:#7c3aed">View raw JSON →</a>
</p>
{% endblock %}
```

- [ ] **Step 3: Write the integration test**

Create `tests/integration/dashboard/winners/test_hall_of_fame_ui.py`:

```python
"""HTML tests for /ui/leaderboard/el-farol."""

from __future__ import annotations

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
from atp.dashboard.v2.services.winners import reset_caches_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_caches_for_tests()
    yield
    reset_caches_for_tests()


def _make_client(test_database):
    app = create_test_app()

    async def _override_session():
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session
    return ASGITransport(app=app)


@pytest.mark.anyio
async def test_hof_empty_state(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    transport = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ui/leaderboard/el-farol")
    assert r.status_code == 200
    assert "No completed El Farol tournaments yet." in r.text
    assert r.headers["Cache-Control"] == "public, s-maxage=60"


@pytest.mark.anyio
async def test_hof_renders_top_agents(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    starts = datetime(2026, 5, 1, 12, 0, 0)
    alice = User(
        username="alice", email="a@e.com", hashed_password="x",
        is_admin=False, is_active=True,
    )
    bob = User(
        username="bob", email="b@e.com", hashed_password="x",
        is_admin=False, is_active=True,
    )
    db_session.add_all([alice, bob])
    await db_session.flush()
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID, name="alfa", agent_type="tournament",
        owner_id=alice.id, purpose="tournament",
    )
    b = Agent(
        tenant_id=DEFAULT_TENANT_ID, name="beta", agent_type="tournament",
        owner_id=bob.id, purpose="tournament",
    )
    db_session.add_all([a, b])
    await db_session.flush()
    t = Tournament(
        tenant_id=DEFAULT_TENANT_ID, game_type="el_farol",
        config={"name": "T"}, status=TournamentStatus.COMPLETED,
        starts_at=starts, ends_at=starts + timedelta(minutes=10),
        num_players=2, total_rounds=5, round_deadline_s=30,
        join_token=None, pending_deadline=starts,
    )
    db_session.add(t)
    await db_session.flush()
    db_session.add_all([
        Participant(
            tournament_id=t.id, user_id=alice.id, agent_id=a.id,
            agent_name="alfa", total_score=20.0,
        ),
        Participant(
            tournament_id=t.id, user_id=bob.id, agent_id=b.id,
            agent_name="beta", total_score=30.0,
        ),
    ])
    await db_session.commit()

    transport = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ui/leaderboard/el-farol")
    assert r.status_code == 200
    # bob (30) ranks above alice (20) by total score
    bob_idx = r.text.find("beta")
    alice_idx = r.text.find("alfa")
    assert 0 < bob_idx < alice_idx


@pytest.mark.anyio
async def test_hof_strict_bounds(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    transport = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in (
            "/ui/leaderboard/el-farol?limit=0",
            "/ui/leaderboard/el-farol?limit=201",
            "/ui/leaderboard/el-farol?offset=-1",
        ):
            r = await client.get(path)
            assert r.status_code == 422, path
```

- [ ] **Step 4: Run the integration tests**

Run: `uv run pytest tests/integration/dashboard/winners/test_hall_of_fame_ui.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Run the full winners test suite**

Run: `uv run pytest tests/unit/dashboard/winners/ tests/integration/dashboard/winners/ -v`
Expected: all PASS.

- [ ] **Step 6: Type-check + lint**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py`
Run: `uv run ruff check packages/atp-dashboard/atp/dashboard/v2/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/winners_hall_of_fame.html tests/integration/dashboard/winners/test_hall_of_fame_ui.py
git commit -m "feat(winners): /ui/leaderboard/el-farol Hall of Fame page"
```

---

## Task 7: Add "Winners →" link in `tournament_detail.html`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
- Modify: `tests/integration/dashboard/test_tournament_detail_ui.py` (or create a new small test if you prefer; existing file already covers the page)

**Goal:** Discoverability — a link to the new winners page only when the gates are met.

- [ ] **Step 1: Edit `tournament_detail.html`**

Open `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`. Inside the `<div class="tournament-meta">` block (around line 26-54), find the El Farol cards-replay branch:

```jinja
    {% if t.game_type == "el_farol" %}
      {% if cards_match_id %}
      <span><a href="/ui/matches/{{ cards_match_id }}">Cards replay →</a></span>
      {% else %}
      <span><a href="/ui/matches">Browse El Farol matches →</a></span>
      {% endif %}
    {% endif %}
```

Add immediately after that block (still inside `tournament-meta`):

```jinja
    {% if t.game_type == "el_farol" and t.status == "completed" and not t.join_token %}
    <span><a href="/ui/tournaments/{{ t.id }}/winners">Winners →</a></span>
    {% endif %}
```

- [ ] **Step 2: Add a small assertion to an existing tournament-detail test**

Open `tests/integration/dashboard/test_tournament_detail_ui.py`. Find the test that covers a completed public El Farol tournament (search for `el_farol` and `COMPLETED` together; pick the first such test). At the end of that test's HTML assertions, add:

```python
    assert "/winners" in response.text
    assert "Winners →" in response.text
```

If no test currently covers a completed public El Farol tournament, add a new test:

```python
@pytest.mark.anyio
async def test_tournament_detail_shows_winners_link_for_completed_public_el_farol(
    test_database: Database, db_session: AsyncSession, disable_dashboard_auth,
):
    # Reuse the seeding helper from test_winners_tournament_ui if importable;
    # otherwise inline a minimal Tournament(status=COMPLETED, game_type='el_farol',
    # join_token=None) with one Participant.
    ...
    transport = _make_client(test_database)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/ui/tournaments/{tid}")
    assert r.status_code == 200
    assert f'/ui/tournaments/{tid}/winners' in r.text
    assert "Winners →" in r.text
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/integration/dashboard/test_tournament_detail_ui.py -v`
Expected: PASS.

- [ ] **Step 4: Sanity-check that the link does NOT appear for pending/active/cancelled/private tournaments**

Run: `uv run pytest tests/integration/dashboard/test_tournament_detail_ui.py -v -k 'pending or active or cancelled or private'`
Expected: PASS — confirm no test asserts "Winners →" appears in those branches; if any do, the link guard is wrong.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html tests/integration/dashboard/test_tournament_detail_ui.py
git commit -m "feat(winners): Winners → link on completed public el_farol tournaments"
```

---

## Task 8: Sidebar "El Farol Hall of Fame" link

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`

**Goal:** Discoverable Hall of Fame in the global nav.

- [ ] **Step 1: Edit `base_ui.html`**

Open `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`. Find the `<ul class="sidebar-nav">` block. Add a new flat entry **after** the existing `Leaderboard` link:

```jinja
                <li><a href="/ui/leaderboard/el-farol" class="{% if active_page == 'el_farol_hall_of_fame' %}active{% endif %}">El Farol Hall of Fame</a></li>
```

- [ ] **Step 2: Quick smoke check that the link renders**

Run: `uv run pytest tests/integration/dashboard/winners/test_hall_of_fame_ui.py::test_hof_empty_state -v`
The page extends `base_ui.html`, so the new sidebar link is in the HTML. Add an inline assertion if not already covered:

Append to `test_hof_empty_state`:

```python
    assert "El Farol Hall of Fame" in r.text
```

Run again: `uv run pytest tests/integration/dashboard/winners/test_hall_of_fame_ui.py::test_hof_empty_state -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html tests/integration/dashboard/winners/test_hall_of_fame_ui.py
git commit -m "feat(winners): sidebar link to El Farol Hall of Fame"
```

---

## Task 9: Smoke-test extension

**Files:**
- Modify: `scripts/smoke_el_farol_prod.py`

**Goal:** End-to-end check on prod data — after a tournament finishes, fetch the winners page and confirm the agent name shows up.

- [ ] **Step 1: Locate the relevant section**

Open `scripts/smoke_el_farol_prod.py`. Find where the tournament transitions to `completed` status (search for `COMPLETED` or `status == "completed"`).

- [ ] **Step 2: Add the fetch + assertion**

First, confirm `httpx` is already imported (the script almost certainly imports it for its other HTTP calls — search for `import httpx`). If not, add `import httpx` at the top.

Locate the existing variable that holds the list of agent names the script registered (commonly named something like `agent_names`, `bots`, `participants`, or similar — grep within the script for the names it joins/registers). Use that variable below as `EXISTING_AGENT_NAMES_VAR`.

After the tournament-completed block, add:

```python
    # Smoke-check the winners page renders for the just-completed tournament.
    winners_url = f"{base_url}/ui/tournaments/{tournament_id}/winners"
    resp = httpx.get(winners_url, timeout=10.0)
    assert resp.status_code == 200, (
        f"winners page HTTP {resp.status_code} for tournament {tournament_id}"
    )
    # Each participant's agent_name should appear at least once.
    for agent_name in EXISTING_AGENT_NAMES_VAR:  # <- replace with actual variable
        assert agent_name in resp.text, (
            f"missing agent {agent_name!r} on winners page"
        )
    print(f"  ✓ winners page renders ({len(EXISTING_AGENT_NAMES_VAR)} agents)")
```

If the script does not currently track agent names as a list, fall back to a presence check that any agent name participant rows show up:

```python
    assert "<table>" in resp.text and "<td>" in resp.text, (
        "winners page rendered but contains no result table"
    )
```

- [ ] **Step 3: Manually run the smoke (optional, requires staging)**

Run: `uv run python scripts/smoke_el_farol_prod.py --base-url=https://atp-staging.pr0sto.space`
Expected: existing checks pass; new winners-page check passes.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_el_farol_prod.py
git commit -m "test(winners): extend prod smoke to fetch winners page"
```

---

## Final verification

- [ ] **Step 1: Run the entire winners test surface**

Run: `uv run pytest tests/unit/dashboard/winners/ tests/integration/dashboard/winners/ tests/integration/dashboard/test_tournament_detail_ui.py -v`
Expected: all PASS.

- [ ] **Step 2: Type-check + lint the whole new surface**

Run: `uv run pyrefly check packages/atp-dashboard/atp/dashboard/v2/services/winners.py packages/atp-dashboard/atp/dashboard/v2/routes/winners_api.py packages/atp-dashboard/atp/dashboard/v2/routes/winners_ui.py`
Run: `uv run ruff format .`
Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 3: Coverage check**

Run: `uv run pytest tests/unit/dashboard/winners/ tests/integration/dashboard/winners/ --cov=atp.dashboard.v2.services.winners --cov=atp.dashboard.v2.routes.winners_api --cov=atp.dashboard.v2.routes.winners_ui --cov-report=term-missing`
Expected: ≥80 % coverage on each of the three new modules.

- [ ] **Step 4: Final ad-hoc smoke**

Run the dashboard locally: `uv run atp dashboard` (default port 8080), open `http://127.0.0.1:8080/ui/leaderboard/el-farol`, verify the empty state renders. If you have a completed public El Farol tournament locally, also visit `/ui/tournaments/{id}/winners`.

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feat/tournament-shrink
```

(or whatever branch you're on — confirm with `git rev-parse --abbrev-ref HEAD` first)
