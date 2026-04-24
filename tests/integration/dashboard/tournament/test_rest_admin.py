"""Integration tests for the 6 REST admin endpoints.

Uses dependency override pattern (same as test_benchmark_api.py) to wire
the Plan 2a test DB into the FastAPI app without touching lifespan init_database.
All tests are async (anyio) so they can consume async fixtures.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import User
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app
from atp.dashboard.v2.routes.tournament_api import get_current_user_for_tournament

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def tournament_db_database(tournament_db):
    """Register the plan-2a DB engine as the global Database singleton.

    Uses the sync URL from the async engine to construct a proper Database
    object so auth helpers (which call get_database()) work correctly.
    """
    # tournament_db is an AsyncEngine; derive the aiosqlite URL from it.
    raw_url = str(tournament_db.url)
    # Async engine uses sqlite+aiosqlite:///... but the URL may already be async.
    if not raw_url.startswith("sqlite+aiosqlite"):
        raw_url = raw_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    db = Database(url=raw_url, echo=False)
    # Reuse the existing engine so we share the same WAL-mode connection.
    db._engine = tournament_db
    set_database(db)
    yield db
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
async def seeded_user(tournament_db_database) -> User:
    """Insert user id=1 (admin) and return a detached User object."""
    async with tournament_db_database.session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', "
                "1, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()
    return User(
        id=1,
        username="alice",
        email="alice@test.com",
        hashed_password="x",
        is_active=True,
        is_admin=True,
    )


@pytest.fixture
async def seeded_tournament(tournament_db_database, seeded_user) -> None:
    """Insert tournament id=1 (pending)."""
    async with tournament_db_database.session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO tournaments "
                "(id, tenant_id, game_type, config, rules, status, "
                "num_players, total_rounds, round_deadline_s, "
                "pending_deadline, created_by, created_at) "
                "VALUES (1, 'default', 'prisoners_dilemma', "
                "'{\"name\": \"test\"}', '{}', 'pending', 2, 3, 30, "
                "CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()


@pytest.fixture
def _app(tournament_db_database, seeded_user):
    """FastAPI test app with DB and auth overridden."""
    app = create_test_app()
    sf = tournament_db_database.session_factory

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with sf() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def override_current_user() -> User:
        return seeded_user

    app.dependency_overrides[get_db_session] = override_get_session
    app.dependency_overrides[get_current_user_for_tournament] = override_current_user
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_tournaments_returns_200(_app, seeded_tournament):
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments")
    assert response.status_code == 200
    body = response.json()
    assert "tournaments" in body


@pytest.mark.anyio
async def test_get_tournament_detail_returns_200(_app, seeded_tournament):
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments/1")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert "join_token" not in body  # never serialized on GET
    assert "has_join_token" in body


@pytest.mark.anyio
async def test_get_tournament_missing_returns_404(_app, seeded_tournament):
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments/99999")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_get_rounds_returns_200(_app, seeded_tournament):
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments/1/rounds")
    assert response.status_code == 200
    assert "rounds" in response.json()


@pytest.fixture
async def seeded_tournament_with_actions(
    tournament_db_database, seeded_tournament
) -> None:
    """Extend seeded tournament with 2 participants, a completed round, and
    two actions (one with reasoning, one without)."""
    async with tournament_db_database.session_factory() as s:
        # Extra user id=2 (non-admin) for multi-participant rows
        await s.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (2, 'default', 'bob', 'bob@test.com', 'x', "
                "1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        # LABS-TSA PR-4: seed owning agents so Participant rows satisfy
        # the agent-xor-builtin CHECK.
        await s.execute(
            text(
                "INSERT INTO agents "
                "(id, tenant_id, name, agent_type, purpose, config, "
                "owner_id, created_at, updated_at) "
                "VALUES "
                "(1, 'default', 'alice', 'mcp', 'tournament', '{}', 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
                "(2, 'default', 'bob', 'mcp', 'tournament', '{}', 2, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await s.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_id, agent_name, "
                "total_score, joined_at) "
                "VALUES "
                "(1, 1, 1, 'alice', 3.0, CURRENT_TIMESTAMP), "
                "(1, 2, 2, 'bob', 3.0, CURRENT_TIMESTAMP)"
            )
        )
        await s.execute(
            text(
                "INSERT INTO tournament_rounds "
                "(id, tournament_id, round_number, state, status, started_at) "
                "VALUES (1, 1, 1, '{}', 'completed', CURRENT_TIMESTAMP)"
            )
        )
        await s.execute(
            text(
                "INSERT INTO tournament_actions "
                "(round_id, participant_id, action_data, submitted_at, "
                "payoff, source, reasoning) "
                "VALUES "
                '(1, 1, \'{"choice": "cooperate"}\', CURRENT_TIMESTAMP, '
                "3.0, 'submitted', 'I trust the opponent'), "
                '(1, 2, \'{"choice": "cooperate"}\', CURRENT_TIMESTAMP, '
                "3.0, 'submitted', NULL)"
            )
        )
        await s.commit()


@pytest.mark.anyio
async def test_get_rounds_includes_actions_with_reasoning(
    _app, seeded_tournament_with_actions
):
    """Admin caller sees every action's reasoning (including nulls)."""
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments/1/rounds")
    assert response.status_code == 200
    body = response.json()
    assert "rounds" in body
    assert len(body["rounds"]) == 1
    r = body["rounds"][0]
    assert r["round_number"] == 1
    assert r["status"] == "completed"
    # New shape: nested actions with action_data, payoff, reasoning
    assert "actions" in r
    assert len(r["actions"]) == 2
    names = {a["agent_name"] for a in r["actions"]}
    assert names == {"alice", "bob"}
    alice_row = next(a for a in r["actions"] if a["agent_name"] == "alice")
    bob_row = next(a for a in r["actions"] if a["agent_name"] == "bob")
    # Admin sees reasoning verbatim
    assert alice_row["reasoning"] == "I trust the opponent"
    # Null reasoning still serializes as null (not omitted)
    assert bob_row["reasoning"] is None
    # action_data and payoff always present regardless of gate
    assert alice_row["action_data"] == {"choice": "cooperate"}
    assert alice_row["payoff"] == 3.0


@pytest.mark.anyio
async def test_get_rounds_empty_tournament_returns_empty_list(_app, seeded_tournament):
    """Tournament with zero rounds returns {'rounds': []}, not 404."""
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments/1/rounds")
    assert response.status_code == 200
    assert response.json() == {"rounds": []}


@pytest.mark.anyio
async def test_get_participants_returns_200(_app, seeded_tournament):
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/tournaments/1/participants")
    assert response.status_code == 200
    assert "participants" in response.json()


@pytest.mark.anyio
async def test_create_tournament_returns_201_with_token(
    _app, seeded_user, tournament_db_database
):
    # LABS-TSA PR-4: private tournaments require the creator either to
    # own a tournament-purpose agent or to provide a full builtin roster.
    # Seed one tournament agent for the seeded user so this pre-existing
    # test keeps validating just the 201 + join_token behaviour.
    async with tournament_db_database.session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO agents "
                "(tenant_id, name, agent_type, purpose, config, owner_id, "
                "created_at, updated_at) "
                "VALUES ('default', 'alice-tournament-agent', 'mcp', "
                "'tournament', '{}', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()

    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/api/v1/tournaments",
            json={
                "name": "new",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": True,
            },
        )
    assert response.status_code == 201
    body = response.json()
    assert "join_token" in body
    assert body["join_token"] is not None  # returned once on create for private


@pytest.mark.anyio
async def test_cancel_tournament_returns_200(_app, seeded_tournament):
    async with AsyncClient(
        transport=ASGITransport(app=_app), base_url="http://test"
    ) as c:
        response = await c.post("/api/v1/tournaments/1/cancel")
    assert response.status_code == 200
    assert response.json()["cancelled"] is True
