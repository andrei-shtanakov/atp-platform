"""Regression tests for the unified home activity feed and the tournament
leaderboard section on /ui/leaderboard."""

import os
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import get_password_hash
from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.tournament.models import Participant, TournamentStatus
from atp.dashboard.tournament.models import Tournament as TournamentModel
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def fresh_app():
    # Save env vars so they can be restored — leaving ATP_DISABLE_AUTH=true
    # in the process env leaks into other tests (e.g. unauth_client tests
    # in test_api.py expect 401 from endpoints with auth disabled get 200/201).
    saved_env = {
        k: os.environ.get(k)
        for k in ("ATP_SECRET_KEY", "ATP_DISABLE_AUTH", "ATP_RATE_LIMIT_ENABLED")
    }

    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "true"
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
        disable_auth=True,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    yield app, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    for key, old_val in saved_env.items():
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val
    get_config.cache_clear()


@pytest.mark.anyio
async def test_home_feed_mixes_runs_and_tournaments(fresh_app: tuple) -> None:
    """Both Run and Tournament rows must appear in Recent Activity."""
    app, db = fresh_app
    now = datetime.now()

    async with db.session() as session:
        owner = User(
            username="owner",
            email="owner@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(owner)
        await session.flush()
        bench = Benchmark(name="bench-1")
        session.add(bench)
        await session.flush()
        session.add(
            Run(
                benchmark_id=bench.id,
                user_id=owner.id,
                agent_name="e2e-demo",
                status=RunStatus.COMPLETED,
                started_at=now - timedelta(minutes=2),
            )
        )
        session.add(
            TournamentModel(
                game_type="prisoners_dilemma",
                status=TournamentStatus.COMPLETED,
                pending_deadline=now + timedelta(minutes=30),
                created_at=now - timedelta(minutes=1),
                ends_at=now - timedelta(minutes=1),
            )
        )
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/")

    assert resp.status_code == 200
    body = resp.text
    assert "Run #" in body
    assert "e2e-demo" in body
    assert "Tournament #" in body
    assert "prisoners_dilemma" in body


@pytest.mark.anyio
async def test_tournament_leaderboard_requires_game_type(fresh_app: tuple) -> None:
    """Without ?game_type, tournament section shows the 'select a game' hint."""
    app, _ = fresh_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/leaderboard")
    assert resp.status_code == 200
    assert "Select a game above" in resp.text


@pytest.mark.anyio
async def test_tournament_leaderboard_aggregates_by_game_type(
    fresh_app: tuple,
) -> None:
    """With ?game_type=X, entries show cumulative total_score per agent."""
    app, db = fresh_app
    now = datetime.now()

    async with db.session() as session:
        t1 = TournamentModel(
            game_type="prisoners_dilemma",
            status=TournamentStatus.COMPLETED,
            pending_deadline=now + timedelta(minutes=30),
        )
        t2 = TournamentModel(
            game_type="prisoners_dilemma",
            status=TournamentStatus.COMPLETED,
            pending_deadline=now + timedelta(minutes=30),
        )
        session.add_all([t1, t2])
        await session.flush()

        session.add_all(
            [
                Participant(
                    tournament_id=t1.id,
                    user_id=1,
                    agent_name="cooperator",
                    total_score=90.0,
                ),
                Participant(
                    tournament_id=t2.id,
                    user_id=2,
                    agent_name="cooperator",
                    total_score=85.0,
                ),
                Participant(
                    tournament_id=t1.id,
                    user_id=3,
                    agent_name="defector",
                    total_score=40.0,
                ),
            ]
        )
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/leaderboard?game_type=prisoners_dilemma")
    assert resp.status_code == 200
    body = resp.text
    # cooperator: 90+85 = 175; defector: 40. Order matters.
    assert body.index("cooperator") < body.index("defector")
    assert "175.00" in body
    assert "40.00" in body
