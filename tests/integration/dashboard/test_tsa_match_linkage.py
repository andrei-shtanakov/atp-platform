"""LABS-TSA PR-5 — GameResult dual-write at tournament completion.

Exercises the hook in ``TournamentService._complete_tournament`` that writes
a companion ``GameResult`` row (with a fresh UUID ``match_id`` and the
tournament's ``tournament_id``) so ``/ui/matches`` can surface tournament
outcomes through the existing match-listing machinery.

The UNIQUE partial index ``uq_game_results_tournament_id`` (PR-1) makes
the second completion attempt idempotent — the second insert hits
IntegrityError which the service absorbs inside a SAVEPOINT.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import atp.dashboard.tournament.models  # noqa: F401 (ensure model registration)
from atp.dashboard.models import Base, GameResult, User
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.service import TournamentService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Fresh in-memory SQLite + all tables, one per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_local = async_sessionmaker(engine, expire_on_commit=False)
    async with session_local() as sess:
        yield sess
    await engine.dispose()


@pytest.fixture
async def creator(session: AsyncSession) -> User:
    user = User(
        username="creator",
        email="c@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest.fixture
def bus() -> TournamentEventBus:
    return TournamentEventBus()


class TestDualWrite:
    @pytest.mark.anyio
    async def test_completion_creates_game_result_with_tournament_id(
        self,
        session: AsyncSession,
        creator: User,
        bus: TournamentEventBus,
    ) -> None:
        """Completing a tournament writes a GameResult row with the
        tournament_id set and a fresh UUID match_id (never the join_token).
        """
        svc = TournamentService(session, bus)
        t = Tournament(
            game_type="el_farol",
            num_players=2,
            total_rounds=1,
            round_deadline_s=30,
            created_by=creator.id,
            status=TournamentStatus.ACTIVE,
            join_token="secret-private-token",
        )
        session.add(t)
        await session.commit()

        await svc._complete_tournament(t)
        await session.commit()

        rows = (
            (
                await session.execute(
                    select(GameResult).where(GameResult.tournament_id == t.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.match_id is not None
        # match_id is a fresh UUID, never the join_token
        uuid.UUID(row.match_id)  # parses — raises if not a UUID
        assert row.match_id != t.join_token
        assert row.tournament_id == t.id
        assert row.status == "completed"
        assert row.game_name == "el_farol"
        assert row.num_players == 2
        assert row.num_rounds == 1
        # Phase-7 JSON columns are NULL placeholders until a follow-up
        # ticket ports the full Round/Action reshape. NULL (not empty
        # list) keeps these rows out of the /ui/matches renderability
        # filter so they don't show up as "pre-Phase-7" on detail.
        assert row.actions_json is None
        assert row.day_aggregates_json is None
        assert row.round_payoffs_json is None
        assert row.agents_json is None

    @pytest.mark.anyio
    async def test_idempotent_on_double_completion(
        self,
        session: AsyncSession,
        creator: User,
        bus: TournamentEventBus,
    ) -> None:
        """A second completion call is absorbed by the UNIQUE partial index
        + SAVEPOINT — the GameResult row count stays at 1.
        """
        svc = TournamentService(session, bus)
        t = Tournament(
            game_type="el_farol",
            num_players=2,
            total_rounds=1,
            round_deadline_s=30,
            created_by=creator.id,
            status=TournamentStatus.ACTIVE,
        )
        session.add(t)
        await session.commit()

        await svc._complete_tournament(t)
        await session.commit()
        await svc._complete_tournament(t)  # second call no-ops on match linkage
        await session.commit()

        rows = (
            (
                await session.execute(
                    select(GameResult).where(GameResult.tournament_id == t.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1  # UNIQUE partial index neutralises the second write
