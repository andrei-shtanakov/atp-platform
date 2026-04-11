"""Integration tests for idempotent join under concurrency."""

import asyncio

import pytest
from sqlalchemy import func, select, text

from atp.dashboard.tournament.errors import ConflictError
from atp.dashboard.tournament.models import Participant
from atp.dashboard.tournament.service import TournamentService

# The actual factory and fixture helpers will grow as later tasks
# introduce more integration tests. For this task, build minimal
# seed helpers inline.


async def _seed_user(session, user_id: int, username: str):
    # Use raw SQL for test seeding to avoid coupling to User model imports.
    await session.execute(
        text(
            "INSERT INTO users "
            "(id, tenant_id, username, email, hashed_password, "
            "is_active, is_admin, created_at, updated_at) "
            "VALUES (:id, 'default', :u, :e, 'x', 1, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": user_id, "u": username, "e": f"{username}@test.com"},
    )


async def _seed_tournament(session, tournament_id: int):
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            "total_rounds, round_deadline_s, pending_deadline, created_at) "
            "VALUES (:id, 'default', 'prisoners_dilemma', '{}', '{}', 'pending', "
            "2, 3, 30, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": tournament_id},
    )


@pytest.mark.anyio
async def test_sc5_five_concurrent_joins_same_tournament_all_idempotent(
    session_factory,
):
    """SC-5: five concurrent join calls from the same user to the SAME
    tournament must all succeed idempotently. Exactly one row in DB."""
    async with session_factory() as setup:
        await _seed_user(setup, user_id=1, username="alice")
        await _seed_tournament(setup, tournament_id=1)
        await setup.commit()

    async def one_join():
        async with session_factory() as session:
            # Fetch User inside this session's transaction
            user_row = await session.get(
                __import__("atp.dashboard.models", fromlist=["User"]).User, 1
            )
            svc = TournamentService(session, bus=_DummyBus())
            result = await svc.join(tournament_id=1, user=user_row, agent_name="bot")
            await session.commit()
            return result

    results = await asyncio.gather(*[one_join() for _ in range(5)])

    # All five calls returned successfully
    assert len(results) == 5
    for participant, is_new in results:
        assert participant is not None

    # Exactly one is_new=True
    new_flags = [is_new for _, is_new in results]
    assert new_flags.count(True) == 1
    assert new_flags.count(False) == 4

    # DB has exactly one row
    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.tournament_id == 1)
            .where(Participant.user_id == 1)
        )
        assert count == 1


@pytest.mark.anyio
async def test_sc4_two_concurrent_joins_different_tournaments_one_wins(
    session_factory,
):
    """SC-4: same user joining two different tournaments concurrently
    → one success + one ConflictError via uq_participant_user_active."""
    async with session_factory() as setup:
        await _seed_user(setup, user_id=1, username="alice")
        await _seed_tournament(setup, tournament_id=1)
        await _seed_tournament(setup, tournament_id=2)
        await setup.commit()

    async def one_join(tournament_id):
        async with session_factory() as session:
            user_row = await session.get(
                __import__("atp.dashboard.models", fromlist=["User"]).User, 1
            )
            svc = TournamentService(session, bus=_DummyBus())
            try:
                result = await svc.join(
                    tournament_id=tournament_id,
                    user=user_row,
                    agent_name="bot",
                )
                await session.commit()
                return result
            except ConflictError as e:
                return e

    results = await asyncio.gather(one_join(1), one_join(2))
    successes = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, ConflictError)]

    assert len(successes) == 1
    assert len(conflicts) == 1


class _DummyBus:
    async def publish(self, event):
        pass
