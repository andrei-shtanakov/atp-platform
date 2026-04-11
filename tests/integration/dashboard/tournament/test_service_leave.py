"""Integration tests for leave() and last-participant abandoned cascade."""

import pytest
from sqlalchemy import text

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import (
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


class _CapturedBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


async def _seed_minimal(session, num_participants: int):
    for uid, name in [(1, "alice"), (2, "bob")]:
        await session.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (:id, 'default', :u, :e, 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": uid, "u": name, "e": f"{name}@test.com"},
        )
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            "total_rounds, round_deadline_s, pending_deadline, created_at) "
            "VALUES (1, 'default', 'prisoners_dilemma', '{\"name\": \"t\"}', '{}', "
            "'active', 2, 3, 30, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )
    for i in range(1, num_participants + 1):
        await session.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name, joined_at) "
                "VALUES (1, :uid, :name, CURRENT_TIMESTAMP)"
            ),
            {"uid": i, "name": f"p{i}"},
        )


async def _load_user(session, user_id):
    from atp.dashboard.models import User

    return await session.get(User, user_id)


@pytest.mark.anyio
async def test_leave_sets_released_at(session_factory):
    async with session_factory() as setup:
        await _seed_minimal(setup, num_participants=2)
        await setup.commit()

    bus = _CapturedBus()
    async with session_factory() as session:
        user1 = await _load_user(session, 1)
        svc = TournamentService(session, bus)
        await svc.leave(tournament_id=1, user=user1)
        await session.commit()

    async with session_factory() as verify:
        p = await verify.scalar(
            text("SELECT released_at FROM tournament_participants WHERE user_id = 1")
        )
        assert p is not None  # released_at populated


@pytest.mark.anyio
async def test_leave_non_participant_raises_not_found(session_factory):
    async with session_factory() as setup:
        await _seed_minimal(setup, num_participants=2)
        await setup.commit()

    bus = _CapturedBus()
    async with session_factory() as session:
        # User 99 does not exist as participant
        from atp.dashboard.models import User

        user = User(id=99, username="ghost", is_admin=False, tenant_id="default")
        svc = TournamentService(session, bus)
        with pytest.raises(NotFoundError):
            await svc.leave(tournament_id=1, user=user)


@pytest.mark.anyio
async def test_last_participant_leave_triggers_abandoned_cascade(session_factory):
    async with session_factory() as setup:
        await _seed_minimal(setup, num_participants=2)
        await setup.commit()

    bus = _CapturedBus()
    async with session_factory() as session:
        user1 = await _load_user(session, 1)
        svc = TournamentService(session, bus)
        await svc.leave(tournament_id=1, user=user1)
        await session.commit()

    # User 1 left — tournament still has user 2 active, so no cascade
    async with session_factory() as session:
        user2 = await _load_user(session, 2)
        svc = TournamentService(session, bus)
        await svc.leave(tournament_id=1, user=user2)
        await session.commit()

    # Tournament is now abandoned
    async with session_factory() as verify:
        t = await verify.get(Tournament, 1)
        assert t.status == TournamentStatus.CANCELLED
        assert t.cancelled_reason == CancelReason.ABANDONED
        assert t.cancelled_by is None

    cancel_events = [e for e in bus.events if isinstance(e, TournamentCancelEvent)]
    assert len(cancel_events) == 1
    assert cancel_events[0].cancelled_reason == CancelReason.ABANDONED
