"""Integration tests for the three cancel paths (user/system/abandoned)
and their full cascade effects."""

import pytest
from sqlalchemy import func, select, text

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import (
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


class _CapturingBus:
    def __init__(self) -> None:
        self.events: list = []

    async def publish(self, event: object) -> None:
        self.events.append(event)


async def _seed_active_tournament_with_rounds(session) -> None:
    for uid, name, is_admin in [(1, "alice", 0), (2, "bob", 0), (99, "admin", 1)]:
        await session.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (:id, 'default', :u, :e, 'x', 1, :a, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": uid, "u": name, "e": f"{name}@test.com", "a": is_admin},
        )
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            "total_rounds, round_deadline_s, pending_deadline, created_by, created_at) "
            "VALUES (1, 'default', 'prisoners_dilemma', '{\"name\": \"t\"}', '{}', "
            "'active', 2, 3, 30, CURRENT_TIMESTAMP, 99, CURRENT_TIMESTAMP)"
        )
    )
    # LABS-TSA PR-4: agent-xor-builtin CHECK requires agent_id.
    await session.execute(
        text(
            "INSERT INTO agents "
            "(id, tenant_id, name, agent_type, purpose, config, owner_id, "
            "created_at, updated_at) "
            "VALUES "
            "(1, 'default', 'a', 'mcp', 'tournament', '{}', 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
            "(2, 'default', 'b', 'mcp', 'tournament', '{}', 2, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )
    await session.execute(
        text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_id, agent_name, joined_at) VALUES "
            "(1, 1, 1, 'a', CURRENT_TIMESTAMP), (1, 2, 2, 'b', CURRENT_TIMESTAMP)"
        )
    )
    await session.execute(
        text(
            "INSERT INTO tournament_rounds "
            "(tournament_id, round_number, status, state, started_at) VALUES "
            "(1, 1, 'completed', '{}', CURRENT_TIMESTAMP), "
            "(1, 2, 'waiting_for_actions', '{}', CURRENT_TIMESTAMP)"
        )
    )


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["user_initiated", "pending_timeout"])
async def test_cancel_cascade_full_effect(session_factory, path):
    async with session_factory() as setup:
        await _seed_active_tournament_with_rounds(setup)
        await setup.commit()

    bus = _CapturingBus()
    async with session_factory() as session:
        svc = TournamentService(session, bus)
        if path == "user_initiated":
            from atp.dashboard.models import User

            admin = await session.get(User, 99)
            await svc.cancel_tournament(user=admin, tournament_id=1)
        else:
            # pending_timeout — for this test we reuse the system entry
            # point even though tournament is ACTIVE (service guard allows
            # any status transition via _cancel_impl)
            await svc.cancel_tournament_system(
                tournament_id=1, reason=CancelReason.PENDING_TIMEOUT
            )
        await session.commit()

    async with session_factory() as verify:
        refreshed = await verify.get(Tournament, 1)
        assert refreshed.status == TournamentStatus.CANCELLED
        assert refreshed.cancelled_at is not None

        expected_reason = {
            "user_initiated": CancelReason.ADMIN_ACTION,
            "pending_timeout": CancelReason.PENDING_TIMEOUT,
        }[path]
        assert refreshed.cancelled_reason == expected_reason

        if path == "user_initiated":
            assert refreshed.cancelled_by == 99
        else:
            assert refreshed.cancelled_by is None

        completed = await verify.scalar(
            select(func.count())
            .select_from(Round)
            .where(Round.tournament_id == 1)
            .where(Round.status == RoundStatus.COMPLETED)
        )
        assert completed == 1  # pre-cancel COMPLETED round preserved

        in_flight = await verify.scalar(
            select(func.count())
            .select_from(Round)
            .where(Round.tournament_id == 1)
            .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
        )
        assert in_flight == 0  # transitioned to CANCELLED

        unreleased = await verify.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.tournament_id == 1)
            .where(Participant.released_at.is_(None))
        )
        assert unreleased == 0  # all released

    cancel_events = [e for e in bus.events if isinstance(e, TournamentCancelEvent)]
    assert len(cancel_events) == 1
    assert cancel_events[0].final_rounds_played == 1


@pytest.mark.anyio
async def test_idempotent_cancel_no_double_transition(session_factory):
    async with session_factory() as setup:
        await _seed_active_tournament_with_rounds(setup)
        await setup.commit()

    bus = _CapturingBus()
    async with session_factory() as session:
        from atp.dashboard.models import User

        admin = await session.get(User, 99)
        svc = TournamentService(session, bus)
        await svc.cancel_tournament(user=admin, tournament_id=1)
        await session.commit()

    async with session_factory() as session:
        from atp.dashboard.models import User

        admin = await session.get(User, 99)
        svc = TournamentService(session, bus)
        await svc.cancel_tournament(user=admin, tournament_id=1)
        await session.commit()

    # Exactly one event — second call is an idempotent no-op
    cancel_events = [e for e in bus.events if isinstance(e, TournamentCancelEvent)]
    assert len(cancel_events) == 1


@pytest.mark.anyio
async def test_cancel_publish_failure_returns_success(session_factory):
    async with session_factory() as setup:
        await _seed_active_tournament_with_rounds(setup)
        await setup.commit()

    class _FailingBus:
        async def publish(self, event: object) -> None:
            raise ConnectionError("bus down")

    async with session_factory() as session:
        from atp.dashboard.models import User

        admin = await session.get(User, 99)
        svc = TournamentService(session, _FailingBus())
        # MUST NOT raise
        await svc.cancel_tournament(user=admin, tournament_id=1)
        await session.commit()

    async with session_factory() as verify:
        refreshed = await verify.get(Tournament, 1)
        assert refreshed.status == TournamentStatus.CANCELLED
