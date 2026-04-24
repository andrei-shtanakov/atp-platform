"""Race guard test: force_resolve_round vs submit_action on the same round
cannot both commit a resolution. AD-6 status filter + WAL serialization."""

import asyncio

import pytest
from sqlalchemy import text

from atp.dashboard.tournament.errors import ConflictError
from atp.dashboard.tournament.models import Round, RoundStatus
from atp.dashboard.tournament.service import TournamentService


class _CapturingBus:
    def __init__(self) -> None:
        self.events: list = []

    async def publish(self, event: object) -> None:
        self.events.append(event)


async def _setup_round_with_one_action_pending(session) -> None:
    for uid, name, is_admin in [(1, "alice", 0), (2, "bob", 0)]:
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
            "'active', 2, 3, 30, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"
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
            "(tournament_id, round_number, status, state, deadline, started_at) "
            "VALUES (1, 1, 'waiting_for_actions', '{}', '1970-01-01 00:00:00', "
            "CURRENT_TIMESTAMP)"
        )
    )
    # Only participant 1 (alice) has submitted; participant 2 (bob) is pending
    await session.execute(
        text(
            "INSERT INTO tournament_actions "
            "(round_id, participant_id, action_data, source, submitted_at) "
            "SELECT r.id, p.id, '{\"choice\": \"cooperate\"}', 'submitted', "
            "CURRENT_TIMESTAMP "
            "FROM tournament_rounds r "
            "JOIN tournament_participants p ON p.tournament_id = r.tournament_id "
            "WHERE r.round_number = 1 AND p.user_id = 1"
        )
    )


@pytest.mark.anyio
async def test_force_resolve_vs_submit_action(session_factory):
    """AD-6 guard: deadline path and submit path run concurrently.

    Exactly one must win (commit a resolution); the other must no-op or raise
    ConflictError. The round must end in COMPLETED status.
    """
    async with session_factory() as setup:
        await _setup_round_with_one_action_pending(setup)
        await setup.commit()

    # Resolve the round id after setup
    async with session_factory() as s:
        round_row = await s.get(Round, 1)
        assert round_row is not None
        assert round_row.status == RoundStatus.WAITING_FOR_ACTIONS

    bus = _CapturingBus()

    async def deadline_path() -> str:
        async with session_factory() as session:
            svc = TournamentService(session, bus)
            try:
                await svc.force_resolve_round(round_id=1)
                await session.commit()
                return "deadline_won"
            except ConflictError:
                return "deadline_noop"

    async def submit_path() -> str:
        from sqlalchemy.exc import IntegrityError

        async with session_factory() as session:
            from atp.dashboard.models import User

            user2 = await session.get(User, 2)
            svc = TournamentService(session, bus)
            try:
                await svc.submit_action(
                    tournament_id=1,
                    user=user2,
                    action={"choice": "cooperate"},
                )
                await session.commit()
                return "submit_won"
            except (ConflictError, IntegrityError):
                # IntegrityError fires when deadline_path inserted first and the
                # unique constraint on (round_id, participant_id) blocks our insert.
                return "submit_noop"

    results = await asyncio.gather(deadline_path(), submit_path())
    wins = [r for r in results if r.endswith("_won")]
    # Exactly one must win; the other no-ops
    assert len(wins) == 1, f"Expected 1 winner, got: {results}"

    async with session_factory() as verify:
        round_row = await verify.get(Round, 1)
        assert round_row is not None
        assert round_row.status == RoundStatus.COMPLETED
