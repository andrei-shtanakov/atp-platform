"""LABS-TSA PR-6 — multi-agent-per-user tournament join semantics.

Exercises the agent-keyed Participant uniqueness rolled out in PR-6:
one user registers several tournament agents and plays them all in the
same tournament, and the same agent cannot be active in two
tournaments at once.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select, text

from atp.dashboard.tournament.errors import ConflictError
from atp.dashboard.tournament.models import Participant
from atp.dashboard.tournament.service import TournamentService


class _DummyBus:
    async def publish(self, event):
        pass


async def _seed_alice_with_agents(session, agent_names: list[str]) -> None:
    """Seed alice (user_id=1) plus one tournament-purpose Agent per name."""
    await session.execute(
        text(
            "INSERT INTO users "
            "(id, tenant_id, username, email, hashed_password, "
            "is_active, is_admin, created_at, updated_at) "
            "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )
    for i, name in enumerate(agent_names, start=1):
        await session.execute(
            text(
                "INSERT INTO agents "
                "(id, tenant_id, name, agent_type, purpose, config, "
                "owner_id, created_at, updated_at) "
                "VALUES (:id, 'default', :n, 'mcp', 'tournament', '{}', "
                "1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": i, "n": name},
        )


async def _seed_tournament(
    session, tournament_id: int, *, num_players: int = 2
) -> None:
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            "total_rounds, round_deadline_s, pending_deadline, "
            "created_by, created_at) "
            "VALUES (:id, 'default', 'prisoners_dilemma', '{}', '{}', "
            "'pending', :n, 3, 30, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"
        ),
        {"id": tournament_id, "n": num_players},
    )


async def _load_user(session, user_id: int):
    from atp.dashboard.models import User

    return await session.get(User, user_id)


@pytest.mark.anyio
async def test_user_plays_three_agents_in_same_tournament(session_factory):
    """Alice registers 3 tournament agents and joins the same private
    tournament as 3 distinct Participants. All 3 rows share
    ``user_id=alice.id`` but have different ``agent_id`` values.

    num_players=3 means the tournament transitions PENDING→ACTIVE on
    the 3rd join, but that is a side-effect we don't assert on here —
    the invariants we care about are row count and agent_id uniqueness.
    """
    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["alpha", "beta", "gamma"])
        await _seed_tournament(setup, tournament_id=1, num_players=3)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        for name in ("alpha", "beta", "gamma"):
            participant, is_new = await svc.join(
                tournament_id=1, user=alice, agent_name=name
            )
            assert is_new is True, name
            assert participant.user_id == 1
            assert participant.agent_name == name
        await s.commit()

    async with session_factory() as verify:
        rows = (
            (
                await verify.execute(
                    select(Participant)
                    .where(Participant.tournament_id == 1)
                    .order_by(Participant.agent_name)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 3
        assert {p.user_id for p in rows} == {1}
        assert len({p.agent_id for p in rows}) == 3
        assert all(p.agent_id is not None for p in rows)


@pytest.mark.anyio
async def test_same_agent_cannot_join_two_active_tournaments(session_factory):
    """Alice has one tournament agent. It joins tournament 1; a second
    join to tournament 2 must raise a ConflictError — ``uq_participant
    _agent_active`` forbids the same agent_id from being active in two
    tournaments simultaneously.
    """
    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["solo"])
        await _seed_tournament(setup, tournament_id=1)
        await _seed_tournament(setup, tournament_id=2)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        p1, is_new = await svc.join(tournament_id=1, user=alice, agent_name="solo")
        assert is_new is True
        await s.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        with pytest.raises(ConflictError) as exc_info:
            await svc.join(tournament_id=2, user=alice, agent_name="solo")
        # The conflict message should name the agent for MCP-side clarity.
        assert "solo" in str(exc_info.value)

    # Exactly one active Participant row across both tournaments for
    # this agent.
    async with session_factory() as verify:
        active_count = await verify.scalar(
            select(func.count(Participant.id)).where(
                Participant.agent_id == p1.agent_id,
                Participant.released_at.is_(None),
            )
        )
        assert active_count == 1


@pytest.mark.anyio
async def test_agent_can_join_new_tournament_after_prior_cancelled(session_factory):
    """``_cancel_impl`` sets ``released_at`` on every Participant. After
    that the agent must be free to join a fresh tournament without
    tripping ``uq_participant_agent_active``.
    """
    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["solo"])
        await _seed_tournament(setup, tournament_id=1)
        await _seed_tournament(setup, tournament_id=2)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        await svc.join(tournament_id=1, user=alice, agent_name="solo")
        await s.commit()

    # Simulate cancel: set released_at on the first Participant row and
    # transition tournament 1 to CANCELLED. This mirrors what
    # ``_cancel_impl`` does in production.
    async with session_factory() as cancel_s:
        await cancel_s.execute(
            text(
                "UPDATE tournament_participants SET released_at = CURRENT_TIMESTAMP "
                "WHERE tournament_id = 1"
            )
        )
        await cancel_s.execute(
            text(
                "UPDATE tournaments SET status = 'cancelled', "
                "cancelled_reason = 'abandoned', "
                "cancelled_at = CURRENT_TIMESTAMP WHERE id = 1"
            )
        )
        await cancel_s.commit()

    # Agent must now be free to join tournament 2.
    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        participant, is_new = await svc.join(
            tournament_id=2, user=alice, agent_name="solo"
        )
        assert is_new is True
        assert participant.tournament_id == 2
        await s.commit()
