"""LABS-TSA PR-6 — multi-agent-per-user tournament join semantics.

Exercises the agent-keyed Participant uniqueness rolled out in PR-6:
one user registers several tournament agents and plays them all in the
same tournament, and the same agent cannot be active in two
tournaments at once.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select, text

from atp.dashboard.models import Agent
from atp.dashboard.tournament.errors import ConflictError
from atp.dashboard.tournament.models import Action, Participant
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
    join to tournament 2 must raise a ConflictError —
    ``uq_participant_agent_active`` forbids the same agent_id from being
    active in two tournaments simultaneously.
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


@pytest.mark.anyio
async def test_submit_action_with_agent_id_routes_per_agent(session_factory):
    """Same user, 2 agents, same tournament: each submit_action(agent_id=X)
    resolves to that agent's Participant — distinct Action rows with
    distinct participant_id values.
    """
    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["alpha", "beta"])
        await _seed_tournament(setup, tournament_id=1, num_players=2)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        p_alpha, _ = await svc.join(tournament_id=1, user=alice, agent_name="alpha")
        p_beta, _ = await svc.join(tournament_id=1, user=alice, agent_name="beta")
        await s.commit()

    assert p_alpha.agent_id is not None
    assert p_beta.agent_id is not None
    assert p_alpha.agent_id != p_beta.agent_id

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        await svc.submit_action(
            1, alice, action={"choice": "cooperate"}, agent_id=p_alpha.agent_id
        )
        await svc.submit_action(
            1, alice, action={"choice": "defect"}, agent_id=p_beta.agent_id
        )
        await s.commit()

    async with session_factory() as verify:
        actions = (
            (await verify.execute(select(Action).order_by(Action.participant_id)))
            .scalars()
            .all()
        )
        assert len(actions) == 2
        pids = {a.participant_id for a in actions}
        assert pids == {p_alpha.id, p_beta.id}


@pytest.mark.anyio
async def test_get_state_for_with_agent_id_returns_that_agents_state(
    session_factory,
):
    """get_state_for(agent_id=X) must key the state on agent X's
    Participant, not the first user-matching row. Without the agent_id
    arg the caller would always see the first-registered agent's state.
    """
    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["alpha", "beta"])
        await _seed_tournament(setup, tournament_id=1, num_players=2)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        p_alpha, _ = await svc.join(tournament_id=1, user=alice, agent_name="alpha")
        p_beta, _ = await svc.join(tournament_id=1, user=alice, agent_name="beta")
        # alpha submits; beta still pending.
        await svc.submit_action(
            1, alice, action={"choice": "cooperate"}, agent_id=p_alpha.agent_id
        )
        await s.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        state_alpha = await svc.get_state_for(
            tournament_id=1, user=alice, agent_id=p_alpha.agent_id
        )
        state_beta = await svc.get_state_for(
            tournament_id=1, user=alice, agent_id=p_beta.agent_id
        )
    # PD: "your_turn" reflects pending submission for THIS participant.
    # alpha already submitted → your_turn False; beta has not → True.
    assert state_alpha.your_turn is False
    assert state_beta.your_turn is True


@pytest.mark.anyio
async def test_rejoin_after_leave_raises_previously_participated(
    session_factory,
):
    """An agent that leaves a tournament cannot rejoin the same one.
    The error must surface the 'previously participated' wording, not
    the 'already has an active tournament' message — the released row
    still exists and only ``uq_participant_tournament_agent`` fires.
    """
    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["solo"])
        # num_players=3 so the join does NOT auto-start the tournament
        # when solo joins alone; leaving / rejoining must be testable
        # without races against round_started.
        await _seed_tournament(setup, tournament_id=1, num_players=3)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        await svc.join(tournament_id=1, user=alice, agent_name="solo")
        await s.commit()

    # Mark the participant released (simulate leave of a single-user
    # participant; tournament remains PENDING). Using a direct UPDATE
    # keeps this test independent of leave()'s cascade semantics.
    async with session_factory() as leave_s:
        await leave_s.execute(
            text(
                "UPDATE tournament_participants "
                "SET released_at = CURRENT_TIMESTAMP WHERE tournament_id = 1"
            )
        )
        await leave_s.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        with pytest.raises(ConflictError) as exc_info:
            await svc.join(tournament_id=1, user=alice, agent_name="solo")
    msg = str(exc_info.value).lower()
    assert "previously" in msg or "left tournament" in msg
    assert "already has an active tournament" not in msg


@pytest.mark.anyio
async def test_auto_provision_enforces_tournament_agent_quota(
    session_factory, monkeypatch
):
    """Auto-provision path (agent_id=None, agent_name not yet owned) must
    enforce ``ATP_MAX_TOURNAMENT_AGENTS_PER_USER``. At the cap, a fresh
    name must raise ConflictError and no Agent row must be created.
    """
    monkeypatch.setenv("ATP_MAX_TOURNAMENT_AGENTS_PER_USER", "1")
    monkeypatch.setenv("ATP_SECRET_KEY", "unit-test-secret-key")
    monkeypatch.setenv("ATP_DEBUG", "true")
    # Reset the cached config so the monkeypatched env var is picked up.
    import atp.dashboard.v2.config as cfg_mod

    cfg_mod.get_config.cache_clear()

    async with session_factory() as setup:
        await _seed_alice_with_agents(setup, ["first-agent"])
        await _seed_tournament(setup, tournament_id=1, num_players=2)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        with pytest.raises(ConflictError) as exc_info:
            await svc.join(tournament_id=1, user=alice, agent_name="second-agent")
    msg = str(exc_info.value).lower()
    assert "quota" in msg

    # No second-agent Agent row should have been created.
    async with session_factory() as verify:
        row = await verify.scalar(
            select(Agent).where(
                Agent.owner_id == 1,
                Agent.name == "second-agent",
                Agent.deleted_at.is_(None),
            )
        )
        assert row is None

    cfg_mod.get_config.cache_clear()


@pytest.mark.anyio
async def test_failed_join_does_not_auto_provision_agent(session_factory):
    """When the join itself fails (wrong join_token on a private tournament),
    auto-provision must NOT run — otherwise the orphan Agent row consumes
    a quota slot with nothing to show for it.
    """
    async with session_factory() as setup:
        # Alice has no existing agents.
        await setup.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        # Private tournament with a known join_token.
        await setup.execute(
            text(
                "INSERT INTO tournaments "
                "(id, tenant_id, game_type, config, rules, status, "
                "num_players, total_rounds, round_deadline_s, "
                "pending_deadline, created_by, join_token, created_at) "
                "VALUES (1, 'default', 'prisoners_dilemma', '{}', '{}', "
                "'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1, "
                "'correct-token', CURRENT_TIMESTAMP)"
            )
        )
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        with pytest.raises(ConflictError):
            await svc.join(
                tournament_id=1,
                user=alice,
                agent_name="phantom",
                join_token="WRONG",
            )

    async with session_factory() as verify:
        row = await verify.scalar(
            select(Agent).where(
                Agent.owner_id == 1,
                Agent.name == "phantom",
                Agent.deleted_at.is_(None),
            )
        )
        assert row is None
