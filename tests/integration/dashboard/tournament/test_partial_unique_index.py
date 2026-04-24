"""Integration tests for agent-level partial unique indexes (LABS-TSA PR-6)."""

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from atp.dashboard.tournament.models import Participant


async def _seed_two_tournaments_and_user(session) -> None:
    await session.execute(
        text(
            "INSERT INTO users "
            "(id, tenant_id, username, email, hashed_password, "
            "is_active, is_admin, created_at, updated_at) "
            "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            "total_rounds, round_deadline_s, pending_deadline, created_by, created_at) "
            "VALUES "
            "(1, 'default', 'prisoners_dilemma', '{\"name\":\"t1\"}', '{}', "
            "'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP),"
            "(2, 'default', 'prisoners_dilemma', '{\"name\":\"t2\"}', '{}', "
            "'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"
        )
    )
    # LABS-TSA PR-4: seed agents to satisfy agent-xor-builtin CHECK on
    # Participant inserts below.
    await session.execute(
        text(
            "INSERT INTO agents "
            "(id, tenant_id, name, agent_type, purpose, config, owner_id, "
            "created_at, updated_at) "
            "VALUES "
            "(1, 'default', 'a', 'mcp', 'tournament', '{}', 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
            "(2, 'default', 'b', 'mcp', 'tournament', '{}', 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )


@pytest.mark.anyio
async def test_one_active_per_agent_across_tournaments(session_factory):
    """LABS-TSA PR-6: the same Agent can't be active in two tournaments at
    once — ``uq_participant_agent_active`` enforces this on agent_id.
    """
    async with session_factory() as s:
        await _seed_two_tournaments_and_user(s)
        # Agent 1 joins tournament 1
        await s.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_id, agent_name, joined_at) "
                "VALUES (1, 1, 1, 'a', CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()

    with pytest.raises(IntegrityError) as exc_info:
        async with session_factory() as s:
            # Agent 1 again trying to join tournament 2 while still active
            # in tournament 1 — must be blocked by
            # uq_participant_agent_active.
            await s.execute(
                text(
                    "INSERT INTO tournament_participants "
                    "(tournament_id, user_id, agent_id, agent_name, joined_at) "
                    "VALUES (2, 1, 1, 'a', CURRENT_TIMESTAMP)"
                )
            )
            await s.commit()

    # SQLite reports the column name in UNIQUE errors for partial indexes;
    # the important assertion is that the IntegrityError is raised at all.
    err = str(exc_info.value)
    assert "uq_participant_agent_active" in err or "agent_id" in err


@pytest.mark.anyio
async def test_multiple_agents_same_user_in_one_tournament(session_factory):
    """LABS-TSA PR-6: a single user may play multiple agents in the same
    tournament. Two Participant rows with the same ``user_id`` but
    different ``agent_id`` values must both succeed.
    """
    async with session_factory() as s:
        await _seed_two_tournaments_and_user(s)
        await s.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_id, agent_name, joined_at) "
                "VALUES "
                "(1, 1, 1, 'a', CURRENT_TIMESTAMP), "
                "(1, 1, 2, 'b', CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()

    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.tournament_id == 1)
            .where(Participant.user_id == 1)
        )
        assert count == 2


@pytest.mark.anyio
async def test_same_agent_twice_in_same_tournament_blocked(session_factory):
    """LABS-TSA PR-6: ``uq_participant_tournament_agent`` blocks the same
    agent from appearing twice in one tournament.
    """
    async with session_factory() as s:
        await _seed_two_tournaments_and_user(s)
        await s.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_id, agent_name, joined_at) "
                "VALUES (1, 1, 1, 'a', CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()

    with pytest.raises(IntegrityError) as exc_info:
        async with session_factory() as s:
            await s.execute(
                text(
                    "INSERT INTO tournament_participants "
                    "(tournament_id, user_id, agent_id, agent_name, joined_at) "
                    "VALUES (1, 1, 1, 'a2', CURRENT_TIMESTAMP)"
                )
            )
            await s.commit()

    err = str(exc_info.value)
    assert "uq_participant_tournament_agent" in err or "agent_id" in err


@pytest.mark.anyio
async def test_builtins_not_constrained_by_either_index(session_factory):
    """LABS-TSA PR-6: both new partial unique indexes exempt builtin
    participants (``agent_id IS NULL``). A tournament can seat several
    builtins of the same strategy and a builtin can "be" in many
    tournaments simultaneously without tripping either index.
    """
    async with session_factory() as s:
        await _seed_two_tournaments_and_user(s)
        # 6 builtins in the same tournament — all agent_id NULL. None of
        # the constraints should complain (agent-xor-builtin CHECK is
        # still satisfied because builtin_strategy is set).
        for i in range(6):
            await s.execute(
                text(
                    "INSERT INTO tournament_participants "
                    "(tournament_id, user_id, agent_id, agent_name, "
                    "builtin_strategy, joined_at) "
                    "VALUES (1, NULL, NULL, :n, "
                    "'prisoners_dilemma/tit_for_tat', CURRENT_TIMESTAMP)"
                ),
                {"n": f"builtin-{i}"},
            )
        await s.commit()

    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.tournament_id == 1)
        )
        assert count == 6


@pytest.mark.anyio
async def test_released_rows_do_not_block(session_factory):
    async with session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        # Seed 10 tournaments
        for i in range(1, 11):
            await s.execute(
                text(
                    "INSERT INTO tournaments "
                    "(id, tenant_id, game_type, config, rules, status, "
                    "num_players, total_rounds, round_deadline_s, "
                    "pending_deadline, created_by, created_at) "
                    "VALUES "
                    "(:id, 'default', 'prisoners_dilemma', :cfg, '{}', "
                    "'completed', 2, 3, 30, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"
                ),
                {"id": i, "cfg": f'{{"name":"t{i}"}}'},
            )
        # LABS-TSA PR-4: agent per Participant for agent-xor-builtin CHECK
        for i in range(1, 11):
            await s.execute(
                text(
                    "INSERT INTO agents "
                    "(id, tenant_id, name, agent_type, purpose, config, "
                    "owner_id, created_at, updated_at) "
                    "VALUES (:id, 'default', :n, 'mcp', 'tournament', '{}', "
                    "1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": i, "n": f"bot{i}"},
            )
        # Seed 10 released participants across them
        for i in range(1, 11):
            await s.execute(
                text(
                    "INSERT INTO tournament_participants "
                    "(tournament_id, user_id, agent_id, agent_name, "
                    "joined_at, released_at) "
                    "VALUES (:id, 1, :aid, :n, CURRENT_TIMESTAMP, "
                    "CURRENT_TIMESTAMP)"
                ),
                {"id": i, "aid": i, "n": f"bot{i}"},
            )
        await s.commit()

    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.user_id == 1)
        )
        assert count == 10
