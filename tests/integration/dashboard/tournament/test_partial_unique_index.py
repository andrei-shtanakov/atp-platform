"""Integration tests for uq_participant_user_active partial unique index."""

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


@pytest.mark.anyio
async def test_one_active_per_user_across_tournaments(session_factory):
    async with session_factory() as s:
        await _seed_two_tournaments_and_user(s)
        await s.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name, joined_at) "
                "VALUES (1, 1, 'a', CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()

    with pytest.raises(IntegrityError) as exc_info:
        async with session_factory() as s:
            await s.execute(
                text(
                    "INSERT INTO tournament_participants "
                    "(tournament_id, user_id, agent_name, joined_at) "
                    "VALUES (2, 1, 'b', CURRENT_TIMESTAMP)"
                )
            )
            await s.commit()

    # SQLite reports the column name in UNIQUE errors for partial indexes;
    # the important assertion is that the IntegrityError is raised at all,
    # confirming the uq_participant_user_active partial index is enforced.
    err = str(exc_info.value)
    assert "uq_participant_user_active" in err or "user_id" in err


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
        # Seed 10 released participants across them
        for i in range(1, 11):
            await s.execute(
                text(
                    "INSERT INTO tournament_participants "
                    "(tournament_id, user_id, agent_name, joined_at, released_at) "
                    "VALUES (:id, 1, :n, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": i, "n": f"bot{i}"},
            )
        await s.commit()

    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.user_id == 1)
        )
        assert count == 10
