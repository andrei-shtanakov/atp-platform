"""Integration tests for session_sync delivery on join and reconnect."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from atp.dashboard.mcp.tools import join_tournament
from atp.dashboard.tournament.service import TournamentService


class _CapturingBus:
    async def publish(self, event):
        pass


async def _seed_tournament_in_progress(session):
    await session.execute(
        text(
            "INSERT INTO users "
            "(id, tenant_id, username, email, hashed_password, "
            "is_active, is_admin, created_at, updated_at) "
            "VALUES "
            "(1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
            " CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
            "(2, 'default', 'bob', 'bob@test.com', 'x', 1, 0, "
            " CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            " total_rounds, round_deadline_s, pending_deadline, created_at) "
            "VALUES (1, 'default', 'prisoners_dilemma', '{}', '{}', 'pending', "
            " 2, 3, 30, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
    )


@pytest.mark.anyio
async def test_session_sync_is_first_notification_on_fresh_join(
    session_factory,
):
    async with session_factory() as setup:
        await _seed_tournament_in_progress(setup)
        await setup.commit()

    notifications = []
    ctx = MagicMock()
    ctx.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    async with session_factory() as session:
        from atp.dashboard.models import User

        user = await session.get(User, 1)
        svc = TournamentService(session, _CapturingBus())
        await join_tournament(
            ctx=ctx,
            service=svc,
            user=user,
            tournament_id=1,
            agent_name="bot",
            join_token=None,
        )

    assert len(notifications) >= 1
    first = notifications[0]
    assert first.get("event") == "session_sync"


@pytest.mark.anyio
async def test_session_sync_on_idempotent_rejoin(session_factory):
    async with session_factory() as setup:
        await _seed_tournament_in_progress(setup)
        await setup.commit()

    ctx1 = MagicMock()
    ctx1.session.send_notification = AsyncMock()

    # First join
    async with session_factory() as session:
        from atp.dashboard.models import User

        user = await session.get(User, 1)
        svc = TournamentService(session, _CapturingBus())
        await join_tournament(
            ctx=ctx1,
            service=svc,
            user=user,
            tournament_id=1,
            agent_name="bot",
            join_token=None,
        )

    ctx2 = MagicMock()
    notifications = []
    ctx2.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    # Reconnect / second join — must also emit session_sync
    async with session_factory() as session:
        from atp.dashboard.models import User

        user = await session.get(User, 1)
        svc = TournamentService(session, _CapturingBus())
        await join_tournament(
            ctx=ctx2,
            service=svc,
            user=user,
            tournament_id=1,
            agent_name="bot",
            join_token=None,
        )

    assert len(notifications) >= 1
    assert notifications[0].get("event") == "session_sync"
