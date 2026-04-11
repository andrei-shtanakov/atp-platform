"""Integration tests for the MCP join_tournament tool handler.

Verifies session_sync is emitted as the first notification on every
join call (first and subsequent), per the Plan 2a contract.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from atp.dashboard.mcp.tools import join_tournament
from atp.dashboard.tournament.service import TournamentService


@pytest.mark.anyio
async def test_join_tournament_emits_session_sync_first(session_factory):
    async with session_factory() as setup:
        await setup.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await setup.execute(
            text(
                "INSERT INTO tournaments "
                "(id, tenant_id, game_type, config, rules, status, num_players, "
                "total_rounds, round_deadline_s, pending_deadline, created_at) "
                "VALUES (1, 'default', 'prisoners_dilemma', '{}', '{}', 'pending', "
                "2, 3, 30, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await setup.commit()

    notifications = []

    ctx = MagicMock()
    ctx.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    async with session_factory() as session:
        from atp.dashboard.models import User

        user = await session.get(User, 1)
        svc = TournamentService(session, bus=_CapturingBus())
        result = await join_tournament(
            ctx=ctx,
            service=svc,
            user=user,
            tournament_id=1,
            agent_name="bot",
            join_token=None,
        )

    assert result["joined"] is True
    assert len(notifications) >= 1
    first_notification = notifications[0]
    assert "session_sync" in str(first_notification)


@pytest.mark.anyio
async def test_join_tournament_emits_session_sync_on_reconnect(session_factory):
    """session_sync must also fire on idempotent re-join (is_new=False)."""
    async with session_factory() as setup:
        await setup.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (2, 'default', 'bob', 'bob@test.com', 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await setup.execute(
            text(
                "INSERT INTO tournaments "
                "(id, tenant_id, game_type, config, rules, status, num_players, "
                "total_rounds, round_deadline_s, pending_deadline, created_at) "
                "VALUES (2, 'default', 'prisoners_dilemma', '{}', '{}', 'pending', "
                "2, 3, 30, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await setup.commit()

    notifications: list = []

    ctx = MagicMock()
    ctx.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    async with session_factory() as session:
        from atp.dashboard.models import User

        user = await session.get(User, 2)
        svc = TournamentService(session, bus=_CapturingBus())
        # First join
        await join_tournament(
            ctx=ctx,
            service=svc,
            user=user,
            tournament_id=2,
            agent_name="bot",
            join_token=None,
        )

    notifications.clear()

    ctx2 = MagicMock()
    ctx2.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    async with session_factory() as session2:
        from atp.dashboard.models import User

        user2 = await session2.get(User, 2)
        svc2 = TournamentService(session2, bus=_CapturingBus())
        result = await join_tournament(
            ctx=ctx2,
            service=svc2,
            user=user2,
            tournament_id=2,
            agent_name="bot",
            join_token=None,
        )

    assert result["joined"] is True
    assert result["is_new"] is False
    assert len(notifications) >= 1
    assert "session_sync" in str(notifications[0])


class _CapturingBus:
    async def publish(self, event):
        pass
