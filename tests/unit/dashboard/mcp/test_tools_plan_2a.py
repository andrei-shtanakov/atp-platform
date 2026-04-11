"""Unit tests for the five new Plan 2a MCP tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.mcp.tools import (
    cancel_tournament,
    get_history,
    get_tournament,
    leave_tournament,
    list_tournaments,
)


def _make_service(**method_returns):
    svc = MagicMock()
    for name, value in method_returns.items():
        setattr(svc, name, AsyncMock(return_value=value))
    return svc


def _ctx():
    c = MagicMock()
    c.session.send_notification = AsyncMock()
    return c


def _user():
    u = MagicMock()
    u.id = 1
    u.is_admin = False
    return u


@pytest.mark.anyio
async def test_leave_tournament_calls_service():
    svc = _make_service(leave=None)
    result = await leave_tournament(
        ctx=_ctx(), service=svc, user=_user(), tournament_id=1
    )
    svc.leave.assert_awaited_once_with(tournament_id=1, user=_user_matcher())
    assert result["left"] is True


@pytest.mark.anyio
async def test_get_history_returns_rounds():
    mock_rounds = [MagicMock(round_number=1), MagicMock(round_number=2)]
    svc = _make_service(get_history=mock_rounds)
    result = await get_history(
        ctx=_ctx(),
        service=svc,
        user=_user(),
        tournament_id=1,
        last_n=None,
    )
    assert len(result["rounds"]) == 2


@pytest.mark.anyio
async def test_list_tournaments_returns_filtered():
    mock_tournaments = [MagicMock(id=1), MagicMock(id=2)]
    svc = _make_service(list_tournaments=mock_tournaments)
    result = await list_tournaments(ctx=_ctx(), service=svc, user=_user(), status=None)
    assert len(result["tournaments"]) == 2


@pytest.mark.anyio
async def test_get_tournament_returns_detail():
    mock_t = MagicMock(id=1, name="t")
    svc = _make_service(get_tournament=mock_t)
    result = await get_tournament(
        ctx=_ctx(), service=svc, user=_user(), tournament_id=1
    )
    assert result["tournament"]["id"] == 1


@pytest.mark.anyio
async def test_cancel_tournament_calls_service():
    svc = _make_service(cancel_tournament=None)
    result = await cancel_tournament(
        ctx=_ctx(),
        service=svc,
        user=_user(),
        tournament_id=1,
        reason_detail=None,
    )
    svc.cancel_tournament.assert_awaited_once()
    assert result["cancelled"] is True


def _user_matcher():
    """MagicMock equality helper for keyword arg comparison."""

    class _M:
        def __eq__(self, other):
            return hasattr(other, "id") and other.id == 1

    return _M()
