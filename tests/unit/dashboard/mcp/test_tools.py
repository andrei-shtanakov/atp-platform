"""Tests for MCP tool handlers (mocked service layer)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.anyio
async def test_join_tournament_calls_service_and_returns_participant_info() -> None:
    """The tool's pure impl should call TournamentService.join with the
    right arguments and return a dict with tournament_id and
    participant_id.
    """
    from atp.dashboard.mcp import tools

    fake_user = MagicMock(id=42, is_admin=False)
    fake_participant = MagicMock(id=99)
    fake_service = MagicMock()
    fake_service.join = AsyncMock(return_value=fake_participant)

    result = await tools._join_tournament_impl(
        tournament_id=7,
        agent_name="alice-tft",
        user=fake_user,
        service=fake_service,
    )

    fake_service.join.assert_awaited_once_with(
        tournament_id=7, user=fake_user, agent_name="alice-tft"
    )
    assert result == {
        "tournament_id": 7,
        "participant_id": 99,
        "agent_name": "alice-tft",
        "status": "joined",
    }


@pytest.mark.anyio
async def test_get_current_state_impl_returns_state_dict() -> None:
    from atp.dashboard.mcp import tools

    fake_user = MagicMock(id=42)
    fake_state = MagicMock()
    fake_state.to_dict.return_value = {
        "tournament_id": 7,
        "round_number": 5,
        "your_turn": True,
    }
    fake_service = MagicMock()
    fake_service.get_state_for = AsyncMock(return_value=fake_state)

    result = await tools._get_current_state_impl(
        tournament_id=7, user=fake_user, service=fake_service
    )
    fake_service.get_state_for.assert_awaited_once_with(7, fake_user)
    assert result == {
        "tournament_id": 7,
        "round_number": 5,
        "your_turn": True,
    }


@pytest.mark.anyio
async def test_make_move_impl_passes_action_to_service() -> None:
    from atp.dashboard.mcp import tools

    fake_user = MagicMock(id=42)
    fake_service = MagicMock()
    fake_service.submit_action = AsyncMock(
        return_value={"status": "waiting", "round_number": 1}
    )

    result = await tools._make_move_impl(
        tournament_id=7,
        action={"choice": "cooperate"},
        user=fake_user,
        service=fake_service,
    )
    fake_service.submit_action.assert_awaited_once_with(
        7, fake_user, action={"choice": "cooperate"}
    )
    assert result == {"status": "waiting", "round_number": 1}
