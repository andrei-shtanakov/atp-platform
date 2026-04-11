"""Tests for the per-player notification formatter."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.events import TournamentEvent


@pytest.mark.anyio
async def test_format_round_started_calls_get_state_for_and_wraps_payload() -> None:
    from atp.dashboard.mcp.notifications import _format_notification_for_user

    fake_user = MagicMock(id=42)
    fake_state = MagicMock()
    fake_state.to_dict.return_value = {"round_number": 5, "your_turn": True}
    fake_service = MagicMock()
    fake_service.get_state_for = AsyncMock(return_value=fake_state)

    event = TournamentEvent(
        event_type="round_started",
        tournament_id=7,
        round_number=5,
        data={"total_rounds": 10},
        timestamp=datetime.now(tz=UTC),
    )

    notification = await _format_notification_for_user(event, fake_user, fake_service)

    fake_service.get_state_for.assert_awaited_once_with(7, fake_user)
    assert notification is not None
    assert notification["method"] == "notifications/message"
    assert notification["params"]["data"]["event"] == "round_started"
    assert notification["params"]["data"]["tournament_id"] == 7
    assert notification["params"]["data"]["round_number"] == 5
    assert notification["params"]["data"]["state"] == {
        "round_number": 5,
        "your_turn": True,
    }


@pytest.mark.anyio
async def test_format_tournament_completed_does_not_call_get_state_for() -> None:
    from atp.dashboard.mcp.notifications import _format_notification_for_user

    fake_user = MagicMock(id=42)
    fake_service = MagicMock()
    fake_service.get_state_for = AsyncMock()

    event = TournamentEvent(
        event_type="tournament_completed",
        tournament_id=7,
        round_number=None,
        data={"final_scores": {42: 0.0, 43: 15.0}},
        timestamp=datetime.now(tz=UTC),
    )

    notification = await _format_notification_for_user(event, fake_user, fake_service)

    fake_service.get_state_for.assert_not_awaited()
    assert notification is not None
    assert notification["params"]["data"]["event"] == "tournament_completed"
    assert notification["params"]["data"]["final_scores"] == {42: 0.0, 43: 15.0}
