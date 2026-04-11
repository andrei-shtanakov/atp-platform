"""Tests for the _format_for_user notification dispatcher."""

from datetime import datetime
from unittest.mock import MagicMock

from atp.dashboard.mcp.notifications import _format_for_user
from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason


def _user(user_id: int, is_admin: bool = False):
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _cancel_event(cancelled_by=42, reason=CancelReason.ADMIN_ACTION):
    return TournamentCancelEvent(
        tournament_id=1,
        cancelled_at=datetime(2026, 4, 15, 10, 0, 0),
        cancelled_by=cancelled_by,
        cancelled_reason=reason,
        cancelled_reason_detail=None,
        final_rounds_played=0,
        final_status=TournamentStatus.CANCELLED,
    )


def test_cancel_event_admin_recipient_sees_cancelled_by():
    payload = _format_for_user(_cancel_event(), _user(99, is_admin=True))
    assert payload is not None
    assert payload["event"] == "tournament_cancelled"
    assert payload["cancelled_by"] == 42


def test_cancel_event_non_admin_recipient_hides_cancelled_by():
    payload = _format_for_user(_cancel_event(), _user(99, is_admin=False))
    assert payload is not None
    assert payload["cancelled_by"] is None


def test_system_cancel_has_none_cancelled_by_regardless_of_recipient():
    event = _cancel_event(cancelled_by=None, reason=CancelReason.PENDING_TIMEOUT)
    admin_payload = _format_for_user(event, _user(99, is_admin=True))
    user_payload = _format_for_user(event, _user(99, is_admin=False))
    assert admin_payload["cancelled_by"] is None
    assert user_payload["cancelled_by"] is None
