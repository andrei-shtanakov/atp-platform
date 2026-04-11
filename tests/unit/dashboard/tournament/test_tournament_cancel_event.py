"""Tests for TournamentCancelEvent dataclass + __post_init__ validator."""

from datetime import datetime

import pytest

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason


def _build(**overrides):
    base = dict(
        tournament_id=1,
        cancelled_at=datetime(2026, 4, 15, 10, 0, 0),
        cancelled_by=42,
        cancelled_reason=CancelReason.ADMIN_ACTION,
        cancelled_reason_detail=None,
        final_rounds_played=0,
        final_status=TournamentStatus.CANCELLED,
    )
    base.update(overrides)
    return base


def test_admin_action_with_actor_valid():
    event = TournamentCancelEvent(**_build())
    assert event.cancelled_reason == CancelReason.ADMIN_ACTION
    assert event.cancelled_by == 42


def test_pending_timeout_without_actor_valid():
    event = TournamentCancelEvent(
        **_build(
            cancelled_reason=CancelReason.PENDING_TIMEOUT,
            cancelled_by=None,
        )
    )
    assert event.cancelled_by is None


def test_abandoned_without_actor_valid():
    event = TournamentCancelEvent(
        **_build(
            cancelled_reason=CancelReason.ABANDONED,
            cancelled_by=None,
        )
    )
    assert event.cancelled_by is None


def test_admin_action_without_actor_rejected():
    with pytest.raises(ValueError, match="must have cancelled_by set"):
        TournamentCancelEvent(
            **_build(
                cancelled_reason=CancelReason.ADMIN_ACTION,
                cancelled_by=None,
            )
        )


def test_pending_timeout_with_actor_rejected():
    with pytest.raises(ValueError, match="must have cancelled_by=None"):
        TournamentCancelEvent(
            **_build(
                cancelled_reason=CancelReason.PENDING_TIMEOUT,
                cancelled_by=42,
            )
        )


def test_abandoned_with_actor_rejected():
    with pytest.raises(ValueError, match="must have cancelled_by=None"):
        TournamentCancelEvent(
            **_build(
                cancelled_reason=CancelReason.ABANDONED,
                cancelled_by=42,
            )
        )


def test_final_status_must_be_cancelled():
    with pytest.raises(ValueError, match="final_status must be CANCELLED"):
        TournamentCancelEvent(
            **_build(
                final_status=TournamentStatus.ACTIVE,
            )
        )


def test_is_frozen():
    event = TournamentCancelEvent(**_build())
    with pytest.raises(Exception):  # FrozenInstanceError on dataclass
        event.tournament_id = 999  # type: ignore[misc]
