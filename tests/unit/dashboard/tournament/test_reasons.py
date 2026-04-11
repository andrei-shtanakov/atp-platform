"""Tests for CancelReason enum module."""

from atp.dashboard.tournament.reasons import CancelReason


def test_cancel_reason_has_three_values():
    assert set(CancelReason) == {
        CancelReason.ADMIN_ACTION,
        CancelReason.PENDING_TIMEOUT,
        CancelReason.ABANDONED,
    }


def test_cancel_reason_values_are_stable_strings():
    assert CancelReason.ADMIN_ACTION.value == "admin_action"
    assert CancelReason.PENDING_TIMEOUT.value == "pending_timeout"
    assert CancelReason.ABANDONED.value == "abandoned"


def test_cancel_reason_is_str_enum():
    # StrEnum members compare equal to their string value in Python 3.11+
    assert CancelReason.ADMIN_ACTION == "admin_action"
    assert "abandoned" == CancelReason.ABANDONED
