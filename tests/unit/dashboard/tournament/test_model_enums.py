"""Tests for RoundStatus and ActionSource enums added by Plan 2a."""

from atp.dashboard.tournament.models import ActionSource, RoundStatus


def test_round_status_has_four_values():
    assert set(RoundStatus) == {
        RoundStatus.WAITING_FOR_ACTIONS,
        RoundStatus.IN_PROGRESS,
        RoundStatus.COMPLETED,
        RoundStatus.CANCELLED,
    }


def test_round_status_wire_values_match_vertical_slice():
    assert RoundStatus.WAITING_FOR_ACTIONS.value == "waiting_for_actions"
    assert RoundStatus.IN_PROGRESS.value == "in_progress"
    assert RoundStatus.COMPLETED.value == "completed"
    assert RoundStatus.CANCELLED.value == "cancelled"


def test_action_source_has_three_values():
    # LABS-TSA PR-4: added ActionSource.BUILTIN so runner-synthesised
    # builtin strategy moves are distinguishable from MCP-submitted
    # player moves in UI / admin reporting.
    assert set(ActionSource) == {
        ActionSource.SUBMITTED,
        ActionSource.TIMEOUT_DEFAULT,
        ActionSource.BUILTIN,
    }


def test_action_source_wire_values():
    assert ActionSource.SUBMITTED.value == "submitted"
    assert ActionSource.TIMEOUT_DEFAULT.value == "timeout_default"
    assert ActionSource.BUILTIN.value == "builtin"
