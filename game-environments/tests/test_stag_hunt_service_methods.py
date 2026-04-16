"""Tests for StagHunt game-layer service-facing methods.

Mirrors ``test_el_farol_state_format.py`` and the stricter half of
``test_state_formatter.py`` for PD. Covers validate_action,
default_action_on_timeout, format_state_for_player, and
compute_round_payoffs.
"""

import pytest

from game_envs.core.errors import ValidationError
from game_envs.games.stag_hunt import HARE, STAG, SHConfig, StagHunt


def _game() -> StagHunt:
    return StagHunt(SHConfig())


def test_validate_action_accepts_stag():
    assert _game().validate_action({"choice": STAG}) == {"choice": STAG}


def test_validate_action_accepts_hare():
    assert _game().validate_action({"choice": HARE}) == {"choice": HARE}


def test_validate_action_rejects_unknown_choice():
    with pytest.raises(ValidationError, match="choice must be"):
        _game().validate_action({"choice": "defect"})


def test_validate_action_rejects_missing_choice():
    with pytest.raises(ValidationError, match="choice must be"):
        _game().validate_action({})


def test_validate_action_rejects_non_dict():
    with pytest.raises(ValidationError, match="action must be a dict"):
        _game().validate_action(["stag"])  # type: ignore[arg-type]


def test_default_action_on_timeout_is_hare():
    """Risk-dominant equilibrium: no coordination means prefer safe payoff."""
    assert _game().default_action_on_timeout() == {"choice": HARE}


def test_format_state_empty_history():
    g = _game()
    state = g.format_state_for_player(
        round_number=1,
        total_rounds=3,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0],
    )
    assert state["game_type"] == "stag_hunt"
    assert state["your_history"] == []
    assert state["opponent_history"] == []
    assert state["your_cumulative_score"] == 0.0
    assert state["opponent_cumulative_score"] == 0.0
    assert state["action_schema"]["options"] == [STAG, HARE]
    assert state["total_rounds"] == 3


def test_format_state_populated_history_from_both_sides():
    g = _game()
    history = [
        {"round": 1, "actions": {0: {"choice": STAG}, 1: {"choice": STAG}}},
        {"round": 2, "actions": {0: {"choice": STAG}, 1: {"choice": HARE}}},
    ]

    p0 = g.format_state_for_player(
        round_number=3,
        total_rounds=5,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[4.0, 7.0],
    )
    assert p0["your_history"] == [STAG, STAG]
    assert p0["opponent_history"] == [STAG, HARE]
    assert p0["your_cumulative_score"] == 4.0
    assert p0["opponent_cumulative_score"] == 7.0

    p1 = g.format_state_for_player(
        round_number=3,
        total_rounds=5,
        participant_idx=1,
        action_history=history,
        cumulative_scores=[4.0, 7.0],
    )
    assert p1["your_history"] == [STAG, HARE]
    assert p1["opponent_history"] == [STAG, STAG]
    assert p1["your_cumulative_score"] == 7.0
    assert p1["opponent_cumulative_score"] == 4.0


def test_compute_round_payoffs_mutual_stag():
    """Both stag → mutual_stag payoff (default 4.0 each)."""
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": STAG}, 1: {"choice": STAG}})
    assert payoffs == [4.0, 4.0]


def test_compute_round_payoffs_mutual_hare():
    """Both hare → mutual_hare (default 3.0 each)."""
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": HARE}, 1: {"choice": HARE}})
    assert payoffs == [3.0, 3.0]


def test_compute_round_payoffs_sucker_and_hare():
    """Stag vs hare: stag-player gets sucker (0.0), hare gets hare (3.0)."""
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": STAG}, 1: {"choice": HARE}})
    assert payoffs == [0.0, 3.0]
