"""Tests for BattleOfSexes game-layer service-facing methods.

Mirror of ``test_stag_hunt_state_format.py``. BoS is asymmetric — p0
prefers A, p1 prefers B — so format_state_for_player exposes
``your_preferred`` and payoff assertions check both (A, A) and (B, B)
coordination outcomes.
"""

import pytest
from game_envs.core.errors import ValidationError
from game_envs.games.battle_of_sexes import A, B, BattleOfSexes, BoSConfig


def _game() -> BattleOfSexes:
    return BattleOfSexes(BoSConfig())


def test_validate_action_accepts_a():
    assert _game().validate_action({"choice": A}) == {"choice": A}


def test_validate_action_accepts_b():
    assert _game().validate_action({"choice": B}) == {"choice": B}


def test_validate_action_rejects_unknown_choice():
    with pytest.raises(ValidationError, match="choice must be"):
        _game().validate_action({"choice": "stag"})


def test_validate_action_rejects_missing_choice():
    with pytest.raises(ValidationError, match="choice must be"):
        _game().validate_action({})


def test_validate_action_rejects_non_dict():
    with pytest.raises(ValidationError, match="action must be a dict"):
        _game().validate_action("A")  # type: ignore[arg-type]


def test_default_action_on_timeout_is_a():
    """Schelling focal point — A is the convention; documented in the method."""
    assert _game().default_action_on_timeout() == {"choice": A}


def test_format_state_empty_history_exposes_your_preferred():
    g = _game()
    p0 = g.format_state_for_player(
        round_number=1,
        total_rounds=3,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0],
    )
    assert p0["game_type"] == "battle_of_sexes"
    assert p0["your_preferred"] == A
    p1 = g.format_state_for_player(
        round_number=1,
        total_rounds=3,
        participant_idx=1,
        action_history=[],
        cumulative_scores=[0.0, 0.0],
    )
    assert p1["your_preferred"] == B


def test_format_state_populated_history_from_both_sides():
    g = _game()
    history = [
        {"round": 1, "actions": {0: {"choice": A}, 1: {"choice": A}}},
        {"round": 2, "actions": {0: {"choice": B}, 1: {"choice": B}}},
    ]
    p0 = g.format_state_for_player(
        round_number=3,
        total_rounds=5,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[5.0, 5.0],
    )
    assert p0["your_history"] == [A, B]
    assert p0["opponent_history"] == [A, B]


def test_compute_round_payoffs_coord_on_a():
    """(A, A) → p0 gets preferred_a (3.0), p1 gets other_a (2.0)."""
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": A}, 1: {"choice": A}})
    assert payoffs == [3.0, 2.0]


def test_compute_round_payoffs_coord_on_b():
    """(B, B) → p0 gets other_b (2.0), p1 gets preferred_b (3.0)."""
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": B}, 1: {"choice": B}})
    assert payoffs == [2.0, 3.0]


def test_compute_round_payoffs_mismatch_ab():
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": A}, 1: {"choice": B}})
    assert payoffs == [0.0, 0.0]


def test_compute_round_payoffs_mismatch_ba():
    g = _game()
    payoffs = g.compute_round_payoffs({0: {"choice": B}, 1: {"choice": A}})
    assert payoffs == [0.0, 0.0]
