"""Tests for PublicGoodsGame tournament-facing service methods.

Mirrors ``test_battle_of_sexes_service_methods.py`` / El Farol style —
covers the strict ``validate_action``, the timeout default, the
``format_state_for_player`` snapshot shape, and the symmetry of
``compute_round_payoffs`` against the canonical public-goods formula.
"""

import pytest

from game_envs.core.errors import ValidationError
from game_envs.games.public_goods import PGConfig, PublicGoodsGame


def _game(num_players: int = 4, endowment: float = 20.0) -> PublicGoodsGame:
    return PublicGoodsGame(PGConfig(num_players=num_players, endowment=endowment))


# ---- validate_action ----


def test_validate_action_accepts_float():
    assert _game().validate_action({"contribution": 10.0}) == {"contribution": 10.0}


def test_validate_action_coerces_int_to_float():
    assert _game().validate_action({"contribution": 5}) == {"contribution": 5.0}


def test_validate_action_accepts_zero():
    assert _game().validate_action({"contribution": 0}) == {"contribution": 0.0}


def test_validate_action_accepts_full_endowment():
    assert _game(endowment=20.0).validate_action({"contribution": 20.0}) == {
        "contribution": 20.0
    }


def test_validate_action_rejects_negative():
    with pytest.raises(ValidationError, match=r"\[0, 20\.0\]"):
        _game().validate_action({"contribution": -1.0})


def test_validate_action_rejects_above_endowment():
    with pytest.raises(ValidationError, match=r"\[0, 20\.0\]"):
        _game().validate_action({"contribution": 25.0})


def test_validate_action_rejects_non_number():
    with pytest.raises(ValidationError, match="must be a number"):
        _game().validate_action({"contribution": "ten"})


def test_validate_action_rejects_bool():
    # bool is a subclass of int in Python — guard explicitly so True/False
    # can't be smuggled in as 1/0 through json clients that lose type info.
    with pytest.raises(ValidationError, match="must be a number"):
        _game().validate_action({"contribution": True})


def test_validate_action_rejects_missing_field():
    with pytest.raises(ValidationError, match="must have field 'contribution'"):
        _game().validate_action({"other": 10})


def test_validate_action_rejects_non_dict():
    with pytest.raises(ValidationError, match="must be a dict"):
        _game().validate_action(10.0)


# ---- default_action_on_timeout ----


def test_default_action_on_timeout_is_zero_contribution():
    assert _game().default_action_on_timeout() == {"contribution": 0.0}


# ---- compute_round_payoffs ----


def test_compute_round_payoffs_all_contribute_full():
    """Everyone contributes full endowment → everyone gets
    (0 + 1.6 * total / n) = multiplier * endowment.
    For n=4, endowment=20, multiplier=1.6: each gets 1.6*80/4 = 32.
    """
    g = _game(num_players=4)
    actions = {i: {"contribution": 20.0} for i in range(4)}
    payoffs = g.compute_round_payoffs(actions)
    assert payoffs == [32.0, 32.0, 32.0, 32.0]


def test_compute_round_payoffs_all_free_ride():
    """Zero contributions → each keeps the full endowment."""
    g = _game(num_players=4)
    actions = {i: {"contribution": 0.0} for i in range(4)}
    payoffs = g.compute_round_payoffs(actions)
    assert payoffs == [20.0, 20.0, 20.0, 20.0]


def test_compute_round_payoffs_asymmetric():
    """One full contributor, three free-riders: free-riders get more."""
    g = _game(num_players=4)
    actions = {
        0: {"contribution": 20.0},
        1: {"contribution": 0.0},
        2: {"contribution": 0.0},
        3: {"contribution": 0.0},
    }
    # share = 1.6 * 20 / 4 = 8
    # p0: 20 - 20 + 8 = 8
    # p1,p2,p3: 20 - 0 + 8 = 28
    payoffs = g.compute_round_payoffs(actions)
    assert payoffs == pytest.approx([8.0, 28.0, 28.0, 28.0])
    # Classic free-rider result: contributors earn less than defectors
    assert payoffs[0] < payoffs[1]


def test_compute_round_payoffs_missing_action_treated_as_zero():
    """Timeout-default path: missing entry clamps to 0 contribution."""
    g = _game(num_players=3)
    # participant 2 absent — simulates the server's default-on-timeout flow
    actions = {
        0: {"contribution": 15.0},
        1: {"contribution": 0.0},
    }
    payoffs = g.compute_round_payoffs(actions)
    # share = 1.6 * 15 / 3 = 8
    # p0: 20 - 15 + 8 = 13; p1: 20 + 8 = 28; p2: 20 + 8 = 28
    assert payoffs == pytest.approx([13.0, 28.0, 28.0])


# ---- format_state_for_player ----


def test_format_state_empty_history():
    g = _game(num_players=3)
    state = g.format_state_for_player(
        round_number=1,
        total_rounds=5,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0, 0.0],
    )
    assert state["game_type"] == "public_goods"
    assert state["round_number"] == 1
    assert state["total_rounds"] == 5
    assert state["your_history"] == []
    assert state["all_contributions_by_round"] == []
    assert state["num_players"] == 3
    assert state["endowment"] == 20.0
    assert state["multiplier"] == 1.6
    assert state["your_participant_idx"] == 0
    assert state["your_cumulative_score"] == 0.0
    assert state["all_scores"] == [0.0, 0.0, 0.0]
    # Does NOT include pending_submission — service injects it
    assert "pending_submission" not in state


def test_format_state_exposes_full_contribution_vector():
    """PG is fully observable: all contributions become public after each round."""
    g = _game(num_players=3)
    history = [
        {
            "actions": {
                0: {"contribution": 10.0},
                1: {"contribution": 5.0},
                2: {"contribution": 0.0},
            },
            "payoffs": {},
        }
    ]
    state = g.format_state_for_player(
        round_number=2,
        total_rounds=5,
        participant_idx=1,
        action_history=history,
        cumulative_scores=[18.0, 13.0, 28.0],
    )
    assert state["your_history"] == [5.0]
    assert state["all_contributions_by_round"] == [[10.0, 5.0, 0.0]]
    assert state["your_participant_idx"] == 1
    assert state["your_cumulative_score"] == 13.0
