"""Tests for El Farol game-layer service-facing methods."""

import pytest

from game_envs.core.errors import ValidationError
from game_envs.games.el_farol import MAX_SLOTS_PER_DAY, ElFarolBar, ElFarolConfig


def _game(n: int = 5, num_slots: int = 16) -> ElFarolBar:
    return ElFarolBar(
        ElFarolConfig(
            num_players=n,
            num_slots=num_slots,
            capacity_threshold=3,
        )
    )


def test_validate_action_accepts_sorted_unique_slots():
    g = _game(num_slots=16)
    assert g.validate_action({"slots": [3, 7, 0]}) == {"slots": [0, 3, 7]}


def test_validate_action_accepts_empty_list():
    g = _game()
    assert g.validate_action({"slots": []}) == {"slots": []}


def test_validate_action_rejects_duplicates():
    g = _game()
    with pytest.raises(ValidationError, match="unique"):
        g.validate_action({"slots": [1, 2, 2]})


def test_validate_action_rejects_out_of_range():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"slots": [16]})


def test_validate_action_rejects_negative():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"slots": [-1]})


def test_validate_action_rejects_too_many():
    g = _game(num_slots=16)
    too_many = list(range(MAX_SLOTS_PER_DAY + 1))
    with pytest.raises(ValidationError, match="at most"):
        g.validate_action({"slots": too_many})


def test_validate_action_rejects_non_list():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({"slots": "not-a-list"})


def test_validate_action_rejects_missing_slots():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({})


def test_validate_action_rejects_non_dict():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action([0, 1])  # type: ignore[arg-type]


def test_validate_action_rejects_bool_slots():
    g = _game()
    with pytest.raises(ValidationError, match="not an int"):
        g.validate_action({"slots": [True, False]})
    with pytest.raises(ValidationError, match="not an int"):
        g.validate_action({"slots": [True, True]})


def test_sanitize_is_permissive_where_validate_is_strict():
    """Boundary check from spec §3.1."""
    g = _game(num_slots=16)
    cleaned = g.action_space("player_0").sanitize([1, 2, 2, 99, -1])
    assert set(cleaned).issubset(range(16))
    with pytest.raises(ValidationError):
        g.validate_action({"slots": [1, 2, 2, 99, -1]})


def test_default_action_on_timeout_returns_empty_slots():
    g = _game()
    assert g.default_action_on_timeout() == {"slots": []}


def test_format_state_empty_history():
    g = _game(n=5, num_slots=16)
    state = g.format_state_for_player(
        round_number=1,
        total_rounds=10,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert state["game_type"] == "el_farol"
    assert state["your_history"] == []
    assert state["attendance_by_round"] == []
    assert state["your_cumulative_score"] == 0.0
    assert state["all_scores"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert state["your_participant_idx"] == 0
    assert state["num_slots"] == 16
    assert state["capacity_threshold"] == 3  # from _game() fixture
    assert "action_schema" in state
    assert "pending_submission" not in state


def test_format_state_populated_history_aggregates_attendance():
    g = _game(n=3, num_slots=4)
    history = [
        {
            "round": 1,
            "actions": {
                0: {"slots": [0, 1]},
                1: {"slots": [1, 2]},
                2: {"slots": [0]},
            },
        },
        {
            "round": 2,
            "actions": {
                0: {"slots": [3]},
                1: {"slots": [3]},
                2: {"slots": [3]},
            },
        },
    ]
    state = g.format_state_for_player(
        round_number=3,
        total_rounds=5,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[1.0, 0.0, -1.0],
    )
    assert state["your_history"] == [[0, 1], [3]]
    assert state["attendance_by_round"][0] == [2, 2, 1, 0]
    assert state["attendance_by_round"][1] == [0, 0, 0, 3]


def test_compute_round_payoffs_happy_minus_crowded():
    g = _game(n=3, num_slots=4)
    actions = {
        0: {"slots": [0, 1]},
        1: {"slots": [1, 2]},
        2: {"slots": [0, 1]},
    }
    payoffs = g.compute_round_payoffs(actions)
    # capacity_threshold=3 → slot 1 is crowded (3 attendees), others happy
    # p0: slot 0 happy, slot 1 crowded → 1-1=0
    # p1: slot 1 crowded, slot 2 happy → 1-1=0
    # p2: slot 0 happy, slot 1 crowded → 1-1=0
    assert payoffs == [0.0, 0.0, 0.0]


def test_format_state_your_history_is_defensive_copy():
    g = _game(n=3, num_slots=4)
    history = [
        {
            "round": 1,
            "actions": {
                0: {"slots": [0, 1]},
                1: {"slots": []},
                2: {"slots": []},
            },
        }
    ]
    state = g.format_state_for_player(
        round_number=2,
        total_rounds=5,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[0.0, 0.0, 0.0],
    )
    state["your_history"][0].append(99)
    # mutation must NOT leak into action_history
    assert history[0]["actions"][0]["slots"] == [0, 1]
