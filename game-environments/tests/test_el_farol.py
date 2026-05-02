"""Tests for El Farol game-layer service-facing methods."""

from dataclasses import asdict, replace

import pytest

from game_envs.core.errors import ValidationError
from game_envs.games.el_farol import ElFarolBar, ElFarolConfig


def _game(n: int = 5, num_slots: int = 16) -> ElFarolBar:
    return ElFarolBar(
        ElFarolConfig(
            num_players=n,
            num_slots=num_slots,
            capacity_threshold=3,
            max_total_slots=min(num_slots, 8),
        )
    )


def test_validate_action_accepts_intervals():
    g = _game(num_slots=16)
    assert g.validate_action({"intervals": [[3, 7]]}) == {"intervals": [[3, 7]]}


def test_validate_action_orders_intervals():
    g = _game(num_slots=16)
    # Two non-adjacent intervals submitted out of order get sorted by start.
    assert g.validate_action({"intervals": [[10, 12], [0, 1]]}) == {
        "intervals": [[0, 1], [10, 12]]
    }


def test_validate_action_accepts_empty_list():
    g = _game()
    assert g.validate_action({"intervals": []}) == {"intervals": []}


def test_validate_action_rejects_overlapping_intervals():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="overlap or are adjacent"):
        g.validate_action({"intervals": [[0, 3], [2, 5]]})


def test_validate_action_rejects_adjacent_intervals():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="overlap or are adjacent"):
        g.validate_action({"intervals": [[0, 3], [4, 5]]})


def test_validate_action_rejects_out_of_range():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"intervals": [[0, 16]]})


def test_validate_action_rejects_negative():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"intervals": [[-1, 2]]})


def test_validate_action_rejects_too_many_total_slots():
    g = _game(num_slots=16)
    # max_total_slots=8 → covering 9 slots in one interval must fail.
    with pytest.raises(ValidationError, match="max is"):
        g.validate_action({"intervals": [[0, 8]]})


def test_validate_action_rejects_too_many_intervals():
    g = _game(num_slots=16)
    # Default max_intervals=2.
    with pytest.raises(ValidationError, match="at most"):
        g.validate_action({"intervals": [[0, 0], [2, 2], [4, 4]]})


def test_validate_action_rejects_non_list_intervals():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({"intervals": "not-a-list"})


def test_validate_action_rejects_missing_intervals_key():
    g = _game()
    with pytest.raises(ValidationError, match="intervals"):
        g.validate_action({})


def test_validate_action_rejects_legacy_slots_key():
    g = _game()
    with pytest.raises(ValidationError, match="intervals"):
        g.validate_action({"slots": [0, 1, 2]})


def test_validate_action_rejects_non_dict():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action([[0, 1]])  # type: ignore[arg-type]


def test_validate_action_rejects_bool_bounds():
    g = _game()
    with pytest.raises(ValidationError, match="not an int"):
        g.validate_action({"intervals": [[True, False]]})


def test_validate_action_rejects_bad_pair_shape():
    g = _game()
    with pytest.raises(ValidationError, match="\\[start, end\\] pair"):
        g.validate_action({"intervals": [[1, 2, 3]]})


def test_validate_action_rejects_start_after_end():
    g = _game()
    with pytest.raises(ValidationError, match="start <= end"):
        g.validate_action({"intervals": [[5, 2]]})


def test_default_action_on_timeout_returns_empty_intervals():
    g = _game()
    assert g.default_action_on_timeout() == {"intervals": []}


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
    schema = state["action_schema"]
    assert schema["type"] == "intervals"
    assert schema["shape"] == '{"intervals": [[start, end], ...]}'
    assert schema["max_intervals"] == 2
    assert schema["max_total_slots"] == 8
    assert schema["value_range"] == [0, 15]
    assert "pending_submission" not in state


def test_format_state_populated_history_aggregates_attendance():
    g = _game(n=3, num_slots=4)
    history = [
        {
            "round": 1,
            "actions": {
                0: {"intervals": [[0, 1]]},
                1: {"intervals": [[1, 2]]},
                2: {"intervals": [[0, 0]]},
            },
        },
        {
            "round": 2,
            "actions": {
                0: {"intervals": [[3, 3]]},
                1: {"intervals": [[3, 3]]},
                2: {"intervals": [[3, 3]]},
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
    # your_history is still a flat slot list per round, derived from intervals.
    assert state["your_history"] == [[0, 1], [3]]
    assert state["attendance_by_round"][0] == [2, 2, 1, 0]
    assert state["attendance_by_round"][1] == [0, 0, 0, 3]


def test_compute_round_payoffs_happy_minus_crowded():
    g = _game(n=3, num_slots=4)
    actions = {
        0: {"intervals": [[0, 1]]},
        1: {"intervals": [[1, 2]]},
        2: {"intervals": [[0, 1]]},
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
                0: {"intervals": [[0, 1]]},
                1: {"intervals": []},
                2: {"intervals": []},
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
    assert history[0]["actions"][0]["intervals"] == [[0, 1]]


def test_scoring_mode_validation_rejects_unknown():
    """Unknown scoring_mode value raises ValueError in __post_init__."""
    with pytest.raises(ValueError, match="scoring_mode"):
        ElFarolConfig(num_players=4, scoring_mode="bogus")  # type: ignore[arg-type]


def test_scoring_mode_temporary_default_is_legacy():
    """Until Task 6 flips it, the default remains happy_minus_crowded
    so all existing tests keep passing during incremental rollout.
    Task 6 changes this assertion to assert happy_only."""
    cfg = ElFarolConfig(num_players=4)
    assert cfg.scoring_mode == "happy_minus_crowded"


def test_scoring_mode_dataclass_round_trip():
    """ElFarolConfig with explicit scoring_mode survives asdict/replace
    round-trip and remains equal."""
    original = ElFarolConfig(num_players=4, scoring_mode="happy_only")
    payload = asdict(original)
    rebuilt = ElFarolConfig(**payload)
    assert rebuilt == original
    assert rebuilt.scoring_mode == "happy_only"

    via_replace = replace(original, num_players=8)
    assert via_replace.scoring_mode == "happy_only"


def test_compute_round_payoffs_happy_only_default():
    """In happy_only mode, per-round payoff equals the count of happy
    slots — crowded slots contribute 0, not −1.

    Scenario: 3 players, threshold=3, slot 1 is crowded (3 attendees).
        p0 attends slots 0,1 → slot 0 happy, slot 1 crowded → payoff = 1
        p1 attends slots 1,2 → slot 1 crowded, slot 2 happy → payoff = 1
        p2 attends slots 0,1 → slot 0 happy, slot 1 crowded → payoff = 1
    Under happy_minus_crowded these would all be 0."""
    cfg = ElFarolConfig(
        num_players=3,
        num_slots=4,
        capacity_threshold=3,
        max_total_slots=4,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    actions = {
        0: {"intervals": [[0, 1]]},
        1: {"intervals": [[1, 2]]},
        2: {"intervals": [[0, 1]]},
    }
    payoffs = g.compute_round_payoffs(actions)
    assert payoffs == [1.0, 1.0, 1.0]


def test_step_happy_only_accumulates_t_happy_and_t_crowded():
    """In happy_only mode, both _t_happy and _t_crowded must accumulate
    even though _t_crowded is not in the payoff formula. This keeps
    observation telemetry (your_t_crowded_slots) consistent across modes."""
    cfg = ElFarolConfig(
        num_players=3,
        num_slots=4,
        max_total_slots=4,
        capacity_threshold=3,
        num_rounds=1,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Same scenario as compute_round_payoffs test:
    # slot 1 crowded (3 attendees), slots 0,2 happy.
    # p0: slot 0 happy + slot 1 crowded → t_happy[p0]=1, t_crowded[p0]=1
    # NOTE: step() keys actions by string player_id (compute_round_payoffs
    # keys by int). Adjust template accordingly.
    actions = {
        "player_0": {"intervals": [[0, 1]]},
        "player_1": {"intervals": [[1, 2]]},
        "player_2": {"intervals": [[0, 1]]},
    }
    result = g.step(actions)

    # Per-round payoff = happy (not happy - crowded).
    assert result.payoffs["player_0"] == 1.0
    # _t_crowded accumulated despite not being in the formula.
    assert g._t_crowded["player_0"] == 1.0
    assert g._t_happy["player_0"] == 1.0
