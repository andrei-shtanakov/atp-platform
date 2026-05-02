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


def test_compute_round_payoffs_legacy_mode_happy_minus_crowded():
    g = ElFarolBar(
        ElFarolConfig(
            num_players=3,
            num_slots=4,
            capacity_threshold=3,
            max_total_slots=4,
            scoring_mode="happy_minus_crowded",
        )
    )
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


def test_scoring_mode_default_is_happy_only():
    """The default scoring_mode is happy_only — the new platform-wide
    rule. Tournament total_score will be sum of per-round happy values."""
    cfg = ElFarolConfig(num_players=4)
    assert cfg.scoring_mode == "happy_only"


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


def test_get_payoffs_happy_only_returns_t_happy_sum():
    """In happy_only mode, get_payoffs() returns t_happy directly
    (count, not ratio). Crowded slots have no influence on the final."""
    cfg = ElFarolConfig(
        num_players=2,
        num_slots=4,
        max_total_slots=4,
        capacity_threshold=3,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Seed counters directly — bypass step() so the test isolates
    # get_payoffs() formula from the per-round path.
    g._t_happy["player_0"] = 5.0
    g._t_crowded["player_0"] = 3.0
    g._t_happy["player_1"] = 2.0
    g._t_crowded["player_1"] = 4.0

    final = g.get_payoffs()
    # happy_only: final = t_happy (regardless of t_crowded)
    assert final["player_0"] == 5.0
    assert final["player_1"] == 2.0


def test_get_payoffs_happy_only_disqualification_still_applies():
    """min_total_hours disqualification applies before the formula in
    both modes."""
    cfg = ElFarolConfig(
        num_players=2,  # engine requires >= 2
        num_slots=4,
        max_total_slots=4,
        capacity_threshold=2,
        slot_duration=0.5,
        min_total_hours=10.0,  # require 10h, will not be met
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Attended only 1 slot (0.5 h) — well under 10 h threshold.
    g._t_happy["player_0"] = 1.0
    g._t_crowded["player_0"] = 0.0

    final = g.get_payoffs()
    assert final["player_0"] == 0.0  # disqualified


def test_happy_only_per_round_payoff_is_non_negative():
    """Property test: under happy_only, every per-round payoff ≥ 0
    regardless of action choices. Under legacy this could be negative."""
    import random

    cfg = ElFarolConfig(
        num_players=4,
        num_slots=8,
        capacity_threshold=2,
        max_intervals=2,
        max_total_slots=4,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    g.reset()
    rng = random.Random(42)

    for _ in range(100):
        actions: dict[int, dict[str, list[list[int]]]] = {}
        for p_idx in range(cfg.num_players):
            start = rng.randint(0, cfg.num_slots - 2)
            end = rng.randint(start, min(start + 2, cfg.num_slots - 1))
            actions[p_idx] = {"intervals": [[start, end]]}
        payoffs = g.compute_round_payoffs(actions)
        for p_idx, p in enumerate(payoffs):
            assert p >= 0.0, (
                f"happy_only payoff must be ≥ 0, got {p} for player {p_idx}"
            )


def test_observation_includes_t_crowded_in_both_modes():
    """observe(player_id) returns your_t_crowded_slots regardless of
    scoring_mode — telemetry is mode-independent."""
    for mode in ("happy_only", "happy_minus_crowded"):
        cfg = ElFarolConfig(
            num_players=2,
            num_slots=4,
            max_total_slots=4,
            capacity_threshold=3,
            scoring_mode=mode,  # type: ignore[arg-type]
        )
        g = ElFarolBar(cfg)
        g.reset()
        obs = g.observe("player_0")
        assert "your_t_crowded_slots" in obs.game_state, (
            f"mode={mode} missing your_t_crowded_slots"
        )


def test_to_prompt_happy_only_describes_no_penalty():
    """to_prompt() for happy_only mode must describe the new scoring
    rule (no penalty for crowded), not the legacy ratio formula."""
    cfg = ElFarolConfig(
        num_players=4,
        num_slots=8,
        capacity_threshold=2,
        scoring_mode="happy_only",
    )
    g = ElFarolBar(cfg)
    prompt = g.to_prompt()

    assert "happy" in prompt.lower()
    # New default: no ratio formula, no negative penalty mention.
    assert "t_happy / max" not in prompt
    assert "/ max(total_crowded_slots" not in prompt
    # Should mention that crowded slots have no penalty. Dropped the
    # ``or "0" in prompt`` fallback because the prompt unconditionally
    # contains "0" (slot range, deadline string, etc.) — it would mask
    # a regression where the "no penalty" phrasing disappears.
    assert "no penalty" in prompt.lower()


def test_to_prompt_legacy_describes_ratio_formula():
    """to_prompt() for happy_minus_crowded mode must describe the
    legacy ratio formula."""
    cfg = ElFarolConfig(
        num_players=4,
        num_slots=8,
        capacity_threshold=2,
        scoring_mode="happy_minus_crowded",
    )
    g = ElFarolBar(cfg)
    prompt = g.to_prompt()

    assert "happy" in prompt.lower()
    # Legacy: explicit ratio with the 0.1 floor.
    assert "max(total_crowded_slots, 0.1)" in prompt or "max(t_crowded, 0.1)" in prompt


def test_get_payoffs_legacy_mode_returns_ratio():
    """In happy_minus_crowded mode, get_payoffs() returns
    t_happy / max(t_crowded, 0.1). Direct seeding isolates the formula
    from the per-round path."""
    cfg = ElFarolConfig(
        num_players=2,
        num_slots=4,
        max_total_slots=4,
        capacity_threshold=3,
        scoring_mode="happy_minus_crowded",
    )
    g = ElFarolBar(cfg)
    g.reset()

    # Seed counters directly. Player 0: t_happy=8, t_crowded=4 → 2.0
    # Player 1: t_happy=3, t_crowded=0 → 30.0 (floor: max(0, 0.1) = 0.1)
    g._t_happy["player_0"] = 8.0
    g._t_crowded["player_0"] = 4.0
    g._t_happy["player_1"] = 3.0
    g._t_crowded["player_1"] = 0.0

    final = g.get_payoffs()
    assert final["player_0"] == pytest.approx(2.0)
    assert final["player_1"] == pytest.approx(30.0)
