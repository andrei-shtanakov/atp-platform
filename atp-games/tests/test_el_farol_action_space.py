"""Tests for ``ElFarolActionSpace`` interval-only contract.

Covers the canonical contract:

  - Constructor signature
    ``ElFarolActionSpace(num_slots=16, max_intervals=2, max_total_slots=8)``.
  - ``contains(action)`` accepts only interval shapes:
      1. List of inclusive ``[start, end]`` pairs, e.g. ``[[0, 2], [6, 8]]``.
      2. Dict with an ``"intervals"`` key, e.g. ``{"intervals": [[0, 2]]}``.
      3. The empty list ``[]`` / ``{"intervals": []}`` ("stay home").
  - ``sanitize(action)`` returns a sorted, deduped ``list[int]`` of slot
    indices for valid interval input and ``[]`` for anything else
    (including flat slot lists, which are no longer accepted).

Invariants:
  - Each ``[start, end]`` pair has ``start <= end`` and both in
    ``[0, num_slots - 1]``.
  - Pairs are non-overlapping AND non-adjacent (at least one empty slot
    between them).
  - Total covered slots ``<= max_total_slots``.
  - Number of intervals ``<= max_intervals``.
  - Empty action is valid (means "stay home").
"""

from __future__ import annotations

from game_envs.games.el_farol import ElFarolActionSpace

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructorConfigurable:
    def test_defaults(self) -> None:
        # GIVEN no arguments
        # WHEN constructing with defaults
        aspace = ElFarolActionSpace()
        # THEN default attributes match the Phase-5 defaults
        assert aspace.num_slots == 16
        assert aspace.max_intervals == 2
        assert aspace.max_total_slots == 8

    def test_custom_num_slots(self) -> None:
        # GIVEN a custom num_slots
        # WHEN constructing
        aspace = ElFarolActionSpace(num_slots=24)
        # THEN num_slots is stored on the instance
        assert aspace.num_slots == 24

    def test_custom_max_intervals_and_max_total_slots(self) -> None:
        # GIVEN custom interval limits
        # WHEN constructing
        aspace = ElFarolActionSpace(max_intervals=1, max_total_slots=4)
        # THEN the attributes are readable on the instance
        assert aspace.max_intervals == 1
        assert aspace.max_total_slots == 4


# ---------------------------------------------------------------------------
# contains(): list-of-pairs shape
# ---------------------------------------------------------------------------


class TestContainsListOfPairs:
    def test_empty_list_valid(self) -> None:
        # GIVEN an empty list (stay home)
        aspace = ElFarolActionSpace()
        # WHEN checking containment
        # THEN True
        assert aspace.contains([]) is True

    def test_single_pair_valid(self) -> None:
        # GIVEN one pair
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True
        assert aspace.contains([[0, 2]]) is True

    def test_two_pairs_valid(self) -> None:
        # GIVEN two non-overlapping, non-adjacent pairs
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True
        assert aspace.contains([[0, 2], [6, 8]]) is True

    def test_three_pairs_invalid(self) -> None:
        # GIVEN three pairs
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False (> max_intervals=2)
        assert aspace.contains([[0, 0], [2, 2], [4, 4]]) is False

    def test_pair_start_greater_than_end_invalid(self) -> None:
        # GIVEN a malformed pair where start > end
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False
        assert aspace.contains([[5, 2]]) is False

    def test_adjacent_pairs_invalid(self) -> None:
        # GIVEN adjacent pairs (no empty slot between them)
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False
        assert aspace.contains([[0, 2], [3, 5]]) is False

    def test_overlapping_pairs_invalid(self) -> None:
        # GIVEN overlapping pairs
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False
        assert aspace.contains([[0, 4], [3, 7]]) is False

    def test_pair_out_of_range_invalid(self) -> None:
        # GIVEN a pair where end == num_slots (out of range)
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False (16 >= num_slots=16)
        assert aspace.contains([[0, 16]]) is False

    def test_pair_total_exceeds_max_invalid(self) -> None:
        # GIVEN a pair covering 5 slots with max_total_slots=4
        aspace = ElFarolActionSpace(max_total_slots=4)
        # WHEN checking
        # THEN False
        assert aspace.contains([[0, 4]]) is False

    def test_single_slot_pair_valid(self) -> None:
        # GIVEN a length-1 interval (start == end)
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True
        assert aspace.contains([[3, 3]]) is True


# ---------------------------------------------------------------------------
# contains(): dict-with-intervals shape
# ---------------------------------------------------------------------------


class TestContainsDictIntervals:
    def test_dict_intervals_valid(self) -> None:
        # GIVEN a dict with two valid pairs
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True
        assert aspace.contains({"intervals": [[0, 2], [6, 8]]}) is True

    def test_dict_intervals_empty_valid(self) -> None:
        # GIVEN a dict with an empty intervals list (stay home)
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True
        assert aspace.contains({"intervals": []}) is True

    def test_dict_missing_intervals_key_invalid(self) -> None:
        # GIVEN a dict without the "intervals" key (legacy "slots" payload)
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False
        assert aspace.contains({"slots": [0, 1]}) is False

    def test_dict_with_extra_keys_still_valid(self) -> None:
        # GIVEN a dict with intervals and extra metadata
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True (extra keys are ignored)
        assert aspace.contains({"intervals": [[0, 2]], "note": "morning"}) is True


# ---------------------------------------------------------------------------
# contains(): legacy flat slot list is no longer accepted
# ---------------------------------------------------------------------------


class TestContainsRejectsFlatSlotList:
    def test_flat_slot_list_invalid(self) -> None:
        aspace = ElFarolActionSpace()
        # Even a structurally tidy flat list of slot indices is now rejected.
        assert aspace.contains([0, 1, 2]) is False

    def test_flat_slot_list_two_runs_invalid(self) -> None:
        aspace = ElFarolActionSpace()
        assert aspace.contains([0, 1, 2, 6, 7, 8]) is False


# ---------------------------------------------------------------------------
# sanitize(): normalise to a flat list[int]
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_sanitize_pairs_to_flat(self) -> None:
        # GIVEN interval pairs
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN expansion into flat slot list
        assert aspace.sanitize([[0, 2], [6, 8]]) == [0, 1, 2, 6, 7, 8]

    def test_sanitize_dict_to_flat(self) -> None:
        # GIVEN a dict with intervals
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN expansion into flat slot list
        assert aspace.sanitize({"intervals": [[0, 2]]}) == [0, 1, 2]

    def test_sanitize_empty_list(self) -> None:
        # GIVEN an empty list (stay home)
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN [] (no slots attended)
        assert aspace.sanitize([]) == []

    def test_sanitize_empty_dict_intervals(self) -> None:
        # GIVEN a dict with an empty intervals list
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN []
        assert aspace.sanitize({"intervals": []}) == []

    def test_sanitize_none_returns_empty(self) -> None:
        # GIVEN None input
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN []
        assert aspace.sanitize(None) == []

    def test_sanitize_bogus_returns_empty(self) -> None:
        # GIVEN bogus input types
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN [] for both string and int inputs
        assert aspace.sanitize("hello") == []
        assert aspace.sanitize(42) == []

    def test_sanitize_flat_list_returns_empty(self) -> None:
        # GIVEN a (legacy) flat list of slot indices
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN [] (interval-only contract; legacy flat list rejected)
        assert aspace.sanitize([0, 1, 2]) == []
        assert aspace.sanitize([0, 1, 2, 6, 7, 8]) == []

    def test_sanitize_interval_shape_rejects_third_pair_entirely(self) -> None:
        # GIVEN interval-shape input with too many pairs
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN [] (caller's default-action fallback kicks in)
        assert aspace.sanitize([[0, 0], [2, 2], [4, 4]]) == []

    def test_sanitize_pairs_exceeding_max_total_slots_returns_empty(self) -> None:
        # GIVEN one wide interval beyond max_total_slots
        aspace = ElFarolActionSpace(max_total_slots=4)
        # WHEN sanitizing
        # THEN [] (strict rejection)
        assert aspace.sanitize([[0, 4]]) == []

    def test_sanitize_pairs_out_of_range_returns_empty(self) -> None:
        # GIVEN a pair with an out-of-range end
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN [] (strict rejection)
        assert aspace.sanitize([[14, 99]]) == []


# ---------------------------------------------------------------------------
# Integration with ElFarolBar / ElFarolConfig
# ---------------------------------------------------------------------------


class TestIntegrationWithConfig:
    def test_action_space_from_elfarol_game_uses_config_limits(self) -> None:
        # GIVEN an ElFarolBar constructed from a config with custom limits
        from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

        cfg = ElFarolConfig(
            num_players=8,
            num_slots=16,
            max_intervals=2,
            max_total_slots=6,
        )
        game = ElFarolBar(cfg)
        # WHEN requesting the action space for a player
        aspace = game.action_space("player_0")
        # THEN the action space reflects the config limits
        assert aspace.max_intervals == 2
        assert aspace.max_total_slots == 6

    def test_step_rejects_too_many_intervals(self) -> None:
        # GIVEN a 2-player ElFarolBar with default max_intervals=2
        from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

        cfg = ElFarolConfig(
            num_players=2,
            num_slots=16,
            max_intervals=2,
            max_total_slots=8,
            num_rounds=1,
            capacity_threshold=2,  # solo visits are always happy
        )
        game = ElFarolBar(cfg)
        game.reset()
        # WHEN player_0 submits 3 disjoint intervals (exceeds max_intervals=2)
        # AND player_1 submits a valid 2-run action
        result = game.step(
            {
                "player_0": [[0, 0], [4, 4], [8, 8]],  # 3 intervals
                "player_1": [[0, 1], [4, 5]],  # 2 intervals, valid
            }
        )
        # THEN player_0's action sanitizes to [] (no happy slots, no payoff)
        assert result.payoffs["player_0"] == 0.0
        assert game._t_happy["player_0"] == 0.0
        assert game._t_crowded["player_0"] == 0.0
        # AND player_1 still earns from valid slots (sanity check)
        assert game._t_happy["player_1"] > 0.0
