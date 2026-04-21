"""Tests for the extended ``ElFarolActionSpace`` accepting interval shapes.

Covers the planned contract:

  - Constructor signature
    ``ElFarolActionSpace(num_slots=16, max_intervals=2, max_total_slots=8)``.
  - ``contains(action)`` accepts three canonical shapes:
      1. Flat slot list (legacy), e.g. ``[0, 1, 2, 6, 7, 8]``.
      2. List of inclusive ``[start, end]`` pairs, e.g. ``[[0, 2], [6, 8]]``.
      3. Dict with an ``"intervals"`` key, e.g. ``{"intervals": [[0, 2]]}``.
  - ``sanitize(action)`` returns a sorted, deduped ``list[int]`` of slot
    indices for valid input and ``[]`` for structurally invalid input.

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
# contains(): flat slot list shape
# ---------------------------------------------------------------------------


class TestContainsFlatSlotList:
    def test_empty_list_valid(self) -> None:
        # GIVEN an empty list (stay home)
        aspace = ElFarolActionSpace()
        # WHEN checking containment
        # THEN True
        assert aspace.contains([]) is True

    def test_single_contiguous_run_valid(self) -> None:
        # GIVEN one contiguous run
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True
        assert aspace.contains([0, 1, 2]) is True

    def test_two_runs_valid(self) -> None:
        # GIVEN two runs with gap at 3-5
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True (two runs fit max_intervals=2)
        assert aspace.contains([0, 1, 2, 6, 7, 8]) is True

    def test_three_runs_invalid(self) -> None:
        # GIVEN three distinct runs
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False (> max_intervals=2)
        assert aspace.contains([0, 1, 4, 5, 10, 11]) is False

    def test_unordered_slots_still_validate(self) -> None:
        # GIVEN unsorted input that sorts into two runs
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN True (sort internally before classifying runs)
        assert aspace.contains([8, 7, 0, 1, 2, 6]) is True

    def test_duplicate_slots_invalid(self) -> None:
        # GIVEN duplicates in flat form
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False
        assert aspace.contains([0, 0, 1, 2]) is False

    def test_exceeds_max_total_slots_invalid(self) -> None:
        # GIVEN a flat list longer than max_total_slots
        aspace = ElFarolActionSpace(max_total_slots=4)
        # WHEN checking
        # THEN False (5 slots > 4)
        assert aspace.contains([0, 1, 2, 6, 7]) is False

    def test_out_of_range_slot_invalid(self) -> None:
        # GIVEN out-of-range slot indices
        aspace = ElFarolActionSpace()
        # WHEN checking
        # THEN False in both directions
        assert aspace.contains([16]) is False
        assert aspace.contains([-1]) is False


# ---------------------------------------------------------------------------
# contains(): list-of-pairs shape
# ---------------------------------------------------------------------------


class TestContainsListOfPairs:
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
        # GIVEN a dict without the "intervals" key
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
# sanitize(): normalise to a flat list[int]
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_sanitize_flat_list_passthrough(self) -> None:
        # GIVEN a valid flat list already sorted and unique
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN passthrough
        assert aspace.sanitize([0, 1, 2, 6, 7, 8]) == [0, 1, 2, 6, 7, 8]

    def test_sanitize_flat_list_sorts_and_dedupes(self) -> None:
        # GIVEN an unsorted flat list with duplicates
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN result is sorted and deduplicated
        assert aspace.sanitize([8, 0, 1, 1, 2]) == [0, 1, 2, 8]

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

    def test_sanitize_drops_out_of_range_entries(self) -> None:
        # GIVEN a flat list with one out-of-range slot
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN out-of-range entry is silently dropped
        assert aspace.sanitize([0, 1, 99]) == [0, 1]

    def test_sanitize_truncates_over_max_total_slots(self) -> None:
        # GIVEN a flat list longer than max_total_slots
        aspace = ElFarolActionSpace(max_total_slots=4)
        # WHEN sanitizing
        # THEN at most max_total_slots returned, first 4 in sorted order
        result = aspace.sanitize([0, 1, 2, 3, 4, 5])
        assert len(result) <= 4
        assert result == [0, 1, 2, 3]

    def test_sanitize_interval_shape_rejects_third_pair_entirely(self) -> None:
        # GIVEN interval-shape input with too many pairs
        aspace = ElFarolActionSpace()
        # WHEN sanitizing
        # THEN [] (caller's default-action fallback kicks in)
        assert aspace.sanitize([[0, 0], [2, 2], [4, 4]]) == []


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
