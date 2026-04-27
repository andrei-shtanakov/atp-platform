"""Tests for ``ElFarolAction`` payload validation, including the
legacy ``slots`` -> ``intervals`` normalizer.

The ``_accept_legacy_slots`` model_validator runs BEFORE field validation
(and crucially before ``extra='forbid'``) so pre-1704d clients sending
``{"slots": [...]}`` continue to work after the wire-format flip to
intervals (commit 1704da7). The normalizer:

  * sorts + dedupes the input slot list,
  * run-length-encodes consecutive integers into ``[start, end]`` pairs,
  * drops the original ``slots`` key,
  * passes the resulting intervals through the standard validation pipeline.

When the input cannot be expressed as ``<= _MAX_INTERVALS_PER_DAY``
non-adjacent runs (e.g. ``[0, 2, 4]``), the standard intervals validator
surfaces a clean error rather than silently accepting bad data.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.dashboard.tournament.schemas import ActionTelemetry, ElFarolAction

# ---------------------------------------------------------------------------
# Happy-path normalization: slots -> intervals
# ---------------------------------------------------------------------------


def test_legacy_slots_two_consecutive_become_single_interval() -> None:
    """``[0, 1]`` is one run of two slots -> one ``[0, 1]`` pair."""
    # GIVEN a legacy slots payload with two consecutive slots
    # WHEN we instantiate ElFarolAction
    action = ElFarolAction.model_validate({"game_type": "el_farol", "slots": [0, 1]})

    # THEN the normalizer produces a single inclusive interval
    assert action.intervals == [[0, 1]]


def test_legacy_slots_single_slot_becomes_single_point_interval() -> None:
    """A single slot ``[0]`` is a degenerate run of length 1 -> ``[0, 0]``."""
    # GIVEN one slot
    # WHEN we validate
    action = ElFarolAction.model_validate({"game_type": "el_farol", "slots": [0]})

    # THEN we get a single-point interval (start == end)
    assert action.intervals == [[0, 0]]


def test_legacy_slots_empty_list_means_stay_home() -> None:
    """Empty slots ``[]`` is the canonical "stay home" action -> ``intervals=[]``."""
    # GIVEN an empty slot list
    # WHEN we validate
    action = ElFarolAction.model_validate({"game_type": "el_farol", "slots": []})

    # THEN intervals is empty (the canonical stay-home shape)
    assert action.intervals == []


def test_legacy_slots_out_of_order_are_sorted_then_run_length_encoded() -> None:
    """Unsorted input -> sorted ascending then RLE'd into multiple runs."""
    # GIVEN slots given out of order with two distinct runs
    # WHEN we validate
    action = ElFarolAction.model_validate(
        {"game_type": "el_farol", "slots": [4, 5, 0, 1]}
    )

    # THEN both runs are produced in ascending start order
    assert action.intervals == [[0, 1], [4, 5]]


def test_legacy_slots_duplicates_are_deduped_before_rle() -> None:
    """Duplicate slots collapse — we dedupe via a set before encoding."""
    # GIVEN slots with duplicates that would otherwise distort the RLE
    # WHEN we validate
    action = ElFarolAction.model_validate(
        {"game_type": "el_farol", "slots": [0, 1, 1, 2]}
    )

    # THEN dedupe produces {0, 1, 2}, encoded as a single run
    assert action.intervals == [[0, 2]]


# ---------------------------------------------------------------------------
# Errors propagate through the standard intervals validation pipeline
# ---------------------------------------------------------------------------


def test_legacy_slots_too_many_runs_fails_intervals_validation() -> None:
    """``[0, 2, 4]`` requires three non-adjacent intervals — exceeds
    ``_MAX_INTERVALS_PER_DAY=2``. The normalizer hands off to the
    standard validator which surfaces a clean pydantic error."""
    # GIVEN a slot pattern that produces 3 non-adjacent runs
    # WHEN we validate
    # THEN the standard intervals validator rejects it
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate({"game_type": "el_farol", "slots": [0, 2, 4]})


def test_legacy_slots_exceeds_max_slots_per_day_fails_validation() -> None:
    """9 consecutive slots would RLE to a single interval covering 9
    slots — over ``MAX_SLOTS_PER_DAY=8``. Standard validator catches it."""
    # GIVEN 9 consecutive slots (over the per-day budget)
    # WHEN we validate
    # THEN intervals validation rejects (total slots > MAX_SLOTS_PER_DAY)
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate(
            {
                "game_type": "el_farol",
                "slots": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
        )


# ---------------------------------------------------------------------------
# Pass-through cases: normalizer must not interfere
# ---------------------------------------------------------------------------


def test_both_intervals_and_slots_present_intervals_wins_then_extras_forbid() -> None:
    """When ``intervals`` is already present the normalizer is a no-op,
    leaving the spurious ``slots`` key behind. ``extra='forbid'`` then
    rejects the payload — clients can't smuggle an extra key past us."""
    # GIVEN both shapes in the same payload
    # WHEN we validate
    # THEN intervals is used, leftover ``slots`` triggers extras-forbid
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate(
            {
                "game_type": "el_farol",
                "intervals": [[0, 1]],
                "slots": [3],
            }
        )
    # Sanity: the failure is the extras-forbid, not an intervals-shape error
    msg = str(exc.value).lower()
    assert "extra" in msg or "forbid" in msg or "not permitted" in msg


def test_native_intervals_unchanged_when_no_slots_key() -> None:
    """If ``slots`` is absent the normalizer must be a strict no-op —
    the native intervals path keeps working unchanged."""
    # GIVEN a native intervals payload
    # WHEN we validate
    action = ElFarolAction.model_validate(
        {"game_type": "el_farol", "intervals": [[0, 1]]}
    )

    # THEN intervals are preserved verbatim
    assert action.intervals == [[0, 1]]


def test_legacy_slots_non_list_falls_through_to_standard_error() -> None:
    """Non-list ``slots`` (e.g. a string) means the normalizer can't
    safely RLE — defer to the standard pipeline so the user sees the
    canonical "intervals: Field required" error rather than a
    misleading slot-coercion message."""
    # GIVEN a malformed slots field (not a list)
    # WHEN we validate
    # THEN the normalizer falls through and intervals is reported missing
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate({"game_type": "el_farol", "slots": "not a list"})
    msg = str(exc.value).lower()
    assert "intervals" in msg


def test_legacy_slots_with_non_int_values_falls_through_to_standard_error() -> None:
    """If any slot can't be coerced to int, defer to the standard
    pipeline — the user gets the canonical pydantic error instead of
    a half-converted intervals list."""
    # GIVEN a slot list with a non-int element
    # WHEN we validate
    # THEN the normalizer falls through and intervals is reported missing
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate({"game_type": "el_farol", "slots": [0, "x"]})
    msg = str(exc.value).lower()
    assert "intervals" in msg


# ---------------------------------------------------------------------------
# Companion fields survive normalization
# ---------------------------------------------------------------------------


def test_legacy_slots_preserves_reasoning_and_telemetry_alongside() -> None:
    """The normalizer must only touch ``slots`` -> ``intervals``;
    ``reasoning`` and ``telemetry`` ride along untouched."""
    # GIVEN a legacy slots payload bundled with reasoning + telemetry
    # WHEN we validate
    action = ElFarolAction.model_validate(
        {
            "game_type": "el_farol",
            "slots": [0, 1],
            "reasoning": "hi",
            "telemetry": {
                "model_id": "gpt-4o-mini",
                "tokens_in": 100,
                "tokens_out": 20,
                "cost_usd": 0.0001,
                "decide_ms": 234,
            },
        }
    )

    # THEN intervals were produced AND companion fields are preserved
    assert action.intervals == [[0, 1]]
    assert action.reasoning == "hi"
    assert isinstance(action.telemetry, ActionTelemetry)
    assert action.telemetry.model_id == "gpt-4o-mini"
    assert action.telemetry.decide_ms == 234
