"""Tests for ``ElFarolAction`` payload validation.

The legacy ``{"slots": [...]}`` wire shape is no longer accepted —
``extra="forbid"`` rejects payloads carrying an unknown ``slots`` key.

Coverage focuses on the ``_validate_intervals`` field validator:

  * total covered slots ``<= MAX_SLOTS_PER_DAY``
  * each pair is well-formed ``[start, end]`` with ``0 <= start <= end``
  * pairs are non-overlapping AND non-adjacent
  * the normal happy paths (single pair, two-pair, empty list).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.dashboard.tournament.schemas import ActionTelemetry, ElFarolAction


# ---------------------------------------------------------------------------
# Legacy "slots" payload is rejected as an unknown field
# ---------------------------------------------------------------------------


def test_legacy_slots_payload_rejected_as_extra_field() -> None:
    """``extra="forbid"`` must reject pre-cut clients that send ``slots``."""
    # GIVEN a legacy slots payload (without intervals)
    # WHEN we validate
    # THEN ValidationError fires for the extra field
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate({"game_type": "el_farol", "slots": [0, 1, 2]})
    msg = str(exc.value).lower()
    assert "extra" in msg or "forbid" in msg or "not permitted" in msg


def test_intervals_plus_slots_rejected_as_extra_field() -> None:
    """Even with valid intervals present, an extra ``slots`` key is rejected."""
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate(
            {
                "game_type": "el_farol",
                "intervals": [[0, 1]],
                "slots": [3],
            }
        )
    msg = str(exc.value).lower()
    assert "extra" in msg or "forbid" in msg or "not permitted" in msg


# ---------------------------------------------------------------------------
# Happy paths for the canonical intervals shape
# ---------------------------------------------------------------------------


def test_native_intervals_single_pair() -> None:
    action = ElFarolAction.model_validate(
        {"game_type": "el_farol", "intervals": [[0, 1]]}
    )
    assert action.intervals == [[0, 1]]


def test_native_intervals_two_non_adjacent_pairs() -> None:
    action = ElFarolAction.model_validate(
        {"game_type": "el_farol", "intervals": [[0, 1], [3, 4]]}
    )
    assert action.intervals == [[0, 1], [3, 4]]


def test_native_intervals_empty_means_stay_home() -> None:
    action = ElFarolAction.model_validate(
        {"game_type": "el_farol", "intervals": []}
    )
    assert action.intervals == []


def test_native_intervals_preserves_reasoning_and_telemetry() -> None:
    action = ElFarolAction.model_validate(
        {
            "game_type": "el_farol",
            "intervals": [[0, 1]],
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
    assert action.intervals == [[0, 1]]
    assert action.reasoning == "hi"
    assert isinstance(action.telemetry, ActionTelemetry)
    assert action.telemetry.model_id == "gpt-4o-mini"
    assert action.telemetry.decide_ms == 234


# ---------------------------------------------------------------------------
# _validate_intervals: structural validation
# ---------------------------------------------------------------------------


def test_intervals_total_slots_over_max_rejected() -> None:
    # GIVEN a single pair covering 9 consecutive slots > MAX_SLOTS_PER_DAY=8
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate(
            {"game_type": "el_farol", "intervals": [[0, 8]]}
        )
    assert "max is" in str(exc.value).lower() or "slots" in str(exc.value).lower()


def test_intervals_adjacent_pairs_rejected() -> None:
    # GIVEN two pairs with no empty slot between them
    with pytest.raises(ValidationError) as exc:
        ElFarolAction.model_validate(
            {"game_type": "el_farol", "intervals": [[0, 1], [2, 3]]}
        )
    assert "overlap" in str(exc.value).lower() or "adjacent" in str(exc.value).lower()


def test_intervals_overlapping_pairs_rejected() -> None:
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate(
            {"game_type": "el_farol", "intervals": [[0, 4], [3, 6]]}
        )


def test_intervals_start_after_end_rejected() -> None:
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate(
            {"game_type": "el_farol", "intervals": [[5, 2]]}
        )


def test_intervals_negative_start_rejected() -> None:
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate(
            {"game_type": "el_farol", "intervals": [[-1, 2]]}
        )


def test_intervals_too_many_pairs_rejected() -> None:
    # GIVEN three intervals (max is _MAX_INTERVALS_PER_DAY=2)
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate(
            {
                "game_type": "el_farol",
                "intervals": [[0, 0], [2, 2], [4, 4]],
            }
        )


def test_intervals_bad_pair_shape_rejected() -> None:
    # GIVEN a triple instead of a [start, end] pair
    with pytest.raises(ValidationError):
        ElFarolAction.model_validate(
            {"game_type": "el_farol", "intervals": [[0, 1, 2]]}
        )
