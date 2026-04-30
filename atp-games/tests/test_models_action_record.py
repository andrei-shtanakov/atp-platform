"""Tests for Phase 1 El Farol dashboard data models.

These tests cover four new models being added to atp_games.models:
- IntervalPair: frozen dataclass representing up to 2 contiguous slot intervals
- ActionRecord: one agent's action on one day (with outcome + optional metadata)
- DayAggregate: precomputed per-day per-slot attendance cache
- MatchConfig: game-level configuration persisted with every match

Tests are written TDD-first: the imports below target symbols that do not yet
exist, so the entire module is expected to fail with ImportError on collection
until Phase 1 implementation lands.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from datetime import UTC, datetime

import pytest

from atp_games.models import (
    ActionRecord,
    DayAggregate,
    IntervalPair,
    MatchConfig,
)

# ---------------------------------------------------------------------------
# IntervalPair
# ---------------------------------------------------------------------------


class TestIntervalPair:
    def test_accepts_single_interval(self) -> None:
        pair = IntervalPair(first=(0, 2), second=())
        assert pair.num_visits() == 1
        assert pair.covered_slots() == (0, 1, 2)
        assert pair.total_slots() == 3

    def test_accepts_two_intervals(self) -> None:
        pair = IntervalPair((0, 2), (4, 6))
        assert pair.num_visits() == 2
        assert pair.covered_slots() == (0, 1, 2, 4, 5, 6)
        assert pair.total_slots() == 6

    def test_accepts_empty(self) -> None:
        pair = IntervalPair((), ())
        assert pair.num_visits() == 0
        assert pair.covered_slots() == ()
        assert pair.total_slots() == 0

    def test_accepts_single_slot_visit(self) -> None:
        # A one-slot visit ((3, 3)) is a legitimate edge case.
        pair = IntervalPair((3, 3), ())
        assert pair.num_visits() == 1
        assert pair.total_slots() == 1
        assert pair.covered_slots() == (3,)

    def test_rejects_overlap(self) -> None:
        with pytest.raises(ValueError):
            IntervalPair((0, 4), (3, 7))

    def test_rejects_adjacency(self) -> None:
        # (0,3) ends at slot 3, (4,6) starts at slot 4 — adjacent, breaks
        # the "distinct visits" rule.
        with pytest.raises(ValueError):
            IntervalPair((0, 3), (4, 6))

    def test_accepts_non_adjacent_with_gap_of_one_slot(self) -> None:
        # (0,2) ends at slot 2, (4,6) starts at slot 4 — gap at slot 3.
        pair = IntervalPair((0, 2), (4, 6))
        assert pair.num_visits() == 2
        assert pair.total_slots() == 6

    def test_rejects_total_exceeding_max(self) -> None:
        # Default max_total_slots=8; (0,4) is 5 slots and (6,10) is 5 slots
        # -> 10 total, exceeds 8.
        with pytest.raises(ValueError):
            IntervalPair((0, 4), (6, 10))

        # Custom max_total_slots=4; a 5-slot interval alone exceeds the max.
        with pytest.raises(ValueError):
            IntervalPair((0, 4), (), max_total_slots=4)

    def test_rejects_out_of_range_slots(self) -> None:
        with pytest.raises(ValueError):
            IntervalPair((-1, 2), ())
        # slot 16 is out of range for num_slots=16 (valid indices: 0..15).
        with pytest.raises(ValueError):
            IntervalPair((0, 16), (), num_slots=16)

    def test_rejects_reversed_interval(self) -> None:
        # start > end is invalid.
        with pytest.raises(ValueError):
            IntervalPair((5, 2), ())

    def test_rejects_second_before_first(self) -> None:
        # Canonical ordering: the first interval must precede the second.
        with pytest.raises(ValueError):
            IntervalPair((5, 7), (0, 2))

    def test_covered_slots_sorted_unique(self) -> None:
        pair = IntervalPair((0, 2), (5, 7))
        covered = pair.covered_slots()
        assert covered == tuple(sorted(set(covered)))
        assert covered == (0, 1, 2, 5, 6, 7)

    def test_frozen(self) -> None:
        pair = IntervalPair((0, 2), ())
        with pytest.raises((FrozenInstanceError, AttributeError)):
            pair.first = (3, 5)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ActionRecord
# ---------------------------------------------------------------------------


def _make_required_action_kwargs() -> dict[str, object]:
    """Build the minimal required kwargs for ActionRecord construction."""
    intervals = IntervalPair((0, 2), ())
    return {
        "match_id": "m-001",
        "day": 1,
        "agent_id": "agent-a",
        "intervals": intervals,
        "picks": intervals.covered_slots(),
        "num_visits": intervals.num_visits(),
        "total_slots": intervals.total_slots(),
        "payoff": 1.5,
        "num_under": 2,
        "num_over": 0,
    }


class TestActionRecord:
    def test_required_fields_only(self) -> None:
        kwargs = _make_required_action_kwargs()
        record = ActionRecord(**kwargs)  # type: ignore[arg-type]

        assert record.match_id == "m-001"
        assert record.day == 1
        assert record.agent_id == "agent-a"
        assert record.intervals.covered_slots() == (0, 1, 2)
        assert record.picks == (0, 1, 2)
        assert record.num_visits == 1
        assert record.total_slots == 3
        assert record.payoff == 1.5
        assert record.num_under == 2
        assert record.num_over == 0

        # Optional fields default as documented.
        assert record.intent is None
        assert record.tokens_in is None
        assert record.tokens_out is None
        assert record.decide_ms is None
        assert record.cost_usd is None
        assert record.model_id is None
        assert record.retry_count == 0
        assert record.validation_error is None
        assert record.trace_id is None
        assert record.span_id is None
        assert record.submitted_at is None

    def test_with_all_fields(self) -> None:
        kwargs = _make_required_action_kwargs()
        submitted = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
        record = ActionRecord(
            **kwargs,  # type: ignore[arg-type]
            intent="visit early to avoid crowd",
            tokens_in=123,
            tokens_out=45,
            decide_ms=678,
            cost_usd=0.0009,
            model_id="gpt-4o-mini",
            retry_count=2,
            validation_error="too many slots",
            trace_id="00-abcd-ef01-00",
            span_id="span-42",
            submitted_at=submitted,
        )

        assert record.intent == "visit early to avoid crowd"
        assert record.tokens_in == 123
        assert record.tokens_out == 45
        assert record.decide_ms == 678
        assert record.cost_usd == pytest.approx(0.0009)
        assert record.model_id == "gpt-4o-mini"
        assert record.retry_count == 2
        assert record.validation_error == "too many slots"
        assert record.trace_id == "00-abcd-ef01-00"
        assert record.span_id == "span-42"
        assert record.submitted_at == submitted

    def test_asdict_roundtrip(self) -> None:
        kwargs = _make_required_action_kwargs()
        record = ActionRecord(**kwargs)  # type: ignore[arg-type]

        data = asdict(record)
        assert isinstance(data, dict)

        expected_keys = {
            "match_id",
            "day",
            "agent_id",
            "intervals",
            "picks",
            "num_visits",
            "total_slots",
            "payoff",
            "num_under",
            "num_over",
            "intent",
            "tokens_in",
            "tokens_out",
            "decide_ms",
            "cost_usd",
            "model_id",
            "retry_count",
            "validation_error",
            "trace_id",
            "span_id",
            "submitted_at",
        }
        assert expected_keys.issubset(data.keys())

        # Scalar fields preserve their values and types.
        assert data["match_id"] == "m-001"
        assert data["day"] == 1
        assert data["agent_id"] == "agent-a"
        assert data["payoff"] == pytest.approx(1.5)
        assert data["num_under"] == 2
        assert data["num_over"] == 0
        assert data["intent"] is None
        assert data["retry_count"] == 0

    def test_intent_defaults_none(self) -> None:
        kwargs = _make_required_action_kwargs()
        record = ActionRecord(**kwargs)  # type: ignore[arg-type]
        assert record.intent is None

    def test_retry_count_defaults_zero(self) -> None:
        kwargs = _make_required_action_kwargs()
        record = ActionRecord(**kwargs)  # type: ignore[arg-type]
        assert record.retry_count == 0


# ---------------------------------------------------------------------------
# DayAggregate
# ---------------------------------------------------------------------------


class TestDayAggregate:
    def test_construct_basic(self) -> None:
        slot_attendance = tuple([0] * 16)
        agg = DayAggregate(
            match_id="m-001",
            day=5,
            slot_attendance=slot_attendance,
            over_slots=0,
            total_attendances=0,
        )
        assert agg.match_id == "m-001"
        assert agg.day == 5
        assert agg.slot_attendance == slot_attendance
        assert len(agg.slot_attendance) == 16
        assert agg.over_slots == 0
        assert agg.total_attendances == 0

    def test_over_slots_computed_externally(self) -> None:
        # over_slots is passed in by the caller (runner computes it against
        # the capacity_threshold); the dataclass just stores the value.
        slot_attendance = tuple([10] * 16)
        agg = DayAggregate(
            match_id="m-001",
            day=2,
            slot_attendance=slot_attendance,
            over_slots=7,
            total_attendances=160,
        )
        assert agg.over_slots == 7

    def test_total_attendances_stored(self) -> None:
        slot_attendance = (1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0)
        agg = DayAggregate(
            match_id="m-001",
            day=3,
            slot_attendance=slot_attendance,
            over_slots=2,
            total_attendances=36,
        )
        assert agg.total_attendances == 36
        assert agg.slot_attendance == slot_attendance


# ---------------------------------------------------------------------------
# MatchConfig
# ---------------------------------------------------------------------------


class TestMatchConfig:
    def test_defaults(self) -> None:
        cfg = MatchConfig(
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
        )
        assert cfg.game_id == "el_farol_interval"
        assert cfg.game_version == "1.0.0"
        assert cfg.num_days == 100
        assert cfg.num_slots == 16
        assert cfg.max_intervals == 2
        assert cfg.max_total_slots == 8
        assert cfg.capacity_ratio == pytest.approx(0.6)
        assert cfg.capacity_threshold == 0
        assert cfg.seed is None

    def test_resolve_threshold_8_agents(self) -> None:
        cfg = MatchConfig.resolve(
            num_agents=8,
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
        )
        # floor(0.6 * 8) = floor(4.8) = 4
        assert cfg.capacity_threshold == 4

    def test_resolve_threshold_16_agents(self) -> None:
        cfg = MatchConfig.resolve(
            num_agents=16,
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
        )
        # floor(0.6 * 16) = floor(9.6) = 9
        assert cfg.capacity_threshold == 9

    def test_resolve_threshold_10_agents(self) -> None:
        cfg = MatchConfig.resolve(
            num_agents=10,
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
        )
        # floor(0.6 * 10) = floor(6.0) = 6
        assert cfg.capacity_threshold == 6

    def test_resolve_respects_custom_ratio(self) -> None:
        cfg = MatchConfig.resolve(
            num_agents=10,
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
            capacity_ratio=0.5,
        )
        # floor(0.5 * 10) = 5
        assert cfg.capacity_threshold == 5
        assert cfg.capacity_ratio == pytest.approx(0.5)

    def test_frozen(self) -> None:
        cfg = MatchConfig(
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.num_days = 200  # type: ignore[misc]

    def test_explicit_threshold_preserved_when_nonzero(self) -> None:
        # Direct construction with a nonzero capacity_threshold must be
        # preserved — only MatchConfig.resolve() derives it.
        cfg = MatchConfig(
            game_id="el_farol_interval",
            game_version="1.0.0",
            num_days=100,
            capacity_threshold=5,
        )
        assert cfg.capacity_threshold == 5
