"""Tests for Phase 5 of the El Farol dashboard data-model plan.

Covers the extended ``ElFarolConfig`` with:

  - ``max_intervals: int = 2``
  - ``max_total_slots: int = 8``
  - ``capacity_ratio: float = 0.6``
  - ``capacity_threshold: int = 0`` treated as a sentinel meaning
    "derive from ``floor(capacity_ratio * num_players)`` in
    ``__post_init__``".

Plus additional structural validation rules:

  - ``max_total_slots <= num_slots``
  - ``max_intervals <= max_total_slots``
  - ``capacity_ratio`` strictly in ``(0, 1]``
  - resolved ``capacity_threshold`` must be ``>= 1``
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from game_envs.games.el_farol import ElFarolConfig


class TestElFarolConfigDefaults:
    def test_defaults(self) -> None:
        # GIVEN no arguments
        # WHEN constructing with defaults
        cfg = ElFarolConfig()
        # THEN all Phase-5 defaults are present and threshold is derived
        assert cfg.num_players == 100
        assert cfg.num_slots == 16
        assert cfg.max_intervals == 2
        assert cfg.max_total_slots == 8
        assert cfg.capacity_ratio == pytest.approx(0.6)
        # Derived via floor(0.6 * 100) = 60
        assert cfg.capacity_threshold == 60


class TestCapacityThresholdDerivation:
    def test_ratio_0_6_times_8_players_gives_threshold_4(self) -> None:
        # GIVEN num_players=8 and default capacity_ratio=0.6
        cfg = ElFarolConfig(num_players=8, num_slots=16)
        # THEN floor(0.6 * 8) = 4
        assert cfg.capacity_threshold == 4

    def test_ratio_0_6_times_16_players_gives_threshold_9(self) -> None:
        # GIVEN num_players=16 and default capacity_ratio=0.6
        cfg = ElFarolConfig(num_players=16, num_slots=16)
        # THEN floor(0.6 * 16) = floor(9.6) = 9
        assert cfg.capacity_threshold == 9

    def test_ratio_0_6_times_10_players_gives_threshold_6(self) -> None:
        # GIVEN num_players=10 and default capacity_ratio=0.6
        cfg = ElFarolConfig(num_players=10, num_slots=16)
        # THEN floor(0.6 * 10) = 6
        assert cfg.capacity_threshold == 6

    def test_explicit_threshold_preserved(self) -> None:
        # GIVEN an explicit non-zero capacity_threshold
        cfg = ElFarolConfig(num_players=100, capacity_threshold=50)
        # THEN derivation does not override the explicit value
        assert cfg.capacity_threshold == 50

    def test_custom_ratio_applied(self) -> None:
        # GIVEN a custom capacity_ratio
        cfg = ElFarolConfig(num_players=10, num_slots=16, capacity_ratio=0.5)
        # THEN floor(0.5 * 10) = 5
        assert cfg.capacity_threshold == 5


class TestStructuralValidation:
    def test_rejects_max_total_slots_greater_than_num_slots(self) -> None:
        # GIVEN max_total_slots > num_slots
        # WHEN constructing
        # THEN ValueError
        with pytest.raises(ValueError):
            ElFarolConfig(num_slots=10, max_total_slots=11)

    def test_rejects_max_intervals_greater_than_max_total_slots(self) -> None:
        # GIVEN max_intervals > max_total_slots
        # WHEN constructing
        # THEN ValueError
        with pytest.raises(ValueError):
            ElFarolConfig(max_intervals=5, max_total_slots=4)


class TestCapacityRatioBounds:
    def test_rejects_capacity_ratio_zero(self) -> None:
        # GIVEN capacity_ratio == 0.0 (excluded lower bound)
        # WHEN constructing
        # THEN ValueError
        with pytest.raises(ValueError):
            ElFarolConfig(capacity_ratio=0.0)

    def test_rejects_capacity_ratio_negative(self) -> None:
        # GIVEN capacity_ratio < 0
        # WHEN constructing
        # THEN ValueError
        with pytest.raises(ValueError):
            ElFarolConfig(capacity_ratio=-0.1)

    def test_rejects_capacity_ratio_above_one(self) -> None:
        # GIVEN capacity_ratio > 1
        # WHEN constructing
        # THEN ValueError
        with pytest.raises(ValueError):
            ElFarolConfig(capacity_ratio=1.5)

    def test_accepts_capacity_ratio_exactly_one(self) -> None:
        # GIVEN capacity_ratio == 1.0 (included upper bound)
        # WHEN constructing
        # THEN no error; threshold = floor(1.0 * num_players)
        cfg = ElFarolConfig(num_players=10, num_slots=16, capacity_ratio=1.0)
        assert cfg.capacity_threshold == 10


class TestImmutability:
    def test_frozen(self) -> None:
        # GIVEN a constructed config
        cfg = ElFarolConfig()
        # WHEN attempting to mutate a field
        # THEN FrozenInstanceError (subclass of AttributeError)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.num_slots = 20  # type: ignore[misc]


class TestResolvedThresholdValidation:
    def test_resolved_threshold_also_rejects_below_one(self) -> None:
        # GIVEN num_players=1 with capacity_ratio=0.6 -> floor(0.6) = 0
        # WHEN constructing
        # THEN ValueError (resolved threshold must be >= 1)
        with pytest.raises(ValueError):
            ElFarolConfig(num_players=1, capacity_ratio=0.6)
