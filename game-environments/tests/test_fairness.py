"""Tests for fairness metrics analysis."""

from __future__ import annotations

import pytest

from game_envs.analysis.fairness import (
    FairnessMetrics,
    envy_freeness,
    gini_coefficient,
    proportionality,
    utilitarian_welfare,
)

# --- gini_coefficient tests ---


class TestGiniCoefficient:
    def test_perfect_equality(self):
        """Equal payoffs → Gini = 0."""
        payoffs = {"a": 10.0, "b": 10.0, "c": 10.0}
        assert gini_coefficient(payoffs) == 0.0

    def test_maximum_inequality_two_players(self):
        """One gets everything → Gini = 0.5 for 2 players."""
        payoffs = {"a": 0.0, "b": 100.0}
        g = gini_coefficient(payoffs)
        assert abs(g - 0.5) < 1e-10

    def test_unequal_payoffs(self):
        """Gini > 0 for unequal payoffs."""
        payoffs = {"a": 1.0, "b": 2.0, "c": 3.0}
        g = gini_coefficient(payoffs)
        assert g > 0.0

    def test_single_player(self):
        payoffs = {"a": 42.0}
        assert gini_coefficient(payoffs) == 0.0

    def test_all_zero_payoffs(self):
        payoffs = {"a": 0.0, "b": 0.0}
        assert gini_coefficient(payoffs) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty payoffs"):
            gini_coefficient({})

    def test_known_value(self):
        """Known Gini for {1, 2, 3, 4, 5}."""
        payoffs = {str(i): float(i) for i in range(1, 6)}
        g = gini_coefficient(payoffs)
        # Gini for {1,2,3,4,5}: mean=3, sum|xi-xj| = 40
        # Gini = 40 / (2*5*15) = 40/150 = 0.2667
        assert abs(g - 4 / 15) < 1e-10


# --- envy_freeness tests ---


class TestEnvyFreeness:
    def test_equal_allocation(self):
        """Equal payoffs → envy-free."""
        is_free, pairs = envy_freeness({"a": 5.0, "b": 5.0, "c": 5.0})
        assert is_free is True
        assert pairs == []

    def test_unequal_allocation(self):
        """Unequal payoffs → not envy-free."""
        is_free, pairs = envy_freeness({"a": 1.0, "b": 10.0})
        assert is_free is False
        assert ("a", "b") in pairs
        assert ("b", "a") not in pairs

    def test_three_players_envy(self):
        alloc = {"a": 1.0, "b": 5.0, "c": 10.0}
        is_free, pairs = envy_freeness(alloc)
        assert is_free is False
        # a envies b and c, b envies c
        assert ("a", "b") in pairs
        assert ("a", "c") in pairs
        assert ("b", "c") in pairs
        assert len(pairs) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty allocations"):
            envy_freeness({})


# --- proportionality tests ---


class TestProportionality:
    def test_equal_payoffs_equal_entitlements(self):
        """Equal split with equal entitlements → 1.0."""
        payoffs = {"a": 10.0, "b": 10.0}
        assert proportionality(payoffs) == 1.0

    def test_perfect_proportionality(self):
        """Payoffs match entitlements exactly."""
        payoffs = {"a": 20.0, "b": 30.0}
        entitlements = {"a": 2.0, "b": 3.0}
        assert abs(proportionality(payoffs, entitlements) - 1.0) < 1e-10

    def test_disproportional(self):
        """Payoffs don't match entitlements."""
        payoffs = {"a": 50.0, "b": 50.0}
        entitlements = {"a": 1.0, "b": 3.0}
        p = proportionality(payoffs, entitlements)
        assert 0.0 < p < 1.0

    def test_all_zero_payoffs(self):
        payoffs = {"a": 0.0, "b": 0.0}
        p = proportionality(payoffs)
        assert p == 1.0

    def test_missing_entitlements_raises(self):
        with pytest.raises(ValueError, match="missing"):
            proportionality(
                {"a": 10.0, "b": 20.0},
                {"a": 1.0},
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty payoffs"):
            proportionality({})


# --- utilitarian_welfare tests ---


class TestUtilitarianWelfare:
    def test_sum_of_payoffs(self):
        payoffs = {"a": 10.0, "b": 20.0, "c": 30.0}
        assert utilitarian_welfare(payoffs) == 60.0

    def test_negative_payoffs(self):
        payoffs = {"a": -5.0, "b": 10.0}
        assert utilitarian_welfare(payoffs) == 5.0

    def test_zero_payoffs(self):
        payoffs = {"a": 0.0, "b": 0.0}
        assert utilitarian_welfare(payoffs) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty payoffs"):
            utilitarian_welfare({})


# --- FairnessMetrics serialization ---


class TestFairnessMetricsSerialization:
    def test_round_trip(self):
        metrics = FairnessMetrics(
            gini=0.3,
            envy_free=False,
            envy_pairs=[("a", "b")],
            proportionality=0.8,
            utilitarian_welfare=100.0,
        )
        data = metrics.to_dict()
        restored = FairnessMetrics.from_dict(data)
        assert restored.gini == 0.3
        assert restored.envy_free is False
        assert restored.envy_pairs == [("a", "b")]
        assert restored.proportionality == 0.8
        assert restored.utilitarian_welfare == 100.0

    def test_envy_free_round_trip(self):
        metrics = FairnessMetrics(
            gini=0.0,
            envy_free=True,
            envy_pairs=[],
            proportionality=1.0,
            utilitarian_welfare=50.0,
        )
        data = metrics.to_dict()
        restored = FairnessMetrics.from_dict(data)
        assert restored.envy_free is True
        assert restored.envy_pairs == []
