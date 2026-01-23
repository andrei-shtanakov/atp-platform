"""Tests for baseline comparison module."""

import math

import pytest

from atp.baseline.comparison import (
    ComparisonResult,
    TestComparison,
    compare_results,
    compare_test,
    welchs_t_test,
)
from atp.baseline.models import Baseline, ChangeType, TestBaseline


class TestWelchsTTest:
    """Tests for Welch's t-test implementation."""

    def test_identical_samples(self) -> None:
        """Test t-test with identical samples."""
        t_stat, p_value = welchs_t_test(
            mean1=50.0,
            std1=5.0,
            n1=10,
            mean2=50.0,
            std2=5.0,
            n2=10,
        )

        assert t_stat == 0.0
        assert p_value == 1.0

    def test_different_means(self) -> None:
        """Test t-test with significantly different means."""
        # Large difference, small variance -> significant
        t_stat, p_value = welchs_t_test(
            mean1=80.0,
            std1=2.0,
            n1=20,
            mean2=70.0,
            std2=2.0,
            n2=20,
        )

        assert t_stat > 0  # current > baseline
        assert p_value < 0.05  # Should be significant

    def test_similar_means_high_variance(self) -> None:
        """Test t-test with similar means but high variance."""
        # Small difference, large variance -> not significant
        t_stat, p_value = welchs_t_test(
            mean1=75.0,
            std1=15.0,
            n1=5,
            mean2=72.0,
            std2=15.0,
            n2=5,
        )

        # With high variance and small sample, should not be significant
        assert p_value > 0.05

    def test_small_sample_size(self) -> None:
        """Test t-test with small sample sizes."""
        t_stat, p_value = welchs_t_test(
            mean1=80.0,
            std1=5.0,
            n1=2,
            mean2=70.0,
            std2=5.0,
            n2=2,
        )

        # With n < 2, calculation is limited
        # But n=2 should work
        assert not math.isnan(t_stat)
        assert not math.isnan(p_value)

    def test_edge_case_n_equals_one(self) -> None:
        """Test t-test when n=1."""
        t_stat, p_value = welchs_t_test(
            mean1=80.0,
            std1=0.0,
            n1=1,
            mean2=70.0,
            std2=5.0,
            n2=10,
        )

        # Should return safe defaults
        assert p_value == 1.0

    def test_zero_variance(self) -> None:
        """Test t-test with zero variance (all values identical)."""
        t_stat, p_value = welchs_t_test(
            mean1=75.0,
            std1=0.0,
            n1=10,
            mean2=75.0,
            std2=0.0,
            n2=10,
        )

        # Same means, zero variance = no difference
        assert p_value == 1.0


class TestCompareTest:
    """Tests for compare_test function."""

    def test_no_change(self) -> None:
        """Test comparison with no significant change."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=75.0,
            std=5.0,
            n_runs=10,
            ci_95=(70.0, 80.0),
            success_rate=1.0,
        )

        result = compare_test(
            current_scores=[74.0, 76.0, 75.0, 73.0, 77.0],
            baseline=baseline,
        )

        assert result.change_type == ChangeType.NO_CHANGE
        assert not result.is_significant
        assert result.p_value is not None
        assert result.p_value > 0.05

    def test_regression_detected(self) -> None:
        """Test detection of regression."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=85.0,
            std=2.0,
            n_runs=20,
            ci_95=(84.0, 86.0),
            success_rate=1.0,
        )

        # Significantly lower scores
        result = compare_test(
            current_scores=[70.0, 72.0, 71.0, 69.0, 73.0] * 4,
            baseline=baseline,
        )

        assert result.change_type == ChangeType.REGRESSION
        assert result.is_significant
        assert result.delta is not None
        assert result.delta < 0  # Current is worse

    def test_improvement_detected(self) -> None:
        """Test detection of improvement."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=70.0,
            std=2.0,
            n_runs=20,
            ci_95=(69.0, 71.0),
            success_rate=0.9,
        )

        # Significantly higher scores
        result = compare_test(
            current_scores=[85.0, 87.0, 86.0, 84.0, 88.0] * 4,
            baseline=baseline,
        )

        assert result.change_type == ChangeType.IMPROVEMENT
        assert result.is_significant
        assert result.delta is not None
        assert result.delta > 0  # Current is better

    def test_delta_calculation(self) -> None:
        """Test delta and delta_percent calculation."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=80.0,
            std=2.0,
            n_runs=10,
            ci_95=(78.0, 82.0),
            success_rate=1.0,
        )

        result = compare_test(
            current_scores=[90.0, 90.0, 90.0, 90.0, 90.0],
            baseline=baseline,
        )

        assert result.delta == 10.0  # 90 - 80
        assert result.delta_percent == 12.5  # 10/80 * 100

    def test_empty_scores_raises_error(self) -> None:
        """Test that empty scores raises ValueError."""
        baseline = TestBaseline(
            test_id="test-1",
            test_name="Test One",
            mean_score=80.0,
            std=2.0,
            n_runs=10,
            ci_95=(78.0, 82.0),
            success_rate=1.0,
        )

        with pytest.raises(ValueError):
            compare_test(current_scores=[], baseline=baseline)


class TestCompareResults:
    """Tests for compare_results function."""

    def test_basic_comparison(self) -> None:
        """Test basic comparison of results."""
        baseline = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=5,
            tests={
                "test-1": TestBaseline(
                    test_id="test-1",
                    test_name="Test One",
                    mean_score=80.0,
                    std=2.0,
                    n_runs=5,
                    ci_95=(78.0, 82.0),
                    success_rate=1.0,
                ),
            },
        )

        current_scores = {
            "test-1": [79.0, 80.0, 81.0, 80.0, 79.0],
        }

        result = compare_results(current_scores, baseline)

        assert result.suite_name == "test-suite"
        assert result.agent_name == "test-agent"
        assert result.total_tests == 1
        assert len(result.comparisons) == 1

    def test_new_test_detection(self) -> None:
        """Test detection of new tests."""
        baseline = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=5,
            tests={},  # No tests in baseline
        )

        current_scores = {
            "test-new": [85.0, 86.0, 84.0],
        }
        test_names = {"test-new": "New Test"}

        result = compare_results(current_scores, baseline, test_names)

        assert result.new_tests == 1
        assert result.comparisons[0].change_type == ChangeType.NEW_TEST

    def test_removed_test_detection(self) -> None:
        """Test detection of removed tests."""
        baseline = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=5,
            tests={
                "test-old": TestBaseline(
                    test_id="test-old",
                    test_name="Old Test",
                    mean_score=80.0,
                    std=2.0,
                    n_runs=5,
                    ci_95=(78.0, 82.0),
                    success_rate=1.0,
                ),
            },
        )

        current_scores: dict[str, list[float]] = {}  # No current tests

        result = compare_results(current_scores, baseline)

        assert result.removed_tests == 1
        assert result.comparisons[0].change_type == ChangeType.REMOVED_TEST

    def test_summary_counts(self) -> None:
        """Test summary counts in comparison result."""
        baseline = Baseline(
            suite_name="test-suite",
            agent_name="test-agent",
            runs_per_test=20,
            tests={
                "test-regression": TestBaseline(
                    test_id="test-regression",
                    test_name="Regression Test",
                    mean_score=90.0,
                    std=2.0,
                    n_runs=20,
                    ci_95=(89.0, 91.0),
                    success_rate=1.0,
                ),
                "test-stable": TestBaseline(
                    test_id="test-stable",
                    test_name="Stable Test",
                    mean_score=75.0,
                    std=3.0,
                    n_runs=20,
                    ci_95=(73.0, 77.0),
                    success_rate=1.0,
                ),
                "test-removed": TestBaseline(
                    test_id="test-removed",
                    test_name="Removed Test",
                    mean_score=80.0,
                    std=2.0,
                    n_runs=20,
                    ci_95=(78.0, 82.0),
                    success_rate=1.0,
                ),
            },
        )

        current_scores = {
            "test-regression": [70.0, 71.0, 69.0, 70.0, 72.0] * 4,  # Lower
            "test-stable": [74.0, 76.0, 75.0, 74.0, 76.0] * 4,  # Same
            "test-new": [85.0, 86.0, 84.0, 85.0, 87.0] * 4,  # New
        }
        test_names = {"test-new": "New Test"}

        result = compare_results(current_scores, baseline, test_names)

        assert result.total_tests == 4
        assert result.regressions == 1
        assert result.new_tests == 1
        assert result.removed_tests == 1
        # The stable test might still show as no_change or slight variance


class TestTestComparison:
    """Tests for TestComparison model."""

    def test_to_dict(self) -> None:
        """Test converting TestComparison to dict."""
        comparison = TestComparison(
            test_id="test-1",
            test_name="Test One",
            change_type=ChangeType.REGRESSION,
            current_mean=70.0,
            current_std=3.0,
            current_n_runs=5,
            baseline_mean=80.0,
            baseline_std=2.0,
            baseline_n_runs=10,
            delta=-10.0,
            delta_percent=-12.5,
            t_statistic=-5.0,
            p_value=0.001,
            is_significant=True,
        )

        result = comparison.to_dict()

        assert result["test_id"] == "test-1"
        assert result["change_type"] == "regression"
        assert result["current"]["mean"] == 70.0
        assert result["baseline"]["mean"] == 80.0
        assert result["delta"] == -10.0
        assert result["statistics"]["p_value"] == 0.001


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    def test_has_regressions(self) -> None:
        """Test has_regressions property."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            regressions=1,
            improvements=0,
        )
        assert result.has_regressions is True

        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            regressions=0,
            improvements=0,
        )
        assert result.has_regressions is False

    def test_has_improvements(self) -> None:
        """Test has_improvements property."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            regressions=0,
            improvements=2,
        )
        assert result.has_improvements is True

    def test_to_dict(self) -> None:
        """Test converting ComparisonResult to dict."""
        result = ComparisonResult(
            suite_name="test-suite",
            agent_name="test-agent",
            total_tests=5,
            regressions=1,
            improvements=1,
            no_changes=2,
            new_tests=1,
            removed_tests=0,
        )

        data = result.to_dict()

        assert data["suite_name"] == "test-suite"
        assert data["summary"]["total_tests"] == 5
        assert data["summary"]["regressions"] == 1
        assert data["summary"]["improvements"] == 1
        assert data["summary"]["has_regressions"] is True
        assert data["summary"]["has_improvements"] is True
