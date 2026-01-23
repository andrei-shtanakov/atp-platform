"""Unit tests for StatisticsCalculator."""

import math

import pytest

from atp.statistics.calculator import (
    CV_MODERATE_THRESHOLD,
    CV_STABLE_THRESHOLD,
    CV_UNSTABLE_THRESHOLD,
    StatisticsCalculator,
    _get_t_critical,
)
from atp.statistics.models import StabilityLevel


class TestTCriticalValues:
    """Tests for t-critical value lookup."""

    def test_df_1(self) -> None:
        """Test t-critical for df=1."""
        assert _get_t_critical(1) == 12.706

    def test_df_10(self) -> None:
        """Test t-critical for df=10."""
        assert _get_t_critical(10) == 2.228

    def test_df_30(self) -> None:
        """Test t-critical for df=30."""
        assert _get_t_critical(30) == 2.042

    def test_df_greater_than_30(self) -> None:
        """Test t-critical for df>30 uses z-score."""
        assert _get_t_critical(50) == 1.96
        assert _get_t_critical(100) == 1.96

    def test_df_zero(self) -> None:
        """Test t-critical for df=0."""
        assert _get_t_critical(0) == float("inf")

    def test_df_negative(self) -> None:
        """Test t-critical for negative df."""
        assert _get_t_critical(-5) == float("inf")


class TestCalculateMean:
    """Tests for calculate_mean method."""

    def test_single_value(self) -> None:
        """Test mean of single value."""
        calc = StatisticsCalculator()
        assert calc.calculate_mean([42.0]) == 42.0

    def test_multiple_values(self) -> None:
        """Test mean of multiple values."""
        calc = StatisticsCalculator()
        assert calc.calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_negative_values(self) -> None:
        """Test mean with negative values."""
        calc = StatisticsCalculator()
        assert calc.calculate_mean([-2.0, -1.0, 0.0, 1.0, 2.0]) == 0.0

    def test_empty_raises_error(self) -> None:
        """Test that empty sequence raises ValueError."""
        calc = StatisticsCalculator()
        with pytest.raises(ValueError, match="Cannot calculate mean"):
            calc.calculate_mean([])


class TestCalculateStd:
    """Tests for calculate_std method."""

    def test_single_value(self) -> None:
        """Test std of single value returns 0."""
        calc = StatisticsCalculator()
        assert calc.calculate_std([42.0]) == 0.0

    def test_identical_values(self) -> None:
        """Test std of identical values is 0."""
        calc = StatisticsCalculator()
        assert calc.calculate_std([5.0, 5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_known_std(self) -> None:
        """Test std with known values."""
        calc = StatisticsCalculator()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        # Sample std for this data is approximately 2.138
        expected = 2.138089935299395
        assert calc.calculate_std(values) == pytest.approx(expected)

    def test_ddof_parameter(self) -> None:
        """Test std with different ddof values."""
        calc = StatisticsCalculator()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Population std (ddof=0)
        pop_std = calc.calculate_std(values, ddof=0)
        # Sample std (ddof=1)
        sample_std = calc.calculate_std(values, ddof=1)
        assert pop_std < sample_std  # Sample std is always larger


class TestCalculateMedian:
    """Tests for calculate_median method."""

    def test_single_value(self) -> None:
        """Test median of single value."""
        calc = StatisticsCalculator()
        assert calc.calculate_median([42.0]) == 42.0

    def test_odd_count(self) -> None:
        """Test median with odd number of values."""
        calc = StatisticsCalculator()
        assert calc.calculate_median([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_even_count(self) -> None:
        """Test median with even number of values."""
        calc = StatisticsCalculator()
        assert calc.calculate_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_unsorted_input(self) -> None:
        """Test median correctly sorts input."""
        calc = StatisticsCalculator()
        assert calc.calculate_median([5.0, 2.0, 1.0, 4.0, 3.0]) == 3.0

    def test_empty_raises_error(self) -> None:
        """Test that empty sequence raises ValueError."""
        calc = StatisticsCalculator()
        with pytest.raises(ValueError, match="Cannot calculate median"):
            calc.calculate_median([])


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval method."""

    def test_single_sample(self) -> None:
        """Test CI with single sample returns point estimate."""
        calc = StatisticsCalculator()
        lower, upper = calc.calculate_confidence_interval(mean=50.0, std=5.0, n=1)
        assert lower == 50.0
        assert upper == 50.0

    def test_symmetric_interval(self) -> None:
        """Test CI is symmetric around mean."""
        calc = StatisticsCalculator()
        lower, upper = calc.calculate_confidence_interval(mean=50.0, std=5.0, n=25)
        margin = (upper - lower) / 2
        assert lower == pytest.approx(50.0 - margin)
        assert upper == pytest.approx(50.0 + margin)

    def test_larger_sample_narrows_ci(self) -> None:
        """Test that larger sample size narrows CI."""
        calc = StatisticsCalculator()
        lower_small, upper_small = calc.calculate_confidence_interval(
            mean=50.0, std=5.0, n=5
        )
        lower_large, upper_large = calc.calculate_confidence_interval(
            mean=50.0, std=5.0, n=100
        )

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large
        assert width_large < width_small

    def test_known_ci(self) -> None:
        """Test CI with known values."""
        calc = StatisticsCalculator()
        # For n=25, df=24, t-critical = 2.064
        # Margin = 2.064 * 5 / sqrt(25) = 2.064 * 5 / 5 = 2.064
        lower, upper = calc.calculate_confidence_interval(mean=50.0, std=5.0, n=25)
        expected_margin = 2.064 * 5.0 / math.sqrt(25)
        assert lower == pytest.approx(50.0 - expected_margin)
        assert upper == pytest.approx(50.0 + expected_margin)


class TestCalculateCoefficientOfVariation:
    """Tests for calculate_coefficient_of_variation method."""

    def test_normal_case(self) -> None:
        """Test CV with normal values."""
        calc = StatisticsCalculator()
        cv = calc.calculate_coefficient_of_variation(mean=50.0, std=5.0)
        assert cv == pytest.approx(0.1)

    def test_zero_std(self) -> None:
        """Test CV with zero std."""
        calc = StatisticsCalculator()
        cv = calc.calculate_coefficient_of_variation(mean=50.0, std=0.0)
        assert cv == 0.0

    def test_zero_mean_zero_std(self) -> None:
        """Test CV with zero mean and zero std."""
        calc = StatisticsCalculator()
        cv = calc.calculate_coefficient_of_variation(mean=0.0, std=0.0)
        assert cv == 0.0

    def test_zero_mean_nonzero_std(self) -> None:
        """Test CV with zero mean and non-zero std."""
        calc = StatisticsCalculator()
        cv = calc.calculate_coefficient_of_variation(mean=0.0, std=5.0)
        assert cv == float("inf")

    def test_negative_mean(self) -> None:
        """Test CV uses absolute value of mean."""
        calc = StatisticsCalculator()
        cv = calc.calculate_coefficient_of_variation(mean=-50.0, std=5.0)
        assert cv == pytest.approx(0.1)


class TestAssessStability:
    """Tests for assess_stability method."""

    def test_stable(self) -> None:
        """Test stable assessment (CV < 0.05)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(0.03)
        assert assessment.level == StabilityLevel.STABLE
        assert assessment.cv == 0.03
        assert "Consistent" in assessment.message

    def test_stable_boundary(self) -> None:
        """Test stable boundary (CV = 0.049)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(CV_STABLE_THRESHOLD - 0.001)
        assert assessment.level == StabilityLevel.STABLE

    def test_moderate(self) -> None:
        """Test moderate assessment (0.05 <= CV < 0.15)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(0.10)
        assert assessment.level == StabilityLevel.MODERATE
        assert "Acceptable" in assessment.message

    def test_moderate_boundary_lower(self) -> None:
        """Test moderate lower boundary (CV = 0.05)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(CV_STABLE_THRESHOLD)
        assert assessment.level == StabilityLevel.MODERATE

    def test_moderate_boundary_upper(self) -> None:
        """Test moderate upper boundary (CV = 0.149)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(CV_MODERATE_THRESHOLD - 0.001)
        assert assessment.level == StabilityLevel.MODERATE

    def test_unstable(self) -> None:
        """Test unstable assessment (0.15 <= CV < 0.30)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(0.20)
        assert assessment.level == StabilityLevel.UNSTABLE
        assert "unreliable" in assessment.message

    def test_unstable_boundary_lower(self) -> None:
        """Test unstable lower boundary (CV = 0.15)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(CV_MODERATE_THRESHOLD)
        assert assessment.level == StabilityLevel.UNSTABLE

    def test_unstable_boundary_upper(self) -> None:
        """Test unstable upper boundary (CV = 0.299)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(CV_UNSTABLE_THRESHOLD - 0.001)
        assert assessment.level == StabilityLevel.UNSTABLE

    def test_critical(self) -> None:
        """Test critical assessment (CV >= 0.30)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(0.40)
        assert assessment.level == StabilityLevel.CRITICAL
        assert "Unpredictable" in assessment.message

    def test_critical_boundary(self) -> None:
        """Test critical boundary (CV = 0.30)."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(CV_UNSTABLE_THRESHOLD)
        assert assessment.level == StabilityLevel.CRITICAL

    def test_zero_cv(self) -> None:
        """Test zero CV is stable."""
        calc = StatisticsCalculator()
        assessment = calc.assess_stability(0.0)
        assert assessment.level == StabilityLevel.STABLE


class TestCompute:
    """Tests for compute method."""

    def test_basic_compute(self) -> None:
        """Test computing all statistics."""
        calc = StatisticsCalculator()
        values = [80.0, 85.0, 90.0, 85.0, 80.0]
        result = calc.compute(values)

        assert result.mean == pytest.approx(84.0)
        assert result.median == pytest.approx(85.0)
        assert result.min == 80.0
        assert result.max == 90.0
        assert result.n_runs == 5
        assert result.std >= 0
        assert result.coefficient_of_variation >= 0
        assert result.confidence_interval[0] <= result.mean
        assert result.confidence_interval[1] >= result.mean

    def test_empty_raises_error(self) -> None:
        """Test that empty sequence raises ValueError."""
        calc = StatisticsCalculator()
        with pytest.raises(ValueError, match="Cannot compute statistics"):
            calc.compute([])

    def test_single_value(self) -> None:
        """Test computing with single value."""
        calc = StatisticsCalculator()
        result = calc.compute([42.0])

        assert result.mean == 42.0
        assert result.median == 42.0
        assert result.min == 42.0
        assert result.max == 42.0
        assert result.n_runs == 1
        assert result.std == 0.0
        assert result.coefficient_of_variation == 0.0


class TestComputeWithStability:
    """Tests for compute_with_stability method."""

    def test_returns_both_results(self) -> None:
        """Test that method returns both result and stability."""
        calc = StatisticsCalculator()
        values = [80.0, 85.0, 90.0, 85.0, 80.0]
        result, stability = calc.compute_with_stability(values)

        assert result.mean == pytest.approx(84.0)
        assert stability.cv == result.coefficient_of_variation

    def test_stability_matches_cv(self) -> None:
        """Test stability assessment matches computed CV."""
        calc = StatisticsCalculator()
        # Low variance values
        low_var = [100.0, 100.0, 100.0, 100.0, 100.0]
        _, stability_low = calc.compute_with_stability(low_var)
        assert stability_low.level == StabilityLevel.STABLE

        # High variance values
        high_var = [50.0, 100.0, 50.0, 100.0, 50.0]
        _, stability_high = calc.compute_with_stability(high_var)
        assert stability_high.level in [
            StabilityLevel.UNSTABLE,
            StabilityLevel.CRITICAL,
        ]


class TestComputeTestStatistics:
    """Tests for compute_test_statistics method."""

    def test_with_scores_only(self) -> None:
        """Test computing with only score data."""
        calc = StatisticsCalculator()
        scores = [80.0, 85.0, 90.0, 85.0, 80.0]
        stats = calc.compute_test_statistics(
            test_id="test-001",
            scores=scores,
        )

        assert stats.test_id == "test-001"
        assert stats.n_runs == 5
        assert stats.score_stats is not None
        assert stats.score_stats.mean == pytest.approx(84.0)
        assert stats.score_stability is not None
        assert stats.duration_stats is None

    def test_with_all_metrics(self) -> None:
        """Test computing with all metrics."""
        calc = StatisticsCalculator()
        stats = calc.compute_test_statistics(
            test_id="test-002",
            scores=[80.0, 85.0, 90.0],
            durations=[1.5, 2.0, 1.8],
            steps=[10, 12, 11],
            tokens=[1000, 1200, 1100],
            costs=[0.01, 0.012, 0.011],
        )

        assert stats.test_id == "test-002"
        assert stats.n_runs == 3
        assert stats.score_stats is not None
        assert stats.duration_stats is not None
        assert stats.steps_stats is not None
        assert stats.tokens_stats is not None
        assert stats.cost_stats is not None

    def test_with_explicit_run_counts(self) -> None:
        """Test with explicit successful_runs and total_runs."""
        calc = StatisticsCalculator()
        stats = calc.compute_test_statistics(
            test_id="test-003",
            scores=[80.0, 85.0],
            successful_runs=5,
            total_runs=10,
        )

        assert stats.n_runs == 10
        assert stats.successful_runs == 5
        assert stats.success_rate == 0.5

    def test_no_data_raises_error(self) -> None:
        """Test that no data raises ValueError."""
        calc = StatisticsCalculator()
        with pytest.raises(ValueError, match="At least one data sequence"):
            calc.compute_test_statistics(test_id="test-004")

    def test_overall_stability_from_scores(self) -> None:
        """Test that overall stability is derived from score stability."""
        calc = StatisticsCalculator()
        # Highly consistent scores
        stats = calc.compute_test_statistics(
            test_id="test-005",
            scores=[85.0, 85.0, 85.0, 85.0, 85.0],
        )

        assert stats.overall_stability.level == StabilityLevel.STABLE
        assert stats.overall_stability == stats.score_stability
