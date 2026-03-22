"""Statistical calculations for multiple test runs."""

import math
from collections.abc import Sequence

from .models import (
    StabilityAssessment,
    StabilityLevel,
    StatisticalResult,
    TestRunStatistics,
)

# Stability thresholds from DESIGN-006
CV_STABLE_THRESHOLD = 0.05
CV_MODERATE_THRESHOLD = 0.15
CV_UNSTABLE_THRESHOLD = 0.30

# t-distribution critical values for 95% confidence interval
# Indexed by degrees of freedom (n-1), capped at df=30+
T_CRITICAL_VALUES: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _get_t_critical(df: int) -> float:
    """Get t-critical value for given degrees of freedom.

    For df > 30, uses asymptotic value of 1.96 (z-score).

    Args:
        df: Degrees of freedom (n - 1).

    Returns:
        t-critical value for 95% confidence interval.
    """
    if df <= 0:
        return float("inf")
    if df > 30:
        return 1.96
    return T_CRITICAL_VALUES.get(df, 1.96)


class StatisticsCalculator:
    """Calculator for statistical analysis of test run results.

    Implements DESIGN-006 statistics engine:
    - Mean, std, min, max, median
    - 95% Confidence Interval (t-distribution)
    - Coefficient of Variation
    - StabilityAssessment (stable/moderate/unstable/critical)
    """

    @staticmethod
    def calculate_mean(values: Sequence[float]) -> float:
        """Calculate arithmetic mean.

        Args:
            values: Sequence of numeric values.

        Returns:
            Arithmetic mean.

        Raises:
            ValueError: If values is empty.
        """
        if not values:
            raise ValueError("Cannot calculate mean of empty sequence")
        return sum(values) / len(values)

    @staticmethod
    def calculate_std(values: Sequence[float], ddof: int = 1) -> float:
        """Calculate sample standard deviation.

        Args:
            values: Sequence of numeric values.
            ddof: Delta degrees of freedom (default 1 for sample std).

        Returns:
            Sample standard deviation.

        Raises:
            ValueError: If values has fewer elements than required.
        """
        n = len(values)
        if n < ddof + 1:
            return 0.0  # Not enough data for std calculation

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - ddof)
        return math.sqrt(variance)

    @staticmethod
    def calculate_median(values: Sequence[float]) -> float:
        """Calculate median value.

        Args:
            values: Sequence of numeric values.

        Returns:
            Median value.

        Raises:
            ValueError: If values is empty.
        """
        if not values:
            raise ValueError("Cannot calculate median of empty sequence")

        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2

        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]

    @staticmethod
    def calculate_confidence_interval(
        mean: float,
        std: float,
        n: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval using t-distribution.

        Args:
            mean: Sample mean.
            std: Sample standard deviation.
            n: Sample size.
            confidence: Confidence level (default 0.95 for 95%).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        if n < 2:
            return (mean, mean)  # No CI possible with 1 sample

        df = n - 1
        t_critical = _get_t_critical(df)
        margin = t_critical * std / math.sqrt(n)

        return (mean - margin, mean + margin)

    @staticmethod
    def calculate_coefficient_of_variation(mean: float, std: float) -> float:
        """Calculate coefficient of variation (CV).

        CV = std / |mean|, with special handling for zero mean.

        Args:
            mean: Sample mean.
            std: Sample standard deviation.

        Returns:
            Coefficient of variation (always non-negative).
        """
        if abs(mean) < 1e-10:
            return 0.0 if std < 1e-10 else float("inf")
        return abs(std / mean)

    @staticmethod
    def assess_stability(cv: float) -> StabilityAssessment:
        """Assess stability based on coefficient of variation.

        Thresholds from DESIGN-006:
        - stable: CV < 0.05 - Consistent results
        - moderate: CV 0.05-0.15 - Acceptable variance
        - unstable: CV 0.15-0.30 - Results may be unreliable
        - critical: CV > 0.30 - Unpredictable behavior

        Args:
            cv: Coefficient of variation.

        Returns:
            StabilityAssessment with level, CV value, and message.
        """
        if cv < CV_STABLE_THRESHOLD:
            return StabilityAssessment(
                level=StabilityLevel.STABLE,
                cv=cv,
                message="Consistent results across runs",
            )
        elif cv < CV_MODERATE_THRESHOLD:
            return StabilityAssessment(
                level=StabilityLevel.MODERATE,
                cv=cv,
                message="Acceptable variance in results",
            )
        elif cv < CV_UNSTABLE_THRESHOLD:
            return StabilityAssessment(
                level=StabilityLevel.UNSTABLE,
                cv=cv,
                message="Results may be unreliable - high variance detected",
            )
        else:
            return StabilityAssessment(
                level=StabilityLevel.CRITICAL,
                cv=cv,
                message="Unpredictable behavior - very high variance",
            )

    def compute(self, values: Sequence[float]) -> StatisticalResult:
        """Compute all statistics for a sequence of values.

        Args:
            values: Sequence of numeric values.

        Returns:
            StatisticalResult with all computed statistics.

        Raises:
            ValueError: If values is empty.
        """
        if not values:
            raise ValueError("Cannot compute statistics for empty sequence")

        n = len(values)
        mean = self.calculate_mean(values)
        std = self.calculate_std(values)
        median = self.calculate_median(values)
        min_val = min(values)
        max_val = max(values)
        ci = self.calculate_confidence_interval(mean, std, n)
        cv = self.calculate_coefficient_of_variation(mean, std)

        return StatisticalResult(
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            confidence_interval=ci,
            n_runs=n,
            coefficient_of_variation=cv,
        )

    def compute_with_stability(
        self,
        values: Sequence[float],
    ) -> tuple[StatisticalResult, StabilityAssessment]:
        """Compute statistics and stability assessment.

        Args:
            values: Sequence of numeric values.

        Returns:
            Tuple of (StatisticalResult, StabilityAssessment).

        Raises:
            ValueError: If values is empty.
        """
        stats = self.compute(values)
        stability = self.assess_stability(stats.coefficient_of_variation)
        return stats, stability

    def compute_test_statistics(
        self,
        test_id: str,
        scores: Sequence[float] | None = None,
        durations: Sequence[float] | None = None,
        steps: Sequence[int] | None = None,
        tokens: Sequence[int] | None = None,
        costs: Sequence[float] | None = None,
        successful_runs: int | None = None,
        total_runs: int | None = None,
    ) -> TestRunStatistics:
        """Compute comprehensive statistics for a test.

        Args:
            test_id: Test identifier.
            scores: Test scores (0-100) from each run.
            durations: Execution durations in seconds.
            steps: Number of steps per run.
            tokens: Token counts per run.
            costs: Costs in USD per run.
            successful_runs: Number of successful runs (computed from scores if None).
            total_runs: Total number of runs (computed from scores if None).

        Returns:
            TestRunStatistics with all computed statistics.

        Raises:
            ValueError: If no data provided.
        """
        # Determine n_runs from available data
        all_data = [
            d for d in [scores, durations, steps, tokens, costs] if d is not None
        ]
        if not all_data:
            raise ValueError("At least one data sequence must be provided")

        n_runs = total_runs if total_runs is not None else len(all_data[0])

        # Compute score statistics
        score_stats = None
        score_stability = None
        if scores and len(scores) > 0:
            score_stats, score_stability = self.compute_with_stability(list(scores))

        # Compute duration statistics
        duration_stats = None
        if durations and len(durations) > 0:
            duration_stats = self.compute(list(durations))

        # Compute steps statistics
        steps_stats = None
        if steps and len(steps) > 0:
            steps_stats = self.compute([float(s) for s in steps])

        # Compute tokens statistics
        tokens_stats = None
        if tokens and len(tokens) > 0:
            tokens_stats = self.compute([float(t) for t in tokens])

        # Compute cost statistics
        cost_stats = None
        if costs and len(costs) > 0:
            cost_stats = self.compute(list(costs))

        # Determine success rate
        if successful_runs is not None:
            n_successful = successful_runs
        elif scores:
            n_successful = sum(1 for s in scores if s is not None)
        else:
            n_successful = n_runs  # Assume all successful if no score data

        success_rate = n_successful / n_runs if n_runs > 0 else 0.0

        # Compute overall stability (based on score stability or highest variance)
        if score_stability:
            overall_stability = score_stability
        elif score_stats:
            overall_stability = self.assess_stability(
                score_stats.coefficient_of_variation
            )
        else:
            # Use default stable if no score data
            overall_stability = StabilityAssessment(
                level=StabilityLevel.STABLE,
                cv=0.0,
                message="No score data available for stability assessment",
            )

        return TestRunStatistics(
            test_id=test_id,
            n_runs=n_runs,
            successful_runs=n_successful,
            success_rate=success_rate,
            score_stats=score_stats,
            score_stability=score_stability,
            duration_stats=duration_stats,
            steps_stats=steps_stats,
            tokens_stats=tokens_stats,
            cost_stats=cost_stats,
            overall_stability=overall_stability,
        )
