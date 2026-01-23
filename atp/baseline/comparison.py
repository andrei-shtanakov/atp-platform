"""Comparison logic for baseline regression detection."""

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from atp.statistics.calculator import _get_t_critical

from .models import Baseline, ChangeType, TestBaseline


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class TestComparison(BaseModel):
    """Comparison result for a single test.

    Contains statistical comparison between current results and baseline,
    including p-value from Welch's t-test and change classification.
    """

    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    change_type: ChangeType = Field(..., description="Type of change detected")

    # Current values
    current_mean: float | None = Field(None, description="Current mean score")
    current_std: float | None = Field(None, description="Current standard deviation")
    current_n_runs: int | None = Field(None, description="Current number of runs")

    # Baseline values
    baseline_mean: float | None = Field(None, description="Baseline mean score")
    baseline_std: float | None = Field(None, description="Baseline standard deviation")
    baseline_n_runs: int | None = Field(None, description="Baseline number of runs")

    # Delta
    delta: float | None = Field(
        None, description="Absolute change (current - baseline)"
    )
    delta_percent: float | None = Field(
        None, description="Percentage change from baseline"
    )

    # Statistical significance
    t_statistic: float | None = Field(None, description="Welch's t-test statistic")
    p_value: float | None = Field(None, description="p-value from Welch's t-test")
    is_significant: bool = Field(
        default=False,
        description="Whether change is statistically significant (p < 0.05)",
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result: dict = {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "change_type": self.change_type.value,
        }

        if self.current_mean is not None:
            result["current"] = {
                "mean": round(self.current_mean, 4),
                "std": round(self.current_std or 0.0, 4),
                "n_runs": self.current_n_runs,
            }

        if self.baseline_mean is not None:
            result["baseline"] = {
                "mean": round(self.baseline_mean, 4),
                "std": round(self.baseline_std or 0.0, 4),
                "n_runs": self.baseline_n_runs,
            }

        if self.delta is not None:
            result["delta"] = round(self.delta, 4)
            result["delta_percent"] = (
                round(self.delta_percent, 2) if self.delta_percent is not None else None
            )

        if self.p_value is not None:
            result["statistics"] = {
                "t_statistic": round(self.t_statistic or 0.0, 4),
                "p_value": round(self.p_value, 6),
                "is_significant": self.is_significant,
            }

        return result


class ComparisonResult(BaseModel):
    """Complete comparison result for a test suite.

    Aggregates individual test comparisons and provides summary statistics.
    """

    suite_name: str = Field(..., description="Test suite name")
    agent_name: str = Field(..., description="Agent name")
    baseline_created_at: datetime | None = Field(
        None, description="When baseline was created"
    )
    compared_at: datetime = Field(
        default_factory=_utcnow, description="When comparison was performed"
    )

    # Summary counts
    total_tests: int = Field(default=0, description="Total tests compared")
    regressions: int = Field(default=0, description="Number of regressions detected")
    improvements: int = Field(default=0, description="Number of improvements detected")
    no_changes: int = Field(default=0, description="Number of unchanged tests")
    new_tests: int = Field(default=0, description="Number of new tests")
    removed_tests: int = Field(default=0, description="Number of removed tests")

    # Detailed comparisons
    comparisons: list[TestComparison] = Field(
        default_factory=list, description="Individual test comparisons"
    )

    @property
    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return self.regressions > 0

    @property
    def has_improvements(self) -> bool:
        """Check if any improvements were detected."""
        return self.improvements > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite_name": self.suite_name,
            "agent_name": self.agent_name,
            "baseline_created_at": (
                self.baseline_created_at.isoformat()
                if self.baseline_created_at
                else None
            ),
            "compared_at": self.compared_at.isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "regressions": self.regressions,
                "improvements": self.improvements,
                "no_changes": self.no_changes,
                "new_tests": self.new_tests,
                "removed_tests": self.removed_tests,
                "has_regressions": self.has_regressions,
                "has_improvements": self.has_improvements,
            },
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def welchs_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> tuple[float, float]:
    """Perform Welch's t-test for two samples with unequal variances.

    This test determines if there is a statistically significant difference
    between two means when the samples may have unequal variances.

    Args:
        mean1: Mean of first sample (current).
        std1: Standard deviation of first sample.
        n1: Size of first sample.
        mean2: Mean of second sample (baseline).
        std2: Standard deviation of second sample.
        n2: Size of second sample.

    Returns:
        Tuple of (t_statistic, p_value).
        Returns (0.0, 1.0) if calculation is not possible.
    """
    # Handle edge cases
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0)

    var1 = std1**2
    var2 = std2**2

    # Avoid division by zero
    se1 = var1 / n1
    se2 = var2 / n2
    se_sum = se1 + se2

    if se_sum < 1e-10:
        # Variances are essentially zero - if means are equal, no difference
        if abs(mean1 - mean2) < 1e-10:
            return (0.0, 1.0)
        # Otherwise, difference is effectively infinite
        return (float("inf"), 0.0)

    # Welch's t-statistic
    t_stat = (mean1 - mean2) / math.sqrt(se_sum)

    # Welch-Satterthwaite degrees of freedom
    numerator = se_sum**2
    denominator = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1))

    if denominator < 1e-10:
        df = min(n1, n2) - 1
    else:
        df = numerator / denominator

    # Calculate p-value using t-distribution approximation
    p_value = _calculate_p_value(abs(t_stat), df)

    return (t_stat, p_value)


def _calculate_p_value(t_stat: float, df: float) -> float:
    """Calculate two-tailed p-value from t-statistic and degrees of freedom.

    Uses an approximation based on comparison with critical values.
    For more accurate results, scipy.stats.t.sf() would be preferred,
    but we avoid the dependency.

    Args:
        t_stat: Absolute value of t-statistic.
        df: Degrees of freedom.

    Returns:
        Approximate two-tailed p-value.
    """
    if t_stat <= 0:
        return 1.0

    if df < 1:
        return 1.0

    # Get critical value for df
    t_critical_95 = _get_t_critical(int(df))

    # Simple approximation based on critical values
    # For |t| < t_critical_95, p > 0.05
    # For |t| >= t_critical_95, p <= 0.05

    if t_stat < t_critical_95 * 0.5:
        # Well below critical value - approximate high p-value
        return min(1.0, 1.0 - (t_stat / t_critical_95) * 0.5)
    elif t_stat < t_critical_95:
        # Approaching critical value
        ratio = t_stat / t_critical_95
        return max(0.05, 0.5 * (1.0 - ratio) + 0.05)
    elif t_stat < t_critical_95 * 1.5:
        # Just past critical value
        ratio = (t_stat - t_critical_95) / (t_critical_95 * 0.5)
        return max(0.01, 0.05 * (1.0 - ratio))
    elif t_stat < t_critical_95 * 2.5:
        # Significantly past critical value
        ratio = (t_stat - t_critical_95 * 1.5) / t_critical_95
        return max(0.001, 0.01 * (1.0 - ratio * 0.9))
    else:
        # Very large t-statistic
        return 0.001


def compare_test(
    current_scores: Sequence[float],
    baseline: TestBaseline,
    significance_level: float = 0.05,
) -> TestComparison:
    """Compare current test scores against baseline.

    Uses Welch's t-test to determine if there is a statistically
    significant change from the baseline.

    Args:
        current_scores: Scores from current test runs.
        baseline: Baseline data to compare against.
        significance_level: Threshold for significance (default 0.05).

    Returns:
        TestComparison with detailed comparison results.
    """
    if not current_scores:
        raise ValueError("Current scores cannot be empty")

    # Calculate current statistics
    n_current = len(current_scores)
    current_mean = sum(current_scores) / n_current

    if n_current > 1:
        squared_diffs = sum((x - current_mean) ** 2 for x in current_scores)
        variance = squared_diffs / (n_current - 1)
        current_std = math.sqrt(variance)
    else:
        current_std = 0.0

    # Calculate delta
    delta = current_mean - baseline.mean_score
    delta_percent = (
        (delta / baseline.mean_score * 100) if baseline.mean_score != 0 else 0.0
    )

    # Perform Welch's t-test
    t_stat, p_value = welchs_t_test(
        mean1=current_mean,
        std1=current_std,
        n1=n_current,
        mean2=baseline.mean_score,
        std2=baseline.std,
        n2=baseline.n_runs,
    )

    is_significant = p_value < significance_level

    # Determine change type
    if is_significant:
        if delta < 0:
            change_type = ChangeType.REGRESSION
        else:
            change_type = ChangeType.IMPROVEMENT
    else:
        change_type = ChangeType.NO_CHANGE

    return TestComparison(
        test_id=baseline.test_id,
        test_name=baseline.test_name,
        change_type=change_type,
        current_mean=current_mean,
        current_std=current_std,
        current_n_runs=n_current,
        baseline_mean=baseline.mean_score,
        baseline_std=baseline.std,
        baseline_n_runs=baseline.n_runs,
        delta=delta,
        delta_percent=delta_percent,
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
    )


def compare_results(
    current_scores: dict[str, list[float]],
    baseline: Baseline,
    test_names: dict[str, str] | None = None,
    significance_level: float = 0.05,
) -> ComparisonResult:
    """Compare current test results against a baseline.

    Args:
        current_scores: Dictionary mapping test_id to list of scores.
        baseline: Baseline to compare against.
        test_names: Optional mapping of test_id to test_name for new tests.
        significance_level: Threshold for significance (default 0.05).

    Returns:
        ComparisonResult with all test comparisons.
    """
    test_names = test_names or {}
    comparisons: list[TestComparison] = []

    # Track counts
    regressions = 0
    improvements = 0
    no_changes = 0
    new_tests = 0
    removed_tests = 0

    # Compare tests that exist in both
    all_test_ids = set(current_scores.keys()) | set(baseline.tests.keys())

    for test_id in sorted(all_test_ids):
        has_current = test_id in current_scores
        has_baseline = test_id in baseline.tests

        if has_current and has_baseline:
            # Compare existing test
            comparison = compare_test(
                current_scores=current_scores[test_id],
                baseline=baseline.tests[test_id],
                significance_level=significance_level,
            )
            comparisons.append(comparison)

            if comparison.change_type == ChangeType.REGRESSION:
                regressions += 1
            elif comparison.change_type == ChangeType.IMPROVEMENT:
                improvements += 1
            else:
                no_changes += 1

        elif has_current and not has_baseline:
            # New test
            scores = current_scores[test_id]
            current_mean = sum(scores) / len(scores)
            current_std = 0.0
            if len(scores) > 1:
                squared_diffs = sum((x - current_mean) ** 2 for x in scores)
                variance = squared_diffs / (len(scores) - 1)
                current_std = math.sqrt(variance)

            comparisons.append(
                TestComparison(
                    test_id=test_id,
                    test_name=test_names.get(test_id, test_id),
                    change_type=ChangeType.NEW_TEST,
                    current_mean=current_mean,
                    current_std=current_std,
                    current_n_runs=len(scores),
                )
            )
            new_tests += 1

        else:
            # Removed test
            baseline_test = baseline.tests[test_id]
            comparisons.append(
                TestComparison(
                    test_id=test_id,
                    test_name=baseline_test.test_name,
                    change_type=ChangeType.REMOVED_TEST,
                    baseline_mean=baseline_test.mean_score,
                    baseline_std=baseline_test.std,
                    baseline_n_runs=baseline_test.n_runs,
                )
            )
            removed_tests += 1

    return ComparisonResult(
        suite_name=baseline.suite_name,
        agent_name=baseline.agent_name,
        baseline_created_at=baseline.created_at,
        total_tests=len(comparisons),
        regressions=regressions,
        improvements=improvements,
        no_changes=no_changes,
        new_tests=new_tests,
        removed_tests=removed_tests,
        comparisons=comparisons,
    )
