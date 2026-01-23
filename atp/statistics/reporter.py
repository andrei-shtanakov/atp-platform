"""Statistical reporter for generating summaries from test results."""

from typing import Any

from atp.runner.models import SuiteResult, TestResult
from atp.scoring.models import ScoredTestResult

from .calculator import StatisticsCalculator
from .models import TestRunStatistics


class StatisticalReporter:
    """Generates statistical summaries for test results.

    Produces summaries including:
    - Mean, std, min, max, median for scores
    - 95% Confidence Intervals
    - Coefficient of Variation
    - Stability Assessment
    """

    def __init__(self) -> None:
        """Initialize the statistical reporter."""
        self._calculator = StatisticsCalculator()

    def compute_test_statistics(
        self,
        test_result: TestResult,
        scored_results: list[ScoredTestResult] | None = None,
    ) -> TestRunStatistics:
        """Compute statistics for a single test with multiple runs.

        Args:
            test_result: TestResult containing multiple RunResult objects.
            scored_results: Optional list of ScoredTestResult from scoring.
                           If provided, score statistics will be computed.

        Returns:
            TestRunStatistics with computed metrics.
        """
        # Extract scores if available
        scores: list[float] | None = None
        if scored_results:
            scores = [sr.score for sr in scored_results]

        # Extract durations
        durations = test_result.get_run_durations()

        # Extract steps
        steps = test_result.get_run_steps()

        # Extract tokens
        tokens = test_result.get_run_tokens()

        # Extract costs
        costs = test_result.get_run_costs()

        return self._calculator.compute_test_statistics(
            test_id=test_result.test.id,
            scores=scores if scores else None,
            durations=durations if durations else None,
            steps=steps if steps else None,
            tokens=tokens if tokens else None,
            costs=costs if costs else None,
            successful_runs=test_result.successful_runs,
            total_runs=test_result.total_runs,
        )

    def generate_summary(
        self,
        suite_result: SuiteResult,
        test_statistics: dict[str, TestRunStatistics] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete statistical summary for a suite.

        Args:
            suite_result: SuiteResult from running a test suite.
            test_statistics: Optional pre-computed statistics per test.

        Returns:
            Dictionary containing the statistical summary.
        """
        summary: dict[str, Any] = {
            "suite_name": suite_result.suite_name,
            "agent_name": suite_result.agent_name,
            "total_tests": suite_result.total_tests,
            "passed_tests": suite_result.passed_tests,
            "failed_tests": suite_result.failed_tests,
            "success_rate": round(suite_result.success_rate, 4),
            "total_runs": suite_result.total_runs,
            "runs_per_test": suite_result.runs_per_test,
            "duration_seconds": suite_result.duration_seconds,
        }

        # Add per-test statistics if available
        if test_statistics:
            summary["test_statistics"] = {
                test_id: stats.to_dict() for test_id, stats in test_statistics.items()
            }

        return summary

    def format_text_summary(
        self,
        suite_result: SuiteResult,
        test_statistics: dict[str, TestRunStatistics] | None = None,
    ) -> str:
        """Generate a human-readable text summary.

        Args:
            suite_result: SuiteResult from running a test suite.
            test_statistics: Optional pre-computed statistics per test.

        Returns:
            Formatted text summary.
        """
        lines: list[str] = []

        # Header
        lines.append("=" * 60)
        lines.append(f"Suite: {suite_result.suite_name}")
        lines.append(f"Agent: {suite_result.agent_name}")
        lines.append("=" * 60)

        # Overview
        lines.append("")
        lines.append("Overview:")
        lines.append(
            f"  Tests: {suite_result.passed_tests}/{suite_result.total_tests} passed"
        )
        lines.append(f"  Success Rate: {suite_result.success_rate * 100:.1f}%")
        lines.append(f"  Total Runs: {suite_result.total_runs}")
        lines.append(f"  Runs per Test: {suite_result.runs_per_test}")
        if suite_result.duration_seconds:
            lines.append(f"  Duration: {suite_result.duration_seconds:.2f}s")

        # Per-test statistics
        if test_statistics:
            lines.append("")
            lines.append("-" * 60)
            lines.append("Test Statistics:")
            lines.append("-" * 60)

            for test_id, stats in test_statistics.items():
                lines.append(f"\n  {test_id}:")
                lines.append(
                    f"    Runs: {stats.n_runs} ({stats.successful_runs} successful)"
                )
                lines.append(f"    Success Rate: {stats.success_rate * 100:.1f}%")
                lines.append(
                    f"    Stability: {stats.overall_stability.level.value} "
                    f"(CV={stats.overall_stability.cv:.4f})"
                )
                lines.append(f"    {stats.overall_stability.message}")

                if stats.score_stats:
                    s = stats.score_stats
                    lines.append(
                        f"    Score: {s.mean:.2f} +/- {s.std:.2f} "
                        f"[{s.min:.2f}, {s.max:.2f}]"
                    )
                    lines.append(
                        f"    95% CI: [{s.confidence_interval[0]:.2f}, "
                        f"{s.confidence_interval[1]:.2f}]"
                    )

                if stats.duration_stats:
                    d = stats.duration_stats
                    lines.append(
                        f"    Duration: {d.mean:.2f}s +/- {d.std:.2f}s "
                        f"[{d.min:.2f}s, {d.max:.2f}s]"
                    )

                if stats.steps_stats:
                    st = stats.steps_stats
                    lines.append(
                        f"    Steps: {st.mean:.1f} +/- {st.std:.1f} "
                        f"[{int(st.min)}, {int(st.max)}]"
                    )

                if stats.tokens_stats:
                    t = stats.tokens_stats
                    lines.append(
                        f"    Tokens: {t.mean:.0f} +/- {t.std:.0f} "
                        f"[{int(t.min)}, {int(t.max)}]"
                    )

                if stats.cost_stats:
                    c = stats.cost_stats
                    lines.append(
                        f"    Cost: ${c.mean:.4f} +/- ${c.std:.4f} "
                        f"[${c.min:.4f}, ${c.max:.4f}]"
                    )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def format_json_summary(
        self,
        suite_result: SuiteResult,
        test_statistics: dict[str, TestRunStatistics] | None = None,
    ) -> dict[str, Any]:
        """Generate JSON-serializable summary.

        Args:
            suite_result: SuiteResult from running a test suite.
            test_statistics: Optional pre-computed statistics per test.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return self.generate_summary(suite_result, test_statistics)
