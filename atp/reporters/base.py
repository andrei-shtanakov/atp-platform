"""Base reporter interface and result models."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from atp.evaluators.base import EvalResult
from atp.runner.models import SuiteResult
from atp.scoring.models import ScoredTestResult
from atp.statistics.models import TestRunStatistics


class TestReport(BaseModel):
    """Report data for a single test."""

    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    success: bool = Field(..., description="Whether the test passed")
    score: float | None = Field(None, description="Test score (0-100)")
    duration_seconds: float | None = Field(None, description="Test duration in seconds")
    total_runs: int = Field(default=1, description="Number of runs")
    successful_runs: int = Field(default=1, description="Number of successful runs")
    eval_results: list[EvalResult] = Field(
        default_factory=list, description="Evaluation results"
    )
    scored_result: ScoredTestResult | None = Field(
        None, description="Scored test result"
    )
    statistics: TestRunStatistics | None = Field(
        None, description="Statistical analysis for multiple runs"
    )
    error: str | None = Field(None, description="Error message if failed")


class SuiteReport(BaseModel):
    """Report data for a complete test suite."""

    suite_name: str = Field(..., description="Test suite name")
    agent_name: str = Field(..., description="Agent being tested")
    total_tests: int = Field(..., description="Total number of tests")
    passed_tests: int = Field(..., description="Number of passed tests")
    failed_tests: int = Field(..., description="Number of failed tests")
    success_rate: float = Field(..., description="Success rate (0.0-1.0)")
    duration_seconds: float | None = Field(
        None, description="Total duration in seconds"
    )
    runs_per_test: int = Field(default=1, description="Number of runs per test")
    tests: list[TestReport] = Field(default_factory=list, description="Test reports")
    error: str | None = Field(None, description="Suite-level error")

    @classmethod
    def from_suite_result(
        cls,
        result: SuiteResult,
        eval_results: dict[str, list[EvalResult]] | None = None,
        scored_results: dict[str, ScoredTestResult] | None = None,
        statistics: dict[str, TestRunStatistics] | None = None,
    ) -> "SuiteReport":
        """Create a SuiteReport from a SuiteResult.

        Args:
            result: Suite execution result.
            eval_results: Optional mapping of test_id to evaluation results.
            scored_results: Optional mapping of test_id to scored results.
            statistics: Optional mapping of test_id to statistical analysis.

        Returns:
            SuiteReport instance.
        """
        eval_results = eval_results or {}
        scored_results = scored_results or {}
        statistics = statistics or {}

        test_reports = []
        for test_result in result.tests:
            test_id = test_result.test.id
            scored = scored_results.get(test_id)

            test_report = TestReport(
                test_id=test_id,
                test_name=test_result.test.name,
                success=test_result.success,
                score=scored.score if scored else None,
                duration_seconds=test_result.duration_seconds,
                total_runs=test_result.total_runs,
                successful_runs=test_result.successful_runs,
                eval_results=eval_results.get(test_id, []),
                scored_result=scored,
                statistics=statistics.get(test_id),
                error=test_result.error,
            )
            test_reports.append(test_report)

        return cls(
            suite_name=result.suite_name,
            agent_name=result.agent_name,
            total_tests=result.total_tests,
            passed_tests=result.passed_tests,
            failed_tests=result.failed_tests,
            success_rate=result.success_rate,
            duration_seconds=result.duration_seconds,
            runs_per_test=result.runs_per_test,
            tests=test_reports,
            error=result.error,
        )


class Reporter(ABC):
    """Base class for reporters.

    Reporters format and output test results in various formats.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the reporter name."""

    @abstractmethod
    def report(self, report: SuiteReport) -> None:
        """Generate and output the report.

        Args:
            report: Suite report data to output.
        """

    @property
    def supports_streaming(self) -> bool:
        """Return whether this reporter supports streaming output.

        Default is False. Override in subclasses that support streaming.
        """
        return False

    def _format_duration(self, seconds: float | None) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string.
        """
        if seconds is None:
            return "N/A"

        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _format_score(self, score: float | None) -> str:
        """Format score for display.

        Args:
            score: Score value (0-100).

        Returns:
            Formatted score string.
        """
        if score is None:
            return "N/A"
        return f"{score:.1f}/100"
