"""Shared result models for the ATP pipeline.

These data models are used across the platform:
evaluators → runner → reporters → dashboard.

Placed in atp-core so that atp-dashboard can depend on atp-core
without a circular dependency on atp-platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from atp.scoring.models import ScoredTestResult  # noqa: F811
    from atp.statistics.models import TestRunStatistics  # noqa: F811
from pydantic.dataclasses import dataclass

from atp.loader.models import TestDefinition
from atp.protocol import ATPEvent, ATPResponse, ResponseStatus

# ---------------------------------------------------------------------------
# Evaluator result models
# ---------------------------------------------------------------------------


class EvalCheck(BaseModel):
    """Single evaluation check result."""

    name: str = Field(..., description="Check name", min_length=1)
    passed: bool = Field(..., description="Whether the check passed")
    score: float = Field(..., description="Score from 0.0 to 1.0", ge=0.0, le=1.0)
    message: str | None = Field(None, description="Human-readable message")
    details: dict[str, Any] | None = Field(None, description="Additional check details")


class EvalResult(BaseModel):
    """Result of an evaluator run containing multiple checks."""

    evaluator: str = Field(..., description="Evaluator name that produced this result")
    checks: list[EvalCheck] = Field(
        default_factory=list, description="List of check results"
    )

    @property
    def passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        """Calculate average score across all checks."""
        if not self.checks:
            return 1.0
        return sum(c.score for c in self.checks) / len(self.checks)

    @property
    def total_checks(self) -> int:
        """Return total number of checks."""
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        """Return number of passed checks."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_checks(self) -> int:
        """Return number of failed checks."""
        return sum(1 for c in self.checks if not c.passed)

    def add_check(self, check: EvalCheck) -> None:
        """Add a check to the result."""
        self.checks.append(check)

    def merge(self, other: EvalResult) -> EvalResult:
        """Merge another EvalResult into this one."""
        return EvalResult(
            evaluator=self.evaluator,
            checks=self.checks + other.checks,
        )

    @classmethod
    def aggregate(cls, results: list[EvalResult]) -> EvalResult:
        """Aggregate multiple EvalResults into one."""
        if not results:
            return cls(evaluator="aggregate", checks=[])
        all_checks: list[EvalCheck] = []
        for result in results:
            all_checks.extend(result.checks)
        return cls(evaluator="aggregate", checks=all_checks)


# ---------------------------------------------------------------------------
# Runner result models
# ---------------------------------------------------------------------------


class ProgressEventType(StrEnum):
    """Types of progress events."""

    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    TEST_FAILED = "test_failed"
    TEST_TIMEOUT = "test_timeout"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    SUITE_STARTED = "suite_started"
    SUITE_COMPLETED = "suite_completed"
    AGENT_EVENT = "agent_event"


class ProgressEvent(BaseModel):
    """Event emitted during test execution for progress tracking."""

    event_type: ProgressEventType = Field(..., description="Type of progress event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Event timestamp",
    )
    suite_name: str | None = Field(None, description="Suite name")
    test_id: str | None = Field(None, description="Test identifier")
    test_name: str | None = Field(None, description="Human-readable test name")
    run_number: int | None = Field(None, description="Current run number (1-indexed)")
    total_runs: int | None = Field(None, description="Total number of runs")
    total_tests: int | None = Field(None, description="Total number of tests")
    completed_tests: int | None = Field(None, description="Number of completed tests")
    success: bool | None = Field(None, description="Whether operation succeeded")
    error: str | None = Field(None, description="Error message if failed")
    agent_event: ATPEvent | None = Field(
        None, description="Original ATP event from agent"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific details",
    )


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class RunResult:
    """Result of a single test run (one execution)."""

    test_id: str
    run_number: int
    response: ATPResponse
    events: list[ATPEvent] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if run completed successfully."""
        return self.response.status == ResponseStatus.COMPLETED

    @property
    def duration_seconds(self) -> float | None:
        """Calculate run duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class TestResult:
    """Result of a test (possibly multiple runs)."""

    test: TestDefinition
    runs: list[RunResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if all runs completed successfully."""
        if not self.runs:
            return False
        return all(run.success for run in self.runs)

    @property
    def status(self) -> ResponseStatus:
        """Get aggregate status from all runs."""
        if not self.runs:
            return ResponseStatus.FAILED
        for run in self.runs:
            if run.response.status == ResponseStatus.TIMEOUT:
                return ResponseStatus.TIMEOUT
        if all(run.response.status == ResponseStatus.COMPLETED for run in self.runs):
            return ResponseStatus.COMPLETED
        completed = sum(
            1 for run in self.runs if run.response.status == ResponseStatus.COMPLETED
        )
        if completed > 0:
            return ResponseStatus.PARTIAL
        return ResponseStatus.FAILED

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def successful_runs(self) -> int:
        return sum(1 for run in self.runs if run.success)

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def get_run_durations(self) -> list[float]:
        return [
            run.duration_seconds
            for run in self.runs
            if run.duration_seconds is not None
        ]

    def get_run_steps(self) -> list[int]:
        return [
            run.response.metrics.total_steps
            for run in self.runs
            if run.response.metrics and run.response.metrics.total_steps is not None
        ]

    def get_run_tokens(self) -> list[int]:
        return [
            run.response.metrics.total_tokens
            for run in self.runs
            if run.response.metrics and run.response.metrics.total_tokens is not None
        ]

    def get_run_costs(self) -> list[float]:
        return [
            run.response.metrics.cost_usd
            for run in self.runs
            if run.response.metrics and run.response.metrics.cost_usd is not None
        ]


@dataclass
class SuiteResult:
    """Result of running a complete test suite."""

    suite_name: str
    agent_name: str
    tests: list[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    end_time: datetime | None = None
    error: str | None = None

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return sum(1 for t in self.tests if t.success)

    @property
    def failed_tests(self) -> int:
        return sum(1 for t in self.tests if not t.success)

    @property
    def success(self) -> bool:
        return self.total_tests > 0 and all(t.success for t in self.tests)

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_runs(self) -> int:
        return sum(t.total_runs for t in self.tests)

    @property
    def runs_per_test(self) -> int:
        if not self.tests:
            return 0
        return self.tests[0].total_runs if self.tests else 0


# ---------------------------------------------------------------------------
# Reporter result models
# ---------------------------------------------------------------------------


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
    ) -> SuiteReport:
        """Create a SuiteReport from a SuiteResult."""
        rebuild_report_models()
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


_report_models_rebuilt = False


def rebuild_report_models() -> None:
    """Rebuild TestReport/SuiteReport to resolve forward refs.

    Must be called before instantiating TestReport with scored_result
    or statistics fields. Called automatically by from_suite_result().
    """
    global _report_models_rebuilt
    if _report_models_rebuilt:
        return
    from atp.scoring.models import ScoredTestResult as _SR
    from atp.statistics.models import TestRunStatistics as _TS

    TestReport.model_rebuild(
        _types_namespace={"ScoredTestResult": _SR, "TestRunStatistics": _TS}
    )
    SuiteReport.model_rebuild()
    _report_models_rebuilt = True
