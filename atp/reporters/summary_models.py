"""Compact summary models for reporter output."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel

from atp.core.results import SuiteReport, TestReport

FailureKind = Literal[
    "execution_error",
    "malformed_output",
    "value_mismatch",
    "missing_value",
    "forbidden_value",
    "critical_check_failed",
    "scored_failure",
    "unknown_failure",
]


class CompactFailure(BaseModel):
    """One compact, actionable failure reason."""

    kind: FailureKind
    message: str
    evaluator: str | None = None
    check: str | None = None
    path: str | None = None
    expected: Any | None = None
    received: Any | None = None


class FailureReasonCount(BaseModel):
    """Count of compact failures by kind."""

    kind: str
    count: int


class CompactTestSummary(BaseModel):
    """Compact per-test summary."""

    test_id: str
    test_name: str
    status: Literal["passed", "failed", "malformed", "error"]
    score: float | None
    duration_seconds: float | None
    failure: CompactFailure | None = None

    @classmethod
    def from_test(cls, test: TestReport) -> CompactTestSummary:
        """Build a compact test summary from a detailed test report."""
        from atp.reporters.summary_extractor import CompactFailureExtractor

        failure = CompactFailureExtractor.extract(test)
        if test.error:
            status: Literal["passed", "failed", "malformed", "error"] = "error"
        elif failure is not None and failure.kind == "malformed_output":
            status = "malformed"
        elif test.success is False:
            status = "failed"
        else:
            status = "passed"

        return cls(
            test_id=test.test_id,
            test_name=test.test_name,
            status=status,
            score=test.score,
            duration_seconds=test.duration_seconds,
            failure=failure,
        )


class CompactSuiteSummary(BaseModel):
    """Compact suite summary derived from an existing suite report."""

    version: Literal["compact-summary-v1"] = "compact-summary-v1"
    suite_name: str
    agent_name: str
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    malformed_tests: int
    errored_tests: int
    success_rate: float
    duration_seconds: float | None
    runs_per_test: int
    failures: list[CompactTestSummary]
    passed: list[CompactTestSummary] | None = None
    top_failure_reasons: list[FailureReasonCount]
    truncated_failures: int = 0
    error: str | None = None

    @classmethod
    def from_report(
        cls,
        report: SuiteReport,
        *,
        include_passed: bool = False,
        max_failures: int | None = None,
    ) -> CompactSuiteSummary:
        """Build a compact suite summary from a detailed suite report."""
        tests = [CompactTestSummary.from_test(test) for test in report.tests]
        failure_tests = [test for test in tests if test.status != "passed"]
        reason_counts = Counter(
            test.failure.kind for test in failure_tests if test.failure is not None
        )
        if max_failures is not None and max_failures >= 0:
            visible_failures = failure_tests[:max_failures]
            truncated_failures = max(0, len(failure_tests) - len(visible_failures))
        else:
            visible_failures = failure_tests
            truncated_failures = 0

        passed_tests = [test for test in tests if test.status == "passed"]
        return cls(
            suite_name=report.suite_name,
            agent_name=report.agent_name,
            success=report.error is None and report.passed_tests == report.total_tests,
            total_tests=report.total_tests,
            passed_tests=report.passed_tests,
            failed_tests=report.failed_tests,
            malformed_tests=sum(1 for test in tests if test.status == "malformed"),
            errored_tests=sum(1 for test in tests if test.status == "error"),
            success_rate=report.success_rate,
            duration_seconds=report.duration_seconds,
            runs_per_test=report.runs_per_test,
            failures=visible_failures,
            passed=passed_tests if include_passed else None,
            top_failure_reasons=[
                FailureReasonCount(kind=kind, count=count)
                for kind, count in reason_counts.most_common()
            ],
            truncated_failures=truncated_failures,
            error=report.error,
        )
