"""Tests for compact summary report models."""

from __future__ import annotations

from atp.core.results import EvalCheck, EvalResult
from atp.reporters.base import SuiteReport, TestReport
from atp.reporters.summary_models import CompactSuiteSummary, CompactTestSummary


def _citation_details(
    *,
    reason: str = "expected source policy-current.md, got archive/policy-2023.md",
    path: str = "$.requirements[0].citations.deadline.path",
    field: str = "path",
    expected_value: object = "policy-current.md",
    received_value: object = "archive/policy-2023.md",
) -> dict:
    return {
        "critical_pass": False,
        "malformed": False,
        "details": {
            "results": [
                {
                    "expected": {
                        "output_path": "$.requirements[0].citations.deadline",
                        "source_path": "policy-current.md",
                        "page": None,
                        "line_start": 14,
                        "line_end": 14,
                    },
                    "ok": False,
                    "reason": reason,
                    "path": path,
                    "field": field,
                    "expected_value": expected_value,
                    "received_value": received_value,
                }
            ],
            "n": 1,
        },
        "grader_version": "citation_grounding@1",
    }


def _failed_citation_test(test_id: str = "case-failed") -> TestReport:
    return TestReport(
        test_id=test_id,
        test_name="Failed citation case",
        success=False,
        score=0.0,
        duration_seconds=4.12,
        eval_results=[
            EvalResult(
                evaluator="agent_eval_case",
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=False,
                        score=0.0,
                        message="citation source mismatch",
                        details=_citation_details(),
                    )
                ],
            )
        ],
    )


def _passed_test() -> TestReport:
    return TestReport(
        test_id="case-passed",
        test_name="Passed case",
        success=True,
        score=100.0,
        duration_seconds=1.25,
    )


def test_compact_suite_summary_passing_suite_has_no_failures_and_success_true() -> None:
    report = SuiteReport(
        suite_name="req-extraction",
        agent_name="test-agent",
        total_tests=1,
        passed_tests=1,
        failed_tests=0,
        success_rate=1.0,
        duration_seconds=1.25,
        runs_per_test=1,
        tests=[_passed_test()],
    )

    summary = CompactSuiteSummary.from_report(report)

    assert summary.version == "compact-summary-v1"
    assert summary.success is True
    assert summary.total_tests == 1
    assert summary.passed_tests == 1
    assert summary.failed_tests == 0
    assert summary.malformed_tests == 0
    assert summary.errored_tests == 0
    assert summary.failures == []
    assert summary.passed is None
    assert summary.top_failure_reasons == []
    assert summary.truncated_failures == 0
    assert summary.error is None


def test_compact_test_summary_failed_test_includes_first_failure() -> None:
    summary = CompactTestSummary.from_test(_failed_citation_test())

    assert summary.test_id == "case-failed"
    assert summary.test_name == "Failed citation case"
    assert summary.status == "failed"
    assert summary.score == 0.0
    assert summary.duration_seconds == 4.12
    assert summary.failure is not None
    assert summary.failure.kind == "value_mismatch"
    assert summary.failure.path == "$.requirements[0].citations.deadline.path"


def test_compact_suite_summary_counts_malformed_and_error_breakdowns() -> None:
    malformed = TestReport(
        test_id="case-malformed",
        test_name="Malformed case",
        success=False,
        eval_results=[
            EvalResult(
                evaluator="agent_eval_case",
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=False,
                        score=0.0,
                        details={
                            "critical_pass": False,
                            "malformed": True,
                            "details": {"reason": "output is not valid JSON"},
                            "grader_version": "citation_grounding@1",
                        },
                    )
                ],
            )
        ],
    )
    errored = TestReport(
        test_id="case-error",
        test_name="Execution error case",
        success=False,
        error="adapter timed out",
    )
    report = SuiteReport(
        suite_name="req-extraction",
        agent_name="test-agent",
        total_tests=2,
        passed_tests=0,
        failed_tests=2,
        success_rate=0.0,
        tests=[malformed, errored],
    )

    summary = CompactSuiteSummary.from_report(report)

    assert summary.success is False
    assert summary.malformed_tests == 1
    assert summary.errored_tests == 1
    assert [failure.status for failure in summary.failures] == ["malformed", "error"]
    assert summary.top_failure_reasons[0].kind == "malformed_output"
    assert summary.top_failure_reasons[0].count == 1


def test_compact_suite_summary_include_passed_adds_passed_test_summaries() -> None:
    report = SuiteReport(
        suite_name="req-extraction",
        agent_name="test-agent",
        total_tests=2,
        passed_tests=1,
        failed_tests=1,
        success_rate=0.5,
        tests=[_passed_test(), _failed_citation_test()],
    )

    summary = CompactSuiteSummary.from_report(report, include_passed=True)

    assert summary.passed is not None
    assert [test.test_id for test in summary.passed] == ["case-passed"]
    assert [test.test_id for test in summary.failures] == ["case-failed"]


def test_compact_suite_summary_max_failures_truncates_failures() -> None:
    tests = [_failed_citation_test("case-1"), _failed_citation_test("case-2")]
    report = SuiteReport(
        suite_name="req-extraction",
        agent_name="test-agent",
        total_tests=2,
        passed_tests=0,
        failed_tests=2,
        success_rate=0.0,
        tests=tests,
    )

    summary = CompactSuiteSummary.from_report(report, max_failures=1)

    assert [failure.test_id for failure in summary.failures] == ["case-1"]
    assert summary.truncated_failures == 1
    assert summary.top_failure_reasons[0].kind == "value_mismatch"
    assert summary.top_failure_reasons[0].count == 2
