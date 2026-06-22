"""Tests for the compact summary reporter."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from atp.core.results import EvalCheck, EvalResult
from atp.reporters.base import SuiteReport, TestReport
from atp.reporters.summary_reporter import SummaryReporter


def _citation_details(
    *,
    message: str = "expected source policy-current.md, got archive/policy-2023.md",
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
                    "reason": message,
                    "path": "$.requirements[0].citations.deadline.path",
                    "field": "path",
                    "expected_value": expected_value,
                    "received_value": received_value,
                }
            ],
            "n": 1,
        },
        "grader_version": "citation_grounding@1",
    }


def _passed_test() -> TestReport:
    return TestReport(
        test_id="case-passed",
        test_name="Passed case",
        success=True,
        score=100.0,
        duration_seconds=1.0,
    )


def _failed_test(details: dict | None = None) -> TestReport:
    return TestReport(
        test_id="case-failed",
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
                        details=details or _citation_details(),
                    )
                ],
            )
        ],
    )


def _suite(tests: list[TestReport]) -> SuiteReport:
    passed = sum(1 for test in tests if test.success)
    total = len(tests)
    return SuiteReport(
        suite_name="req-extraction",
        agent_name="test-agent",
        total_tests=total,
        passed_tests=passed,
        failed_tests=total - passed,
        success_rate=passed / total if total else 0.0,
        duration_seconds=5.12,
        runs_per_test=1,
        tests=tests,
    )


def test_summary_reporter_name_is_summary() -> None:
    reporter = SummaryReporter(output=StringIO(), use_colors=False)

    assert reporter.name == "summary"


def test_summary_reporter_console_passing_suite_shows_compact_counts() -> None:
    output = StringIO()
    reporter = SummaryReporter(output=output, use_colors=False)

    reporter.report(_suite([_passed_test()]))

    result = output.getvalue()
    assert "ATP Summary" in result
    assert "Result: PASSED" in result
    assert "Tests: 1 passed, 0 failed, 0 malformed, 0 error" in result
    assert "Failures:" not in result


def test_summary_reporter_console_failing_suite_shows_failures_only_by_default() -> (
    None
):
    output = StringIO()
    reporter = SummaryReporter(output=output, use_colors=False)

    reporter.report(_suite([_passed_test(), _failed_test()]))

    result = output.getvalue()
    assert "Result: FAILED" in result
    assert "case-failed" in result
    assert "case-passed" not in result


def test_summary_reporter_console_value_mismatch_includes_expected_received() -> None:
    output = StringIO()
    reporter = SummaryReporter(output=output, use_colors=False)

    reporter.report(_suite([_failed_test()]))

    result = output.getvalue()
    assert "reason: value_mismatch" in result
    assert "path: $.requirements[0].citations.deadline.path" in result
    assert "expected: policy-current.md" in result
    assert "received: archive/policy-2023.md" in result


def test_summary_reporter_json_outputs_compact_summary_shape() -> None:
    output = StringIO()
    reporter = SummaryReporter(output=output, format="json", indent=2)

    reporter.report(_suite([_failed_test()]))

    result = json.loads(output.getvalue())
    assert result["version"] == "compact-summary-v1"
    assert result["suite_name"] == "req-extraction"
    assert result["failures"][0]["test_id"] == "case-failed"
    assert result["failures"][0]["failure"]["kind"] == "value_mismatch"
    assert "summary" not in result
    assert "tests" not in result


def test_summary_reporter_file_output_creates_parent_directories(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "nested" / "summary.json"
    reporter = SummaryReporter(output_file=output_file, format="json")

    reporter.report(_suite([_passed_test()]))

    assert output_file.exists()
    result = json.loads(output_file.read_text())
    assert result["version"] == "compact-summary-v1"


def test_summary_reporter_include_passed_includes_passed_test_summaries() -> None:
    output = StringIO()
    reporter = SummaryReporter(
        output=output,
        format="json",
        include_passed=True,
    )

    reporter.report(_suite([_passed_test(), _failed_test()]))

    result = json.loads(output.getvalue())
    assert result["passed"] is not None
    assert [test["test_id"] for test in result["passed"]] == ["case-passed"]


def test_summary_reporter_large_values_are_bounded_or_omitted() -> None:
    output = StringIO()
    large_expected = "expected-" + ("x" * 5000)
    large_received = "received-" + ("y" * 5000)
    details = _citation_details(
        message="expected source policy-current.md, got archive/policy-2023.md",
        expected_value=large_expected,
        received_value=large_received,
    )
    reporter = SummaryReporter(output=output, format="json")

    reporter.report(_suite([_failed_test(details)]))

    result = output.getvalue()
    assert large_expected not in result
    assert large_received not in result
    assert "x" * 500 not in result
    assert "y" * 500 not in result
