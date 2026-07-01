"""Tests for compact failure extraction."""

from __future__ import annotations

import pytest

from atp.core.results import EvalCheck, EvalResult
from atp.reporters.base import TestReport
from atp.reporters.summary_extractor import (
    CitationGroundingFailureExtractor,
    CompactFailureExtractor,
)


def _citation_check(
    *,
    reason: str,
    path: str | None = None,
    field: str | None = None,
    expected_value: object | None = None,
    received_value: object | None = None,
    expected: dict | None = None,
    malformed: bool = False,
) -> EvalCheck:
    details: dict = {
        "critical_pass": False,
        "malformed": malformed,
        "details": {"reason": reason} if malformed else {},
        "grader_version": "citation_grounding@1",
    }
    if not malformed:
        result = {
            "expected": expected
            or {
                "output_path": "$.requirements[0].citations.deadline",
                "source_path": "policy-current.md",
                "page": None,
                "line_start": 14,
                "line_end": 14,
            },
            "ok": False,
            "reason": reason,
        }
        if path is not None:
            result["path"] = path
        if field is not None:
            result["field"] = field
        if expected_value is not None:
            result["expected_value"] = expected_value
        if received_value is not None:
            result["received_value"] = received_value
        details["details"] = {"results": [result], "n": 1}
    return EvalCheck(
        name="critical_check",
        passed=False,
        score=0.0,
        message=reason,
        details=details,
    )


@pytest.mark.parametrize(
    ("check", "path", "expected", "received"),
    [
        (
            _citation_check(
                reason="expected source policy-current.md, got archive/policy-2023.md",
                path="$.requirements[0].citations.deadline.path",
                field="path",
                expected_value="policy-current.md",
                received_value="archive/policy-2023.md",
            ),
            "$.requirements[0].citations.deadline.path",
            "policy-current.md",
            "archive/policy-2023.md",
        ),
        (
            _citation_check(
                reason="citation page does not match expected page",
                path="$.requirements[0].citations.deadline.page",
                field="page",
                expected_value=2,
                received_value=None,
            ),
            "$.requirements[0].citations.deadline.page",
            2,
            None,
        ),
        (
            _citation_check(
                reason="citation line range does not match expected range",
                path="$.requirements[0].citations.deadline.line_range",
                field="line_range",
                expected_value={"line_start": 14, "line_end": 14},
                received_value={"line_start": 13, "line_end": 15},
            ),
            "$.requirements[0].citations.deadline.line_range",
            {"line_start": 14, "line_end": 14},
            {"line_start": 13, "line_end": 15},
        ),
    ],
)
def test_citation_extractor_value_mismatches_map_expected_received(
    check: EvalCheck,
    path: str,
    expected: object,
    received: object,
) -> None:
    failure = CitationGroundingFailureExtractor.extract("agent_eval_case", check)

    assert failure is not None
    assert failure.kind == "value_mismatch"
    assert failure.evaluator == "agent_eval_case"
    assert failure.check == "critical_check"
    assert failure.path == path
    assert failure.expected == expected
    assert failure.received == received


def test_citation_extractor_missing_output_path_maps_missing_value() -> None:
    check = _citation_check(
        reason="output_path not found: $.requirements[0].citations.deadline",
        path="$.requirements[0].citations.deadline",
        field="deadline",
        expected_value="citation object",
        received_value="missing",
    )

    failure = CitationGroundingFailureExtractor.extract("agent_eval_case", check)

    assert failure is not None
    assert failure.kind == "missing_value"
    assert failure.path == "$.requirements[0].citations.deadline"
    assert failure.expected == "citation object"
    assert failure.received == "missing"


def test_citation_extractor_forbidden_source_maps_forbidden_value() -> None:
    check = _citation_check(
        reason="forbidden source cited: archive/policy-2023.md",
        expected={"source_path": "archive/policy-2023.md", "status": "obsolete"},
    )

    failure = CitationGroundingFailureExtractor.extract("agent_eval_case", check)

    assert failure is not None
    assert failure.kind == "forbidden_value"
    assert failure.path == "$.**.path"
    assert failure.expected == "not archive/policy-2023.md"
    assert failure.received == "archive/policy-2023.md"


def test_citation_extractor_malformed_verdict_maps_malformed_output() -> None:
    check = _citation_check(reason="output is not valid JSON", malformed=True)

    failure = CitationGroundingFailureExtractor.extract("agent_eval_case", check)

    assert failure is not None
    assert failure.kind == "malformed_output"
    assert failure.message == "output is not valid JSON"
    assert failure.path is None
    assert failure.expected is None
    assert failure.received is None


def test_citation_extractor_unsupported_check_falls_back_to_concise_failure() -> None:
    check = _citation_check(reason="metadata mismatch for policy-current.md: status")

    failure = CitationGroundingFailureExtractor.extract("agent_eval_case", check)

    assert failure is not None
    assert failure.kind == "critical_check_failed"
    assert failure.message == "metadata mismatch for policy-current.md: status"
    assert failure.evaluator == "agent_eval_case"
    assert failure.check == "critical_check"
    assert failure.path is None
    assert failure.expected is None
    assert failure.received is None


def test_compact_failure_extractor_execution_error_wins_over_failed_checks() -> None:
    test = TestReport(
        test_id="case-error",
        test_name="Execution error",
        success=False,
        error="adapter timed out",
        eval_results=[
            EvalResult(
                evaluator="agent_eval_case",
                checks=[
                    _citation_check(
                        reason="expected source policy-current.md, got other.md",
                        path="$.requirements[0].citations.deadline.path",
                        expected_value="policy-current.md",
                        received_value="other.md",
                    )
                ],
            )
        ],
    )

    failure = CompactFailureExtractor.extract(test)

    assert failure is not None
    assert failure.kind == "execution_error"
    assert failure.message == "adapter timed out"
    assert failure.evaluator is None
    assert failure.check is None


def test_compact_failure_extractor_unsupported_check_uses_generic_message() -> None:
    test = TestReport(
        test_id="case-failed",
        test_name="Failed case",
        success=False,
        eval_results=[
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(
                        name="file_exists",
                        passed=False,
                        score=0.0,
                        message="raw model output: SECRET_PROMPT_TEXT",
                    )
                ],
            )
        ],
    )

    failure = CompactFailureExtractor.extract(test)

    assert failure is not None
    assert failure.kind == "critical_check_failed"
    assert failure.message == "artifact:file_exists failed"
    assert "SECRET_PROMPT_TEXT" not in failure.message
    assert failure.evaluator == "artifact"
    assert failure.check == "file_exists"
