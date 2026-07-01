"""Failure extraction for compact summary reports."""

from __future__ import annotations

from typing import Any

from atp.core.results import EvalCheck, TestReport
from atp.reporters.summary_models import CompactFailure, FailureKind

MAX_MESSAGE_LENGTH = 500
MAX_STRING_VALUE_LENGTH = 300


class CompactFailureExtractor:
    """Extract the most actionable compact failure from a test report."""

    @staticmethod
    def extract(test: TestReport) -> CompactFailure | None:
        """Return the first compact failure for a test, if any."""
        if test.error:
            return CompactFailure(
                kind="execution_error",
                message=_truncate_string(test.error, MAX_MESSAGE_LENGTH),
            )

        failed_checks: list[tuple[str, EvalCheck]] = []
        for eval_result in test.eval_results:
            for check in eval_result.checks:
                if not check.passed:
                    failed_checks.append((eval_result.evaluator, check))

        for evaluator, check in failed_checks:
            failure = CitationGroundingFailureExtractor.extract(evaluator, check)
            if failure is not None and failure.kind == "malformed_output":
                return failure

        for evaluator, check in failed_checks:
            failure = CitationGroundingFailureExtractor.extract(evaluator, check)
            if failure is not None and failure.kind != "malformed_output":
                return failure

        if failed_checks:
            evaluator, check = failed_checks[0]
            return CompactFailure(
                kind="critical_check_failed",
                message=_generic_check_message(evaluator, check),
                evaluator=evaluator,
                check=check.name,
            )

        if test.success is False:
            return CompactFailure(
                kind="unknown_failure",
                message="test failed without failed checks",
            )
        return None


class CitationGroundingFailureExtractor:
    """Extract compact failures from citation_grounding@1 check details."""

    @staticmethod
    def extract(evaluator: str, check: EvalCheck) -> CompactFailure | None:
        """Return a compact citation-grounding failure when details are supported."""
        if evaluator != "agent_eval_case" or check.name != "critical_check":
            return None
        details = check.details
        if not isinstance(details, dict):
            return None
        if details.get("grader_version") != "citation_grounding@1":
            return None

        verdict_details = details.get("details")
        if not isinstance(verdict_details, dict):
            verdict_details = {}

        if details.get("malformed") is True:
            reason = (
                verdict_details.get("reason") or check.message or "malformed output"
            )
            return CompactFailure(
                kind="malformed_output",
                message=_truncate_string(str(reason), MAX_MESSAGE_LENGTH),
                evaluator=evaluator,
                check=check.name,
            )

        result = _first_failed_result(verdict_details.get("results"))
        if result is None:
            return CompactFailure(
                kind="critical_check_failed",
                message=_generic_check_message(evaluator, check),
                evaluator=evaluator,
                check=check.name,
            )

        reason = str(result.get("reason") or check.message or "critical check failed")
        if reason.startswith("expected source ") and " got " in reason:
            return _failure_from_result(
                kind="value_mismatch",
                message="citation source mismatch",
                evaluator=evaluator,
                check=check,
                result=result,
            )
        if reason == "citation page does not match expected page":
            return _failure_from_result(
                kind="value_mismatch",
                message=reason,
                evaluator=evaluator,
                check=check,
                result=result,
            )
        if reason == "citation line range does not match expected range":
            return _failure_from_result(
                kind="value_mismatch",
                message=reason,
                evaluator=evaluator,
                check=check,
                result=result,
            )
        if reason.startswith("output_path not found: "):
            return _failure_from_result(
                kind="missing_value",
                message=reason,
                evaluator=evaluator,
                check=check,
                result=result,
            )
        if reason.startswith("forbidden source cited: "):
            source_path = reason.removeprefix("forbidden source cited: ")
            return CompactFailure(
                kind="forbidden_value",
                message=reason,
                evaluator=evaluator,
                check=check.name,
                path="$.**.path",
                expected=_bound_value(f"not {source_path}"),
                received=_bound_value(source_path),
            )

        return CompactFailure(
            kind="critical_check_failed",
            message=_truncate_string(reason, MAX_MESSAGE_LENGTH),
            evaluator=evaluator,
            check=check.name,
        )


def _first_failed_result(results: Any) -> dict[str, Any] | None:
    if not isinstance(results, list):
        return None
    for result in results:
        if isinstance(result, dict) and result.get("ok") is False:
            return result
    return None


def _failure_from_result(
    *,
    kind: FailureKind,
    message: str,
    evaluator: str,
    check: EvalCheck,
    result: dict[str, Any],
) -> CompactFailure:
    return CompactFailure(
        kind=kind,
        message=_truncate_string(message, MAX_MESSAGE_LENGTH),
        evaluator=evaluator,
        check=check.name,
        path=_optional_string(result.get("path")),
        expected=_bound_value(result.get("expected_value")),
        received=_bound_value(result.get("received_value")),
    )


def _message_from_check(check: EvalCheck) -> str:
    return _truncate_string(check.message or "check failed", MAX_MESSAGE_LENGTH)


def _generic_check_message(evaluator: str, check: EvalCheck) -> str:
    return _truncate_string(
        f"{evaluator}:{check.name} failed",
        MAX_MESSAGE_LENGTH,
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _bound_value(value: Any) -> Any | None:
    if isinstance(value, str):
        return _truncate_string(value, MAX_STRING_VALUE_LENGTH)
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, list):
        if len(value) > 20:
            return None
        return [_bound_value(item) for item in value]
    if isinstance(value, dict):
        if len(value) > 20:
            return None
        return {
            str(key): _bound_value(item)
            for key, item in value.items()
            if len(str(key)) <= 80
        }
    return _truncate_string(str(value), MAX_STRING_VALUE_LENGTH)


def _truncate_string(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."
