"""Tests for the report_benchmark reporter (report_benchmark-v1 conformance)."""

import json
from pathlib import Path

import jsonschema
import pytest

from atp.reporters.benchmark_reporter import build_report_benchmark_payload

SCHEMA = json.loads(Path("method/contract/report_benchmark-v1.schema.json").read_text())


def _case_result(
    case_id: str,
    axis_level: str,
    critical_pass: bool,
    rubric: float,
    malformed: bool = False,
):
    return {
        "case_id": case_id,
        "axis_level": axis_level,
        "critical_pass": critical_pass,
        "rubric_score": rubric,
        "malformed": malformed,
        "tokens": 920,
        "cost_usd": 0.0123,
        "duration_seconds": 4.2,
        "error_class": None,
    }


def test_payload_conforms_to_contract_and_aggregates() -> None:
    results = [
        _case_result("clean-001", "clean", True, 0.9),
        _case_result("moderate-001", "moderate", False, 0.4),
    ]
    payload = build_report_benchmark_payload(
        run_id="run-abc",
        benchmark_id="code-review",
        agent_id="claude_code",
        ts="2026-06-13T10:00:00Z",
        case_results=results,
    )
    jsonschema.validate(payload, SCHEMA)  # raises on any contract drift
    assert payload["benchmark_id"] == "code-review"
    assert payload["agent_id"] == "claude_code"
    assert payload["score"] == 0.5
    assert payload["score_components"]["critical_pass_rate"] == 0.5
    assert payload["score_components"]["malformed_rate"] == 0.0
    assert payload["breakpoint_axis_level"] == "moderate"
    assert payload["per_task_total_count"] == 2
    assert payload["total_tokens"] == 1840


def test_malformed_rate_aggregated_and_contract_conformant() -> None:
    # A malformed case counts toward malformed_rate AND fails critical_pass; it is
    # not double-counted as a missed defect, and the payload stays contract-valid.
    results = [
        _case_result("clean-001", "clean", True, 0.9),
        _case_result("mild-001", "mild", False, 0.0, malformed=True),
        _case_result("moderate-001", "moderate", False, 0.3),
    ]
    payload = build_report_benchmark_payload(
        run_id="run-mf",
        benchmark_id="code-review",
        agent_id="claude_code",
        ts="2026-06-14T10:00:00Z",
        case_results=results,
    )
    jsonschema.validate(payload, SCHEMA)
    assert payload["score_components"]["malformed_rate"] == round(1 / 3, 6)
    assert payload["score_components"]["critical_pass_rate"] == round(1 / 3, 6)


def test_malformed_rate_defaults_zero_for_legacy_case_dicts() -> None:
    # Case dicts predating P3 carry no "malformed" key; the reporter defaults it.
    legacy = {
        "case_id": "c",
        "axis_level": "clean",
        "critical_pass": True,
        "rubric_score": 0.8,
        "tokens": 10,
        "cost_usd": 0.0,
        "duration_seconds": 1.0,
        "error_class": None,
    }
    payload = build_report_benchmark_payload(
        run_id="run-legacy",
        benchmark_id="code-review",
        agent_id="claude_code",
        ts="2026-06-14T10:00:00Z",
        case_results=[legacy],
    )
    assert payload["score_components"]["malformed_rate"] == 0.0


@pytest.mark.parametrize(
    ("raw_error_class", "expected_error_class"),
    [
        ("failed", "test_failure"),
        ("no_run", "test_failure"),
        ("timeout", "timeout"),
        ("cancelled", "other"),
        ("partial", "other"),
        ("unexpected_status", "other"),
        (None, None),
    ],
)
def test_payload_normalizes_per_task_error_class_to_contract_enum(
    raw_error_class: str | None, expected_error_class: str | None
) -> None:
    case = _case_result("moderate-001", "moderate", False, 0.0)
    case["error_class"] = raw_error_class

    payload = build_report_benchmark_payload(
        run_id=f"run-error-{raw_error_class or 'none'}",
        benchmark_id="code-review",
        agent_id="claude_code",
        ts="2026-06-14T10:00:00Z",
        case_results=[case],
    )

    assert payload["per_task"][0]["error_class"] == expected_error_class
    jsonschema.validate(payload, SCHEMA)
