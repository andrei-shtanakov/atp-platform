"""Tests for the report_benchmark reporter (report_benchmark-v1 conformance)."""

import json
from pathlib import Path

import jsonschema

from atp.reporters.benchmark_reporter import build_report_benchmark_payload

SCHEMA = json.loads(Path("method/contract/report_benchmark-v1.schema.json").read_text())


def _case_result(case_id: str, axis_level: str, critical_pass: bool, rubric: float):
    return {
        "case_id": case_id,
        "axis_level": axis_level,
        "critical_pass": critical_pass,
        "rubric_score": rubric,
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
    assert payload["breakpoint_axis_level"] == "moderate"
    assert payload["per_task_total_count"] == 2
    assert payload["total_tokens"] == 1840
