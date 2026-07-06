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


def test_mean_run_pass_rate_surfaces_flake_and_per_task_runs() -> None:
    # Two cases both "pass" (majority), but one is a flaky 2/3. critical_pass_rate
    # is 1.0 (binary); mean_run_pass_rate exposes the flake as (1.0 + 2/3)/2.
    solid = _case_result("clean-001", "clean", True, 0.0)
    solid["runs_graded"] = 3
    solid["run_pass_count"] = 3
    flaky = _case_result("severe-001", "severe", True, 0.0)
    flaky["runs_graded"] = 3
    flaky["run_pass_count"] = 2
    payload = build_report_benchmark_payload(
        run_id="r",
        benchmark_id="req-extraction",
        agent_id="pi@gpt-5",
        ts="2026-07-06T10:00:00Z",
        case_results=[solid, flaky],
    )
    jsonschema.validate(payload, SCHEMA)
    assert payload["score_components"]["critical_pass_rate"] == 1.0
    assert payload["score_components"]["mean_run_pass_rate"] == round(
        (1.0 + 2 / 3) / 2, 6
    )
    by_id = {t["task_index"]: t for t in payload["per_task"]}
    assert by_id[1]["run_pass_count"] == 2
    assert by_id[1]["runs_graded"] == 3


def test_mean_run_pass_rate_legacy_dict_equals_binary() -> None:
    # A case dict without runs_graded (runs=1 / pre-#232) degrades to binary:
    # mean_run_pass_rate == critical_pass_rate, and per_task runs are 1/1 or 0/1.
    payload = build_report_benchmark_payload(
        run_id="r",
        benchmark_id="code-review",
        agent_id="a",
        ts="2026-06-13T10:00:00Z",
        case_results=[
            _case_result("a", "clean", True, 0.0),
            _case_result("b", "moderate", False, 0.0),
        ],
    )
    assert payload["score_components"]["mean_run_pass_rate"] == 0.5
    assert payload["score_components"]["critical_pass_rate"] == 0.5
    by_id = {t["task_index"]: t for t in payload["per_task"]}
    assert (by_id[0]["run_pass_count"], by_id[0]["runs_graded"]) == (1, 1)
    assert (by_id[1]["run_pass_count"], by_id[1]["runs_graded"]) == (0, 1)


def test_mean_run_pass_rate_counts_all_infra_case_as_zero() -> None:
    # An all-infra case emits runs_graded=0 (from _grade_case). It must be
    # PRESERVED as (0, 0) in per_task — not degraded to legacy (0, 1) — and
    # count as a 0.0 fraction in mean_run_pass_rate (matching how
    # critical_pass_rate treats it as a fail), never dividing by zero.
    good = _case_result("clean-001", "clean", True, 0.0)
    good["runs_graded"] = 3
    good["run_pass_count"] = 3
    infra = _case_result("severe-001", "severe", False, 0.0)
    infra["runs_graded"] = 0
    infra["run_pass_count"] = 0
    infra["error_class"] = "timeout"
    payload = build_report_benchmark_payload(
        run_id="r",
        benchmark_id="req-extraction",
        agent_id="a",
        ts="2026-07-06T10:00:00Z",
        case_results=[good, infra],
    )
    jsonschema.validate(payload, SCHEMA)
    # (1.0 + 0.0) / 2 — the infra case drags it down, not excluded.
    assert payload["score_components"]["mean_run_pass_rate"] == 0.5
    by_id = {t["task_index"]: t for t in payload["per_task"]}
    assert (by_id[1]["run_pass_count"], by_id[1]["runs_graded"]) == (0, 0)


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
