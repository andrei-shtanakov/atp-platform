"""Tests for the agent-eval-case schema model."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from atp_method.schema import AgentEvalCase, Grader


def test_all_example_cases_validate(example_cases_dir: Path) -> None:
    """Every shipped req-extraction example case parses and validates."""
    files = sorted(example_cases_dir.glob("*.yaml"))
    assert len(files) >= 1  # don't hard-code the count; new cases are expected
    for f in files:
        case = AgentEvalCase.model_validate(yaml.safe_load(f.read_text()))
        assert case.id.startswith("case-req-extraction-")
        assert case.family == "req-extraction"
        assert case.capability == "calibration"


def test_duplicate_tools_rejected() -> None:
    """environment.tools must be unique (schema uniqueItems)."""
    with pytest.raises(ValueError, match="duplicate"):
        AgentEvalCase.model_validate(_minimal(tools=["file_read", "file_read"]))


def test_invalid_created_date_rejected() -> None:
    """provenance.created must be an ISO date (schema format: date)."""
    doc = _minimal()
    doc["provenance"]["created"] = "June 9, 2026"
    with pytest.raises(ValueError, match="ISO date"):
        AgentEvalCase.model_validate(doc)


def test_invalid_tag_pattern_rejected() -> None:
    """Tags must match the controlled-vocabulary pattern."""
    doc = _minimal()
    doc["tags"] = ["Has-Caps"]
    with pytest.raises(ValueError, match="tags must match"):
        AgentEvalCase.model_validate(doc)


def test_duplicate_tags_rejected() -> None:
    """Tags must be unique (schema uniqueItems)."""
    doc = _minimal()
    doc["tags"] = ["dup", "dup"]
    with pytest.raises(ValueError, match="unique"):
        AgentEvalCase.model_validate(doc)


def test_none_tool_must_be_exclusive() -> None:
    """'none' cannot be combined with other tools."""
    with pytest.raises(ValueError, match="none"):
        AgentEvalCase.model_validate(_minimal(tools=["none", "file_read"]))


def test_rubric_grader_requires_rubric() -> None:
    """A rubric grader without a rubric is rejected."""
    doc = _minimal()
    doc["grader"] = {
        "type": "rubric",
        "critical_check": "x",
        "scoring": "y",
    }
    with pytest.raises(ValueError, match="rubric"):
        AgentEvalCase.model_validate(doc)


def test_volatility_requires_inject_turn() -> None:
    """requirements_volatility cases must include an inject turn."""
    doc = _minimal(construction_axis="requirements_volatility")
    with pytest.raises(ValueError, match="inject"):
        AgentEvalCase.model_validate(doc)


def test_extra_field_forbidden() -> None:
    """Unknown top-level fields are rejected (additionalProperties: false)."""
    doc = _minimal()
    doc["unexpected"] = True
    with pytest.raises(ValueError):
        AgentEvalCase.model_validate(doc)


def test_findings_match_grader_accepts_structured_ground_truth() -> None:
    g = Grader(
        type="programmatic",
        checker="findings_match",
        critical_check="flag the SQL injection",
        scoring="fail if critical_check fails",
        expected_findings=[
            {
                "rule_ids": ["SEC-011", "cwe-89"],
                "anchor": 'f"SELECT',
                "severity": "critical",
            }
        ],
        must_not_flag=[{"anchor": "cursor.execute(query, (user_id,))"}],
    )
    assert g.expected_findings is not None
    assert g.expected_findings[0].anchor == 'f"SELECT'
    assert g.must_not_flag is not None
    assert g.must_not_flag[0].anchor.startswith("cursor")


def test_findings_match_grader_requires_expected_findings_key() -> None:
    # missing expected_findings entirely under findings_match checker -> invalid
    with pytest.raises(ValidationError):
        Grader(
            type="programmatic",
            checker="findings_match",
            critical_check="x",
            scoring="y",
        )


def test_findings_match_grader_allows_empty_expected_findings() -> None:
    # a COMPLIANT case has no planted defect: expected_findings present but empty is OK
    g = Grader(
        type="programmatic",
        checker="findings_match",
        critical_check="must not invent issues",
        scoring="fail if critical_check fails",
        expected_findings=[],
        must_not_flag=[{"anchor": "cursor.execute(q, (uid,))"}],
    )
    assert g.expected_findings == []


def test_grader_type_findings_match_now_rejected() -> None:
    # findings_match is no longer a grader.type; it is a checker under programmatic
    with pytest.raises(ValidationError):
        Grader(
            type="findings_match",
            critical_check="x",
            scoring="y",
            expected_findings=[],
        )


def test_programmatic_checker_findings_requires_expected_findings() -> None:
    with pytest.raises(ValidationError, match="expected_findings"):
        Grader(
            type="programmatic",
            checker="findings_match",
            critical_check="x",
            scoring="y",
        )


def test_programmatic_checker_findings_accepts_empty_expected() -> None:
    g = Grader(
        type="programmatic",
        checker="findings_match",
        critical_check="x",
        scoring="y",
        expected_findings=[],
    )
    assert g.checker == "findings_match"


def test_checker_requires_programmatic_type() -> None:
    with pytest.raises(ValidationError, match="programmatic"):
        Grader(
            type="rubric",
            checker="findings_match",
            critical_check="x",
            scoring="y",
            rubric=[{"criterion": "c", "weight": 1.0}],
        )


def _minimal(
    *,
    tools: list[str] | None = None,
    construction_axis: str = "information_conditions",
) -> dict:
    """A minimal valid case dict, overridable for negative tests."""
    return {
        "id": "case-demo-001",
        "version": 1,
        "family": "demo",
        "status": "active",
        "suite_type": "probe",
        "capability": "calibration",
        "construction_axis": construction_axis,
        "axis_level": "clean",
        "instruction": "do the thing",
        "environment": {"tools": tools or ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "fabricates a value",
        "grader": {
            "type": "programmatic",
            "critical_check": "no fabricated value present",
            "scoring": "fail if critical_check fails",
        },
        "provenance": {"author": "test", "created": "2026-06-09"},
    }
