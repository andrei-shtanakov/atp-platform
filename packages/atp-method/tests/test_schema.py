"""Tests for the agent-eval-case schema model."""

from pathlib import Path

import pytest
import yaml

from atp_method.schema import AgentEvalCase


def test_all_example_cases_validate(example_cases_dir: Path) -> None:
    """Every shipped req-extraction example case parses and validates."""
    files = sorted(example_cases_dir.glob("*.yaml"))
    assert len(files) == 3
    for f in files:
        case = AgentEvalCase.model_validate(yaml.safe_load(f.read_text()))
        assert case.id.startswith("case-req-extraction-")
        assert case.family == "req-extraction"
        assert case.capability == "calibration"


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
