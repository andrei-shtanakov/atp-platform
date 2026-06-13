"""Tests for the agent-eval-case → TestDefinition loader."""

from pathlib import Path

import yaml

from atp_method.loader import (
    METHOD_CRITICAL_CHECK,
    METHOD_RUBRIC,
    case_to_test_definition,
    load_case,
)
from atp_method.schema import AgentEvalCase


def test_load_clean_case_structure(clean_case_path: Path) -> None:
    """The clean example maps to a structurally correct TestDefinition."""
    td = load_case(clean_case_path)

    assert td.id == "case-req-extraction-fabricated-deadline-clean-001"
    # instruction becomes the task the agent sees
    assert "Extract atomic requirements" in td.task.description
    # the failure mode becomes the test description
    assert "deadline" in (td.description or "").lower()


def test_critical_check_is_a_hard_gate(clean_case_path: Path) -> None:
    """grader.critical_check maps to a critical assertion (hard gate)."""
    td = load_case(clean_case_path)
    critical = [a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK]
    assert len(critical) == 1
    assert critical[0].critical is True
    assert critical[0].config["check"]
    assert critical[0].config["expected_failure_mode"]


def test_rubric_assertion_present_when_rubric_exists(clean_case_path: Path) -> None:
    """A rubric grader produces a method_rubric assertion carrying the rubric."""
    td = load_case(clean_case_path)
    rubric = [a for a in td.assertions if a.type == METHOD_RUBRIC]
    assert len(rubric) == 1
    assert rubric[0].config["rubric"]  # non-empty list of criteria


def test_governance_and_sweep_tags(clean_case_path: Path) -> None:
    """family/capability/axis/level/suite become namespaced tags for filtering."""
    td = load_case(clean_case_path)
    assert "family_req_extraction" in td.tags
    assert "capability_calibration" in td.tags
    assert "level_clean" in td.tags
    assert "suite_probe" in td.tags


def test_tools_mapped_to_allowed_tools(clean_case_path: Path) -> None:
    """environment.tools maps to constraints.allowed_tools."""
    td = load_case(clean_case_path)
    assert td.constraints.allowed_tools == ["file_read"]


def test_none_tools_becomes_empty_allow_list() -> None:
    """A text-only case ('none') maps to an empty allow-list."""
    doc = {
        "id": "case-demo-001",
        "version": 1,
        "family": "demo",
        "status": "active",
        "suite_type": "probe",
        "capability": "calibration",
        "construction_axis": "information_conditions",
        "axis_level": "clean",
        "instruction": "do the thing",
        "environment": {"tools": ["none"], "side_effects": "none"},
        "expected_failure_mode": "x",
        "grader": {
            "type": "programmatic",
            "critical_check": "c",
            "scoring": "s",
            "expected_findings": [],
        },
        "provenance": {"author": "t", "created": "2026-06-09"},
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(doc))
    assert td.constraints.allowed_tools == []


def test_no_rubric_emits_only_critical(example_cases_dir: Path) -> None:
    """A programmatic case without a rubric yields a single critical assertion."""
    # Build a no-rubric case from the clean example's dict.
    clean = yaml.safe_load(
        (
            example_cases_dir / "case-req-extraction-fabricated-deadline-clean-001.yaml"
        ).read_text()
    )
    clean["grader"] = {
        "type": "programmatic",
        "critical_check": "no fabricated value",
        "scoring": "fail if critical fails",
        "expected_findings": [],
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(clean))
    assert [a.type for a in td.assertions] == [METHOD_CRITICAL_CHECK]
