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
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(clean))
    assert [a.type for a in td.assertions] == [METHOD_CRITICAL_CHECK]


def test_loader_threads_checker_into_critical_config() -> None:
    """grader.checker is threaded into the critical-check assertion config."""
    doc = {
        "id": "case-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "review",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses it",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [],
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        },
        "provenance": {"author": "a", "created": "2026-06-14"},
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(doc))
    crit = next(a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK)
    assert crit.config["checker"] == "findings_match"


def test_loader_appends_behavior_assertions_as_normal_assertions() -> None:
    """Top-level behavior_assertions become ATP behavior assertions."""
    behavior_config = {
        "expected_tool_calls": [
            {
                "tool": "file_read",
                "status": "success",
                "input_matches": [{"path": "$.path", "equals": "policy-current.md"}],
            }
        ],
        "forbidden_tool_calls": [
            {
                "tool": "file_read",
                "input_matches": [
                    {"path": "$.path", "equals": "archive/policy-2023.md"}
                ],
            }
        ],
    }
    doc = {
        "id": "case-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "review",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses it",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [],
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        },
        "behavior_assertions": [
            {"type": "behavior", "critical": True, "config": behavior_config}
        ],
        "provenance": {"author": "a", "created": "2026-06-14"},
    }

    td = case_to_test_definition(AgentEvalCase.model_validate(doc))

    assert [assertion.type for assertion in td.assertions] == [
        METHOD_CRITICAL_CHECK,
        "behavior",
    ]
    behavior = td.assertions[1]
    assert behavior.critical is True
    assert behavior.config == behavior_config


def test_loader_preserves_reserved_critical_config_values() -> None:
    """grader.config cannot override critical-check fields derived from the case."""
    expected_schema = {
        "type": "object",
        "properties": {"findings": {"type": "array"}},
    }
    expected_findings = [
        {"rule_ids": ["deadline"], "anchor": "deadline", "severity": "critical"}
    ]
    doc = {
        "id": "case-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "review",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses the planted deadline",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "gold": "top-level-gold",
            "expected_findings": expected_findings,
            "critical_check": "flag the planted deadline",
            "scoring": "fail if critical fails",
            "config": {
                "checker": "json_path",
                "gold": "config-gold",
                "expected_findings": [],
                "schema": {"type": "string"},
            },
        },
        "output_contract": {
            "artifact_name": "findings",
            "schema": expected_schema,
        },
        "provenance": {"author": "a", "created": "2026-06-14"},
    }

    td = case_to_test_definition(AgentEvalCase.model_validate(doc))

    crit = next(a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK)
    reserved_config = {
        "checker": crit.config["checker"],
        "gold": crit.config["gold"],
        "expected_findings": crit.config["expected_findings"],
        "schema": crit.config["schema"],
    }
    assert reserved_config == {
        "checker": "findings_match",
        "gold": "top-level-gold",
        "expected_findings": expected_findings,
        "schema": expected_schema,
    }


def test_tags_include_case_version() -> None:
    """The case version is emitted as a version_<n> tag for SP-1 persistence."""
    doc = {
        "id": "case-1",
        "version": 3,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "x",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "m",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [],
            "critical_check": "c",
            "scoring": "s",
        },
        "provenance": {"author": "a", "created": "2026-06-14"},
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(doc))
    assert "version_3" in td.tags


def test_tags_include_task_type_and_language_when_set() -> None:
    from atp_method.loader import case_to_test_definition
    from atp_method.schema import AgentEvalCase

    doc = {  # a valid case WITH the two fields
        "id": "case-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "task_type": "review",
        "language": "python",
        "instruction": "x",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "m",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [],
            "critical_check": "c",
            "scoring": "s",
        },
        "provenance": {"author": "a", "created": "2026-06-15"},
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(doc))
    assert "task_type_review" in td.tags
    assert "language_python" in td.tags


def test_tags_omit_task_type_language_when_absent() -> None:
    from atp_method.loader import case_to_test_definition
    from atp_method.schema import AgentEvalCase

    doc = {  # same case WITHOUT task_type/language
        "id": "case-2",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "x",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "m",
        "grader": {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [],
            "critical_check": "c",
            "scoring": "s",
        },
        "provenance": {"author": "a", "created": "2026-06-15"},
    }
    td = case_to_test_definition(AgentEvalCase.model_validate(doc))
    assert not any(t.startswith("task_type_") for t in td.tags)
    assert not any(t.startswith("language_") for t in td.tags)
