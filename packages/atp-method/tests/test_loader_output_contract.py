"""Loader threads output_contract into input_data + the critical assertion."""

from atp_method.loader import METHOD_CRITICAL_CHECK, case_to_test_definition
from atp_method.schema import AgentEvalCase


def _case() -> AgentEvalCase:
    return AgentEvalCase.model_validate(
        {
            "id": "case-x-clean-001",
            "version": 1,
            "family": "x",
            "status": "active",
            "suite_type": "probe",
            "capability": "correctness",
            "construction_axis": "output_structure",
            "axis_level": "clean",
            "instruction": "extract",
            "environment": {"tools": ["none"], "side_effects": "none"},
            "expected_failure_mode": "fabricates",
            "output_contract": {
                "artifact_name": "answer",
                "schema": {"type": "object", "required": ["requirements"]},
                "format_instruction": "Return ONLY JSON {requirements:[...]}",
            },
            "grader": {
                "type": "programmatic",
                "checker": "json_path",
                "critical_check": "no fabricated deadline",
                "scoring": "fail if critical fails",
                "config": {
                    "assertions": [
                        {
                            "path": "$.requirements[1].deadline",
                            "op": "equals",
                            "expected": None,
                        }
                    ]
                },
            },
            "provenance": {"author": "a", "created": "2026-06-16"},
        }
    )


def test_output_contract_goes_into_input_data() -> None:
    td = case_to_test_definition(_case())
    oc = td.task.input_data["output_contract"]
    assert oc["format_instruction"].startswith("Return ONLY JSON")
    assert oc["schema"]["required"] == ["requirements"]


def test_critical_assertion_carries_schema_and_assertions() -> None:
    td = case_to_test_definition(_case())
    crit = next(a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK)
    assert crit.config["checker"] == "json_path"
    assert crit.config["schema"]["required"] == ["requirements"]
    assert crit.config["assertions"][0]["path"] == "$.requirements[1].deadline"


def test_findings_match_case_gets_null_schema_empty_assertions() -> None:
    case = AgentEvalCase.model_validate(
        {
            "id": "case-y-clean-001",
            "version": 1,
            "family": "y",
            "status": "active",
            "suite_type": "probe",
            "capability": "safety_compliance",
            "construction_axis": "adversarial_environment",
            "axis_level": "clean",
            "instruction": "review",
            "environment": {"tools": ["none"], "side_effects": "none"},
            "expected_failure_mode": "misses",
            "grader": {
                "type": "programmatic",
                "checker": "findings_match",
                "critical_check": "find it",
                "scoring": "x",
                "expected_findings": [],
            },
            "provenance": {"author": "a", "created": "2026-06-16"},
        }
    )
    td = case_to_test_definition(case)
    crit = next(a for a in td.assertions if a.type == METHOD_CRITICAL_CHECK)
    assert crit.config["schema"] is None
    assert crit.config["assertions"] == []
    assert "output_contract" not in td.task.input_data
