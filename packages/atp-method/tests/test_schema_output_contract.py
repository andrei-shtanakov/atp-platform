"""Schema support for output_contract, run_mode, and json_path grader config."""

import pytest
from pydantic import ValidationError

from atp_method.schema import AgentEvalCase, Grader, OutputContract


def _minimal_grader(**over) -> dict:
    base = dict(
        type="programmatic",
        checker="json_path",
        critical_check="x must hold",
        scoring="fail if critical fails",
        config={"assertions": [{"path": "$.a", "op": "equals", "expected": 1}]},
    )
    base.update(over)
    return base


def _minimal_case(**over) -> dict:
    base = dict(
        id="case-x-clean-001",
        version=1,
        family="x",
        status="active",
        suite_type="probe",
        capability="correctness",
        construction_axis="output_structure",
        axis_level="clean",
        instruction="do x",
        environment={"tools": ["none"], "side_effects": "none"},
        expected_failure_mode="fails x",
        grader=_minimal_grader(),
        provenance={"author": "a", "created": "2026-06-16"},
        output_contract={
            "artifact_name": "answer",
            "json_schema": {"type": "object"},
            "format_instruction": "return JSON",
        },
        run_mode="text_out",
    )
    base.update(over)
    return base


def test_output_contract_parses_and_aliases_schema() -> None:
    oc = OutputContract.model_validate(
        {"artifact_name": "answer", "schema": {"type": "object"}}
    )
    assert oc.artifact_name == "answer"
    assert oc.json_schema == {"type": "object"}
    assert oc.model_dump(by_alias=True)["schema"] == {"type": "object"}


def test_case_with_output_contract_and_run_mode_valid() -> None:
    case = AgentEvalCase.model_validate(_minimal_case())
    assert case.run_mode == "text_out"
    assert case.output_contract is not None


def test_run_mode_unwired_tier_rejected() -> None:
    with pytest.raises(ValidationError, match="run_mode"):
        AgentEvalCase.model_validate(_minimal_case(run_mode="workspace"))


def test_run_mode_defaults_to_text_out() -> None:
    doc = _minimal_case()
    del doc["run_mode"]
    assert AgentEvalCase.model_validate(doc).run_mode == "text_out"


def test_json_path_requires_assertions() -> None:
    with pytest.raises(ValidationError, match="assertions"):
        Grader.model_validate(_minimal_grader(config={"assertions": []}))
    with pytest.raises(ValidationError, match="assertions"):
        Grader.model_validate(_minimal_grader(config=None))
    with pytest.raises(ValidationError, match="assertions"):
        Grader.model_validate(_minimal_grader(config={"assertions": "oops"}))
