import json as json_mod
from pathlib import Path

from atp_method.envelopes import build_prompt, get_envelope
from atp_method.loader import load_case

_CASES = sorted(
    (Path(__file__).resolve().parents[3] / "method" / "cases" / "code-review").glob(
        "*.yaml"
    )
)
assert len(_CASES) == 15, f"expected 15 code-review cases, got {len(_CASES)}"


def test_every_code_review_case_declares_object_output_contract() -> None:
    import yaml

    from atp_method.schema import AgentEvalCase

    for path in _CASES:
        case = AgentEvalCase.model_validate(yaml.safe_load(path.read_text()))
        assert case.output_contract is not None, path.name
        schema = case.output_contract.json_schema
        assert schema.get("type") == "object", path.name
        assert "findings" in schema.get("required", []), path.name
        assert case.output_contract.format_instruction, path.name


def test_code_review_prompt_uses_object_format_instruction() -> None:
    # With output_contract present, build_prompt switches to the generic
    # envelope + the case's object format_instruction (not the array envelope).
    td = load_case(_CASES[0])
    request = {
        "task": {
            "description": td.task.description,
            "input_data": td.task.input_data,
        }
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert '"findings"' in prompt
    assert "JSON array of findings" not in prompt  # the old review envelope text


def test_object_output_grades_through_case_schema() -> None:
    import yaml
    from atp.evaluators.findings.matcher import grade_findings

    from atp_method.schema import AgentEvalCase

    # Pick the moderate SQLi case (one seeded SEC-011 defect).
    path = next(p for p in _CASES if "sqli-moderate" in p.name)
    case = AgentEvalCase.model_validate(yaml.safe_load(path.read_text()))
    assert case.output_contract is not None
    schema = case.output_contract.json_schema
    expected = [f.model_dump() for f in (case.grader.expected_findings or [])]
    must_not = [m.model_dump() for m in (case.grader.must_not_flag or [])]

    anchor = expected[0]["anchor"]
    good = json_mod.dumps(
        {"findings": [{"rule_id": "SEC-011", "anchor": anchor, "severity": "critical"}]}
    )
    r_ok = grade_findings(good, expected, must_not, schema=schema)
    assert r_ok.malformed is False
    assert r_ok.critical_pass is True

    # The pre-migration bare-array form now violates the object contract.
    bad = json_mod.dumps(
        [{"rule_id": "SEC-011", "anchor": anchor, "severity": "critical"}]
    )
    r_bad = grade_findings(bad, expected, must_not, schema=schema)
    assert r_bad.malformed is True
