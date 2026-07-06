"""The shipped code-review cases load + validate under the new grader shape."""

import json
from pathlib import Path

import jsonschema
import yaml

from atp_method.loader import load_case
from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CASES = sorted((ROOT / "method" / "cases" / "code-review").glob("*.yaml"))
REQ_CASES = sorted((ROOT / "method" / "cases" / "req-extraction").glob("*.yaml"))
SCHEMA = json.loads((ROOT / "method" / "agent-eval-case.schema.json").read_text())


def test_cases_present() -> None:
    # 5-level SQLi breakpoint sweep (clean..very_severe) + P2 correctness cases
    # (L1 off-by-one, L2 predicate, F1/F2/F3 FP-discipline, S1/S2 spec-violation,
    # D1/D2 cross-file + D3 single-file distractor).
    assert len(CASES) == 15
    axis_levels = {yaml.safe_load(p.read_text())["axis_level"] for p in CASES}
    assert axis_levels == {"clean", "mild", "moderate", "severe", "very_severe"}


def test_cases_validate_pydantic_and_contract() -> None:
    for path in CASES:
        doc = yaml.safe_load(path.read_text())
        # pydantic
        case = AgentEvalCase.model_validate(doc)
        assert case.grader.type == "programmatic"
        assert case.grader.checker == "findings_match"
        # JSON contract (the canonical cross-project schema)
        jsonschema.validate(doc, SCHEMA)
        # routing dimensions
        assert case.task_type == "review"
        assert case.language == "python"
        # loader path
        td = load_case(path)
        assert td.assertions


def test_req_extraction_cases_present() -> None:
    # 3 trap families (fabricated deadline/actor/condition) across a
    # clean..very_severe breakpoint axis, plus the read-only corpus ladder
    # (clean/moderate/severe/very_severe — Path A severity ladder).
    assert len(REQ_CASES) == 17
    axis_levels = {yaml.safe_load(p.read_text())["axis_level"] for p in REQ_CASES}
    assert axis_levels == {"clean", "mild", "moderate", "severe", "very_severe"}


def test_req_extraction_cases_are_deterministic() -> None:
    for path in REQ_CASES:
        doc = yaml.safe_load(path.read_text())
        case = AgentEvalCase.model_validate(doc)
        assert case.grader.type == "programmatic"
        assert case.task_type == "req-extraction"
        jsonschema.validate(doc, SCHEMA)
        # The one read-only corpus case grounds with citation_grounding; every
        # other (inline) case is a deterministic json_path check.
        expected = "citation_grounding" if "corpus" in path.name else "json_path"
        assert case.grader.checker == expected, path.name
