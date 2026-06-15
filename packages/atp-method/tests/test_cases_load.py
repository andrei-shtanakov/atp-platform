"""The shipped code-review cases load + validate under the new grader shape."""

import json
from pathlib import Path

import jsonschema
import yaml

from atp_method.loader import load_case
from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CASES = sorted((ROOT / "method" / "cases" / "code-review").glob("*.yaml"))
SCHEMA = json.loads((ROOT / "method" / "agent-eval-case.schema.json").read_text())


def test_cases_present() -> None:
    # Full 5-level breakpoint sweep: clean / mild / moderate / severe / very_severe.
    assert len(CASES) == 5
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
