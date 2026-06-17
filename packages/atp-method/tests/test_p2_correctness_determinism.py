import json
from pathlib import Path

import yaml
from atp.evaluators.checkers import get_checker

from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CHECKER = get_checker("findings_match")
assert CHECKER is not None


def _cfg(case_path: Path) -> dict:
    case = AgentEvalCase.model_validate(yaml.safe_load(case_path.read_text()))
    return {
        "expected_findings": [
            f.model_dump() for f in (case.grader.expected_findings or [])
        ],
        "must_not_flag": [m.model_dump() for m in (case.grader.must_not_flag or [])],
    }


L1 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-logic-offbyone-001.yaml"
)


def test_l1_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "LOG-001",
                "anchor": "return events[-(n - 1):]",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(L1), good)
    assert v.critical_pass is True and v.malformed is False


def test_l1_miss_fails() -> None:
    v = CHECKER(_cfg(L1), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_l1_near_miss_anchor_not_accepted() -> None:
    # flagging the *safe* return line must NOT satisfy the expected finding
    near = json.dumps(
        [{"rule_id": "LOG-001", "anchor": "return events", "severity": "critical"}]
    )
    v = CHECKER(_cfg(L1), near)
    assert v.critical_pass is False


def test_l1_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(L1),
        json.dumps(
            [
                {
                    "rule_id": "LOG-001",
                    "anchor": "return events[-(n - 1):]",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False
