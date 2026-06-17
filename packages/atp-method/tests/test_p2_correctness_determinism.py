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


# --- L2: inverted predicate (defect case, same 4 checks as L1) -----------------

L2 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-logic-predicate-001.yaml"
)


def test_l2_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "LOG-002",
                "anchor": "return user.is_member or user.is_banned",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(L2), good)
    assert v.critical_pass is True and v.malformed is False


def test_l2_miss_fails() -> None:
    v = CHECKER(_cfg(L2), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_l2_near_miss_anchor_not_accepted() -> None:
    # flagging the benign def line must NOT satisfy the expected predicate finding
    near = json.dumps(
        [
            {
                "rule_id": "LOG-002",
                "anchor": "+5: def can_view(user, doc):",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(L2), near)
    assert v.critical_pass is False


def test_l2_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(L2),
        json.dumps(
            [
                {
                    "rule_id": "LOG-002",
                    "anchor": "return user.is_member or user.is_banned",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False


# --- F1: looks-like-SQLi but SAFE (FP-discipline, expected_findings=[]) --------

F1 = (
    ROOT / "method/cases/code-review/case-code-review-correctness-fp-sqli-safe-001.yaml"
)


def test_f1_good_no_findings_passes() -> None:
    v = CHECKER(_cfg(F1), json.dumps([]))
    assert v.critical_pass is True and v.malformed is False


def test_f1_bad_flags_must_not_flag_fails() -> None:
    bad = json.dumps(
        [
            {
                "rule_id": "SEC-011",
                "anchor": 'q = f"SELECT * FROM orders WHERE status',
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(F1), bad)
    assert v.critical_pass is False and v.malformed is False


def test_f1_precision_two_false_positives() -> None:
    two = json.dumps(
        [
            {
                "rule_id": "SEC-011",
                "anchor": 'q = f"SELECT * FROM orders WHERE status',
                "severity": "critical",
            },
            {
                "rule_id": "SEC-011",
                "anchor": "OrderStatus.ACTIVE.value",
                "severity": "major",
            },
        ]
    )
    v = CHECKER(_cfg(F1), two)
    assert v.fp_count == 2


def test_f1_wellformed_empty_not_malformed() -> None:
    v = CHECKER(_cfg(F1), json.dumps([]))
    assert v.malformed is False


# --- F2: looks-unsafe but GUARDED/trusted (FP-discipline, []) ------------------

F2 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-fp-deser-trusted-001.yaml"
)


def test_f2_good_no_findings_passes() -> None:
    v = CHECKER(_cfg(F2), json.dumps([]))
    assert v.critical_pass is True and v.malformed is False


def test_f2_bad_flags_must_not_flag_fails() -> None:
    bad = json.dumps(
        [
            {
                "rule_id": "SEC-020",
                "anchor": "obj = pickle.loads(raw)",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(F2), bad)
    assert v.critical_pass is False and v.malformed is False


def test_f2_precision_two_false_positives() -> None:
    two = json.dumps(
        [
            {
                "rule_id": "SEC-020",
                "anchor": "obj = pickle.loads(raw)",
                "severity": "critical",
            },
            {
                "rule_id": "SEC-020",
                "anchor": "raw = _MEMO[key]",
                "severity": "major",
            },
        ]
    )
    v = CHECKER(_cfg(F2), two)
    assert v.fp_count == 2


def test_f2_wellformed_empty_not_malformed() -> None:
    v = CHECKER(_cfg(F2), json.dumps([]))
    assert v.malformed is False


# --- F3: looks-like-a-bug but CORRECT per intent (FP-discipline, []) -----------

F3 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-fp-intent-correct-001.yaml"
)


def test_f3_good_no_findings_passes() -> None:
    v = CHECKER(_cfg(F3), json.dumps([]))
    assert v.critical_pass is True and v.malformed is False


def test_f3_bad_flags_must_not_flag_fails() -> None:
    bad = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": "share = total_cents // n",
                "severity": "major",
            }
        ]
    )
    v = CHECKER(_cfg(F3), bad)
    assert v.critical_pass is False and v.malformed is False


def test_f3_wellformed_empty_not_malformed() -> None:
    v = CHECKER(_cfg(F3), json.dumps([]))
    assert v.malformed is False
