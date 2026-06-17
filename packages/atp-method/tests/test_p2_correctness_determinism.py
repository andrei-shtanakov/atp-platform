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


# --- S1: cents integer-invariant violation (defect, SPEC-001) ------------------

S1 = ROOT / "method/cases/code-review/case-code-review-correctness-spec-cents-001.yaml"


def test_s1_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": "total = subtotal_cents * 1.0825",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(S1), good)
    assert v.critical_pass is True and v.malformed is False


def test_s1_miss_fails() -> None:
    v = CHECKER(_cfg(S1), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_s1_near_miss_anchor_not_accepted() -> None:
    # flagging the compliant summation line must NOT satisfy the expected finding
    near = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": "subtotal_cents = sum(item.price_cents for item in cart)",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(S1), near)
    assert v.critical_pass is False and v.fp_count == 1


def test_s1_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(S1),
        json.dumps(
            [
                {
                    "rule_id": "SPEC-001",
                    "anchor": "total = subtotal_cents * 1.0825",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False


# --- S2: unbounded list endpoint (defect, SPEC-002) ----------------------------

S2 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-spec-pagination-001.yaml"
)


def test_s2_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "SPEC-002",
                "anchor": "return session.query(User).all()",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(S2), good)
    assert v.critical_pass is True and v.malformed is False


def test_s2_miss_fails() -> None:
    v = CHECKER(_cfg(S2), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_s2_near_miss_anchor_not_accepted() -> None:
    # flagging the route decorator must NOT satisfy the expected finding
    near = json.dumps(
        [
            {
                "rule_id": "SPEC-002",
                "anchor": '+22: @app.get("/users")',
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(S2), near)
    assert v.critical_pass is False and v.fp_count == 1


def test_s2_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(S2),
        json.dumps(
            [
                {
                    "rule_id": "SPEC-002",
                    "anchor": "return session.query(User).all()",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False


# --- D1: CROSS-FILE double-conversion buried in refactor noise -----------------

D1 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-distractor-doubleconv-001.yaml"  # noqa: E501
)


def test_d1_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": "shipping_total_cents = to_cents(shipping_cents)",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(D1), good)
    assert v.critical_pass is True and v.malformed is False


def test_d1_miss_fails() -> None:
    v = CHECKER(_cfg(D1), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_d1_near_miss_distractor_not_accepted() -> None:
    # flagging the (correct) to_cents definition must NOT match AND is a FP
    near = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": "return round(dollars * 100)",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(D1), near)
    assert v.critical_pass is False and v.fp_count == 1


def test_d1_precision_two_false_positives() -> None:
    two = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": "return round(dollars * 100)",
                "severity": "major",
            },
            {
                "rule_id": "SPEC-001",
                "anchor": "grand_total_cents = line_total_cents + shipping_total_cents",
                "severity": "major",
            },
        ]
    )
    v = CHECKER(_cfg(D1), two)
    assert v.fp_count >= 2


def test_d1_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(D1),
        json.dumps(
            [
                {
                    "rule_id": "SPEC-001",
                    "anchor": "shipping_total_cents = to_cents(shipping_cents)",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False


def test_d1_overflag_dict_line_is_fp() -> None:
    # an over-flag on a newly-covered dict-construction line is a real FP now
    over = json.dumps(
        [
            {
                "rule_id": "SPEC-001",
                "anchor": '+50:         "shipping": shipping_total_cents,',
                "severity": "major",
            }
        ]
    )
    v = CHECKER(_cfg(D1), over)
    assert v.fp_count >= 1 and v.critical_pass is False


# --- D2: CROSS-FILE swapped-argument boundary bug under noise -------------------

D2 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-distractor-argorder-001.yaml"  # noqa: E501
)


def test_d2_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "LOG-002",
                "anchor": "left_cents = remaining(used_cents, cap_cents)",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(D2), good)
    assert v.critical_pass is True and v.malformed is False


def test_d2_miss_fails() -> None:
    v = CHECKER(_cfg(D2), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_d2_near_miss_distractor_not_accepted() -> None:
    # flagging the CORRECT remaining(cap, used) call in is_exhausted is a FP
    near = json.dumps(
        [
            {
                "rule_id": "LOG-002",
                "anchor": "return remaining(cap, used) <= 0",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(D2), near)
    assert v.critical_pass is False and v.fp_count == 1


def test_d2_precision_two_false_positives() -> None:
    two = json.dumps(
        [
            {
                "rule_id": "LOG-002",
                "anchor": "return remaining(cap, used) <= 0",
                "severity": "major",
            },
            {
                "rule_id": "LOG-002",
                "anchor": "return charge_cents <= left_cents",
                "severity": "major",
            },
        ]
    )
    v = CHECKER(_cfg(D2), two)
    assert v.fp_count >= 2


def test_d2_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(D2),
        json.dumps(
            [
                {
                    "rule_id": "LOG-002",
                    "anchor": "left_cents = remaining(used_cents, cap_cents)",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False


def test_d2_overflag_compliant_line_is_fp() -> None:
    # an over-flag on a covered, compliant line is a real FP now
    over = json.dumps(
        [
            {
                "rule_id": "LOG-002",
                "anchor": "return charge_cents <= left_cents",
                "severity": "major",
            }
        ]
    )
    v = CHECKER(_cfg(D2), over)
    assert v.fp_count >= 1 and v.critical_pass is False


# --- D3: SINGLE-FILE buried boundary bug under noise (contrast control) --------

D3 = (
    ROOT
    / "method/cases/code-review/case-code-review-correctness-distractor-singlefile-001.yaml"  # noqa: E501
)


def test_d3_good_passes() -> None:
    good = json.dumps(
        [
            {
                "rule_id": "LOG-001",
                "anchor": "if page > last_page:",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(D3), good)
    assert v.critical_pass is True and v.malformed is False


def test_d3_miss_fails() -> None:
    v = CHECKER(_cfg(D3), json.dumps([]))
    assert v.critical_pass is False and v.malformed is False


def test_d3_near_miss_distractor_not_accepted() -> None:
    # flagging the correct has_next_page predicate is a FP, not the defect
    near = json.dumps(
        [
            {
                "rule_id": "LOG-001",
                "anchor": "return page < page_count(total, size)",
                "severity": "critical",
            }
        ]
    )
    v = CHECKER(_cfg(D3), near)
    assert v.critical_pass is False and v.fp_count == 1


def test_d3_precision_two_false_positives() -> None:
    two = json.dumps(
        [
            {
                "rule_id": "LOG-001",
                "anchor": "return page < page_count(total, size)",
                "severity": "major",
            },
            {
                "rule_id": "LOG-001",
                "anchor": "return (total + size - 1) // size",
                "severity": "major",
            },
        ]
    )
    v = CHECKER(_cfg(D3), two)
    assert v.fp_count >= 2


def test_d3_wellformed_not_malformed() -> None:
    v = CHECKER(
        _cfg(D3),
        json.dumps(
            [
                {
                    "rule_id": "LOG-001",
                    "anchor": "if page > last_page:",
                    "severity": "critical",
                }
            ]
        ),
    )
    assert v.malformed is False


def test_d3_overflag_fallthrough_line_is_fp() -> None:
    # flagging the unguarded fall-through (part of the bug locus, but not the
    # expected anchor) is now a real FP, not a free unknown_extra
    over = json.dumps(
        [
            {
                "rule_id": "LOG-001",
                "anchor": "+41:     return page",
                "severity": "major",
            }
        ]
    )
    v = CHECKER(_cfg(D3), over)
    assert v.fp_count >= 1 and v.critical_pass is False
