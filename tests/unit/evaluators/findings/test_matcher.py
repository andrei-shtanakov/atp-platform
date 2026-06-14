"""Tests for the deterministic findings matcher (R-07 Phase-1b #1)."""

import pytest
from pydantic import ValidationError

from atp.evaluators.findings.matcher import (
    Finding,
    grade_findings,
    match_findings,
    parse_findings,
    validate_findings,
)

EXPECTED = [
    {
        "rule_ids": ["SEC-011", "sql-injection", "cwe-89"],
        "anchor": 'f"SELECT',
        "severity": "critical",
    }
]
MUST_NOT = [{"anchor": "cursor.execute(query, (user_id,))"}, {"anchor": "logger.debug"}]


def test_anchor_hit_with_synonym_ruleid() -> None:
    findings = [
        {
            "rule_id": "cwe-89",
            "file": "app.py",
            "anchor": 'query = f"SELECT * FROM users',
        }
    ]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.critical_pass is True
    assert r.recall == 1.0
    assert r.false_positives == []


def test_line_number_independence() -> None:
    findings = [{"rule_id": "SQL-Injection", "anchor": 'x = f"SELECT 1"'}]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.critical_pass is True


def test_false_positive_on_compliant_line_fails_gate() -> None:
    findings = [
        {"rule_id": "sql-injection", "anchor": 'f"SELECT'},
        {"rule_id": "SEC-011", "anchor": "logger.debug('x')"},
    ]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.false_positives
    assert r.critical_pass is False


def test_missed_critical_fails_gate() -> None:
    findings = [{"rule_id": "style-1", "anchor": "return jsonify(rows)"}]
    r = match_findings(findings, EXPECTED, MUST_NOT)
    assert r.recall == 0.0
    assert r.critical_pass is False


def test_compliant_case_empty_findings_passes() -> None:
    r = match_findings([], [], MUST_NOT)
    assert r.critical_pass is True
    assert r.recall == 1.0


def test_compliant_case_false_positive_fails() -> None:
    findings = [{"rule_id": "SEC-011", "anchor": "cursor.execute(query, (user_id,))"}]
    r = match_findings(findings, [], MUST_NOT)
    assert r.critical_pass is False


def test_parse_findings_strips_code_fence() -> None:
    text = '```json\n[{"rule_id": "x", "anchor": "y"}]\n```'
    assert parse_findings(text) == [{"rule_id": "x", "anchor": "y"}]


def test_parse_findings_unparseable_returns_none() -> None:
    assert parse_findings("I think there is a SQL injection somewhere.") is None


def test_empty_rule_ids_does_not_crash() -> None:
    # rule_ids present but empty (malformed config) must produce a deterministic
    # result, not raise IndexError.
    expected = [{"rule_ids": [], "anchor": 'f"SELECT', "severity": "critical"}]
    r = match_findings([{"rule_id": "x", "anchor": 'f"SELECT 1'}], expected, [])
    # empty synonym set => rule check is skipped, so the anchor match alone counts
    assert r.critical_pass is True


def test_multiple_findings_on_same_compliant_line_count_as_multiple_fps() -> None:
    findings = [
        {"rule_id": "a", "anchor": "logger.debug('x')"},
        {"rule_id": "b", "anchor": "logger.debug('y')"},
    ]
    r = match_findings(findings, [], MUST_NOT)
    assert len(r.false_positives) == 2
    assert r.critical_pass is False


def test_match_findings_is_never_malformed() -> None:
    # The pure matcher operates on already-trusted dicts; malformation is decided
    # upstream by grade_findings, so MatchResult.malformed defaults False.
    r = match_findings([], [], MUST_NOT)
    assert r.malformed is False


# --- strict Finding validation (R-07 P3) -----------------------------------


def test_finding_accepts_valid_and_ignores_extra_keys() -> None:
    f = Finding.model_validate(
        {
            "rule_id": "SEC-011",
            "anchor": 'f"SELECT',
            "severity": "critical",
            "file": "app.py",
            "line": 42,
            "fix": "use params",
        }
    )
    assert f.rule_id == "SEC-011"
    assert not hasattr(f, "file")  # extra="ignore" drops unknown keys


@pytest.mark.parametrize(
    "bad",
    [
        {"rule_id": "x", "anchor": "y"},  # missing severity
        {"rule_id": "x", "anchor": "y", "severity": "high"},  # bad literal
        {"anchor": "y", "severity": "minor"},  # missing rule_id
        {"rule_id": "x", "severity": "minor"},  # missing anchor
    ],
)
def test_finding_rejects_invalid(bad: dict) -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate(bad)


def test_validate_findings_passes_all_valid() -> None:
    items = [{"rule_id": "x", "anchor": "y", "severity": "major"}]
    assert validate_findings(items) == items


def test_validate_findings_one_bad_malforms_whole_output() -> None:
    # strict-global: a single invalid finding malforms the entire output (no
    # lenient drop-and-continue).
    items = [
        {"rule_id": "x", "anchor": "y", "severity": "major"},
        {"rule_id": "z", "anchor": "w"},  # missing severity
    ]
    assert validate_findings(items) is None


def test_validate_findings_non_dict_element_is_malformed() -> None:
    assert validate_findings(["not a dict"]) is None


def test_grade_findings_unparseable_is_malformed() -> None:
    r = grade_findings("I think there is a bug somewhere", EXPECTED, MUST_NOT)
    assert r.malformed is True
    assert r.critical_pass is False


def test_grade_findings_none_text_is_malformed() -> None:
    r = grade_findings(None, EXPECTED, MUST_NOT)
    assert r.malformed is True
    assert r.critical_pass is False


def test_grade_findings_invalid_finding_is_malformed_not_missed() -> None:
    # A defect-shaped finding without severity is malformed — NOT a missed defect.
    text = '[{"rule_id": "cwe-89", "anchor": "f\\"SELECT"}]'
    r = grade_findings(text, EXPECTED, MUST_NOT)
    assert r.malformed is True
    assert r.critical_pass is False


def test_grade_findings_valid_match_is_not_malformed() -> None:
    text = (
        '[{"rule_id": "cwe-89", "anchor": "x = f\\"SELECT 1", "severity": "critical"}]'
    )
    r = grade_findings(text, EXPECTED, MUST_NOT)
    assert r.malformed is False
    assert r.critical_pass is True


def test_grade_findings_valid_miss_is_not_malformed() -> None:
    # Well-formed output that simply misses the defect: a real miss, not malformed.
    text = '[{"rule_id": "style-1", "anchor": "return rows", "severity": "minor"}]'
    r = grade_findings(text, EXPECTED, MUST_NOT)
    assert r.malformed is False
    assert r.critical_pass is False
