"""Tests for the deterministic findings matcher (R-07 Phase-1b #1)."""

from atp.evaluators.findings.matcher import match_findings, parse_findings

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
