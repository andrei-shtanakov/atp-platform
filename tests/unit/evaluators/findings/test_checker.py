"""Tests for the findings_match checker → CaseVerdict mapping (Phase A-1)."""

import json

from atp.evaluators.checkers import get_checker  # triggers built-in registration

EXPECTED = [
    {"rule_ids": ["SEC-011", "cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
]
MUST_NOT = [{"anchor": "logger.debug"}]


def test_findings_match_registered() -> None:
    assert get_checker("findings_match") is not None


def test_valid_match_verdict() -> None:
    check = get_checker("findings_match")
    assert check is not None
    text = json.dumps(
        [{"rule_id": "cwe-89", "anchor": 'x = f"SELECT 1', "severity": "critical"}]
    )
    v = check({"expected_findings": EXPECTED, "must_not_flag": MUST_NOT}, text)
    assert v.critical_pass is True
    assert v.malformed is False
    assert v.recall == 1.0
    assert v.fp_count == 0
    assert v.grader_version == "findings_match@1"
    assert v.details["malformed"] is False


def test_malformed_verdict() -> None:
    check = get_checker("findings_match")
    assert check is not None
    # missing required severity -> malformed (not a silent miss)
    text = json.dumps([{"rule_id": "cwe-89", "anchor": 'f"SELECT'}])
    v = check({"expected_findings": EXPECTED, "must_not_flag": MUST_NOT}, text)
    assert v.malformed is True
    assert v.critical_pass is False
