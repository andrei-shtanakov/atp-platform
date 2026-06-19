"""Tests for object-form findings unwrap + output_contract schema gate (Task 1)."""

from atp.evaluators.findings.checker import findings_check
from atp.evaluators.findings.matcher import grade_findings, parse_findings

_OBJ_SCHEMA = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rule_id", "anchor", "severity"],
                "properties": {
                    "rule_id": {"type": "string"},
                    "file": {"type": "string"},
                    "anchor": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "major", "minor"],
                    },
                    "fix": {"type": "string"},
                },
            },
        }
    },
}
_EXPECTED = [
    {"rule_ids": ["SEC-011"], "anchor": 'f"...{user_id}"', "severity": "critical"}
]


def test_parse_findings_unwraps_object_form() -> None:
    out = parse_findings(
        '{"findings": [{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]}'
    )
    assert out == [{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]


def test_parse_findings_still_accepts_legacy_array() -> None:
    out = parse_findings(
        '[{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]'
    )
    assert out == [{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]


def test_grade_findings_schema_violation_is_malformed() -> None:
    # findings present but as a bare array → violates the object schema
    r = grade_findings(
        '[{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]',
        _EXPECTED,
        [],
        schema=_OBJ_SCHEMA,
    )
    assert r.malformed is True
    assert r.critical_pass is False


def test_grade_findings_valid_object_matches() -> None:
    payload = (
        '{"findings": [{"rule_id": "SEC-011",'
        ' "anchor": "f\\"...{user_id}\\"", "severity": "critical"}]}'
    )
    r = grade_findings(
        payload,
        _EXPECTED,
        [],
        schema=_OBJ_SCHEMA,
    )
    assert r.malformed is False


def test_findings_check_threads_schema_from_config() -> None:
    v = findings_check(
        {
            "expected_findings": _EXPECTED,
            "must_not_flag": [],
            "schema": _OBJ_SCHEMA,
        },
        '[{"rule_id": "SEC-011", "anchor": "a", "severity": "critical"}]',
    )
    assert v.malformed is True  # bare array rejected by the object schema
