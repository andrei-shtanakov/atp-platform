"""The JSON contract accepts programmatic+checker findings cases (Phase A-1)."""

import json
from pathlib import Path

import jsonschema
import pytest

# repo-root-relative; tests run from packages/atp-method
SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[3] / "method" / "agent-eval-case.schema.json"
    ).read_text()
)


def _case(grader: dict) -> dict:
    return {
        "id": "case-1",
        "version": 1,
        "family": "f",
        "status": "active",
        "suite_type": "probe",
        "capability": "safety_compliance",
        "construction_axis": "adversarial_environment",
        "axis_level": "moderate",
        "instruction": "review",
        "artifacts": [{"id": "d", "type": "text", "content": "x"}],
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "misses it",
        "grader": grader,
        "provenance": {"author": "a", "created": "2026-06-14"},
    }


def test_contract_accepts_programmatic_checker_findings() -> None:
    case = _case(
        {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [
                {"rule_ids": ["SEC-011"], "anchor": 'f"SELECT', "severity": "critical"}
            ],
            "must_not_flag": [{"anchor": "logger.debug"}],
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        }
    )
    jsonschema.validate(case, SCHEMA)  # must not raise


def test_contract_rejects_findings_checker_without_expected_findings() -> None:
    case = _case(
        {
            "type": "programmatic",
            "checker": "findings_match",
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        }
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(case, SCHEMA)


def test_contract_allows_expected_finding_without_severity() -> None:
    # pydantic ExpectedFinding defaults severity to "critical"; the JSON contract
    # must therefore NOT require it (else valid-in-Python YAML fails the contract).
    case = _case(
        {
            "type": "programmatic",
            "checker": "findings_match",
            "expected_findings": [{"rule_ids": ["SEC-011"], "anchor": 'f"SELECT'}],
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        }
    )
    jsonschema.validate(case, SCHEMA)  # must not raise


def test_contract_rejects_empty_checker() -> None:
    case = _case(
        {
            "type": "programmatic",
            "checker": "",
            "expected_findings": [],
            "critical_check": "flag it",
            "scoring": "fail if critical fails",
        }
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(case, SCHEMA)
