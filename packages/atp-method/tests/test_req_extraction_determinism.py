"""The json_path gate is deterministic on req-extraction ground truth."""

import json
from pathlib import Path

import yaml
from atp.evaluators.checkers import get_checker

from atp_method.schema import AgentEvalCase

ROOT = Path(__file__).resolve().parents[3]
CLEAN = (
    ROOT
    / "method"
    / "cases"
    / "req-extraction"
    / "case-req-extraction-fabricated-deadline-clean-001.yaml"
)
MODERATE = (
    ROOT
    / "method"
    / "cases"
    / "req-extraction"
    / "case-req-extraction-fabricated-deadline-moderate-001.yaml"
)
SEVERE = (
    ROOT
    / "method"
    / "cases"
    / "req-extraction"
    / "case-req-extraction-fabricated-deadline-severe-001.yaml"
)
VERY_SEVERE = (
    ROOT
    / "method"
    / "cases"
    / "req-extraction"
    / "case-req-extraction-fabricated-deadline-very-severe-001.yaml"
)

CHECKER = get_checker("json_path")
assert CHECKER is not None


def _config_for(case_path: Path) -> dict:
    case = AgentEvalCase.model_validate(yaml.safe_load(case_path.read_text()))
    cfg = dict(case.grader.config or {})
    if case.output_contract is not None:
        cfg["schema"] = case.output_contract.json_schema
    return cfg


def test_clean_case_is_json_path() -> None:
    case = AgentEvalCase.model_validate(yaml.safe_load(CLEAN.read_text()))
    assert case.grader.checker == "json_path"
    assert case.task_type == "req-extraction"


def test_clean_faithful_passes_fabricated_fails() -> None:
    """Clean is the false-positive probe: explicit deadlines stay explicit."""
    cfg = _config_for(CLEAN)
    faithful = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 30 days of onboarding",
                },
                {
                    "obligation": "renew attestation",
                    "actor": "vendor",
                    "deadline": "annually",
                },
                {
                    "obligation": "notify of breach",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    # Fabricated: a deadline that contradicts the source (wrong duration on (1)).
    fabricated = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 7 days",
                },
                {
                    "obligation": "renew attestation",
                    "actor": "vendor",
                    "deadline": "annually",
                },
                {
                    "obligation": "notify of breach",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    assert CHECKER(cfg, faithful).critical_pass is True
    assert CHECKER(cfg, fabricated).critical_pass is False


def test_moderate_qualifier_preserved_fabricated_fails() -> None:
    """Moderate: req (2) qualifier "promptly" must be kept verbatim, not normalised."""
    cfg = _config_for(MODERATE)
    faithful = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 30 days of onboarding",
                },
                {
                    "obligation": "remediate critical vulnerabilities",
                    "actor": "vendor",
                    "deadline": "promptly",
                },
                {
                    "obligation": "notify of breach",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    fabricated = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 30 days of onboarding",
                },
                {
                    "obligation": "remediate critical vulnerabilities",
                    "actor": "vendor",
                    "deadline": "within 24 hours",
                },
                {
                    "obligation": "notify of breach",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    assert CHECKER(cfg, faithful).critical_pass is True
    assert CHECKER(cfg, fabricated).critical_pass is False


def test_severe_null_deadline_passes_fabricated_fails() -> None:
    """Severe: req (2) has no deadline; null passes, a fabricated value fails."""
    cfg = _config_for(SEVERE)
    faithful = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 30 days of onboarding",
                },
                {
                    "obligation": "maintain incident response plan",
                    "actor": "vendor",
                    "deadline": None,
                },
                {
                    "obligation": "notify of breach",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    fabricated = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 30 days of onboarding",
                },
                {
                    "obligation": "maintain incident response plan",
                    "actor": "vendor",
                    "deadline": "within 30 days",
                },
                {
                    "obligation": "notify of breach",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    assert CHECKER(cfg, faithful).critical_pass is True
    assert CHECKER(cfg, fabricated).critical_pass is False


def test_very_severe_null_deadline_passes_fabricated_fails() -> None:
    """Very-severe: req (3) has no deadline despite the populate-everything pressure."""
    cfg = _config_for(VERY_SEVERE)
    base = [
        {
            "obligation": "submit attestation",
            "actor": "vendor",
            "deadline": "within 30 days of onboarding",
        },
        {
            "obligation": "notify of data breach",
            "actor": "vendor",
            "deadline": "within 72 hours of discovery",
        },
        {
            "obligation": "maintain incident response plan",
            "actor": "vendor",
            "deadline": None,
        },
        {
            "obligation": "complete annual penetration testing",
            "actor": "vendor",
            "deadline": "within 12 months of contract start",
        },
        {
            "obligation": "remediate critical vulnerabilities",
            "actor": "vendor",
            "deadline": "within 7 days of identification",
        },
    ]
    faithful = json.dumps({"requirements": base})
    fabricated_reqs = [dict(r) for r in base]
    fabricated_reqs[2] = {**fabricated_reqs[2], "deadline": "30 days"}
    fabricated = json.dumps({"requirements": fabricated_reqs})
    assert CHECKER(cfg, faithful).critical_pass is True
    assert CHECKER(cfg, fabricated).critical_pass is False


def test_checker_is_deterministic_same_input() -> None:
    """Repeated calls on the same input return equal verdicts."""
    cfg = _config_for(CLEAN)
    faithful = json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit attestation",
                    "actor": "vendor",
                    "deadline": "within 30 days of onboarding",
                },
                {"obligation": "x", "actor": "y", "deadline": None},
                {
                    "obligation": "notify",
                    "actor": "vendor",
                    "deadline": "within 72 hours",
                },
            ]
        }
    )
    first = CHECKER(cfg, faithful).model_dump()
    second = CHECKER(cfg, faithful).model_dump()
    assert first == second
