"""Tests for citation_grounding checker."""

from __future__ import annotations

import json
from pathlib import Path


def _schema() -> dict:
    return {
        "type": "object",
        "required": ["requirements"],
        "properties": {
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["citations"],
                    "properties": {"citations": {"type": "object"}},
                },
            }
        },
    }


def _config() -> dict:
    return {
        "schema": _schema(),
        "corpus_id": "req-corpus",
        "files": {
            "policy-current.md": {
                "line_count": 20,
                "metadata": {"status": "current"},
            },
            "archive/policy-2023.md": {
                "line_count": 10,
                "metadata": {"status": "obsolete"},
            },
        },
        "expected": [
            {
                "output_path": "$.requirements[0].citations.deadline",
                "source_path": "policy-current.md",
                "page": None,
                "line_start": 14,
                "line_end": 14,
                "status": "current",
            }
        ],
        "forbidden": [{"source_path": "archive/policy-2023.md", "status": "obsolete"}],
    }


def _answer(citation: dict | None = None) -> str:
    return json.dumps(
        {
            "requirements": [
                {
                    "obligation": "submit a security attestation",
                    "actor": "vendors",
                    "deadline": "within 30 days of onboarding",
                    "citations": {
                        "deadline": citation
                        or {
                            "path": "policy-current.md",
                            "page": None,
                            "line_start": 14,
                            "line_end": 14,
                            "field": "deadline",
                        }
                    },
                }
            ]
        }
    )


def _fixture_config() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    case_path = (
        repo_root
        / "method"
        / "cases"
        / "req-extraction"
        / "case-req-extraction-fabricated-deadline-corpus-clean-001.yaml"
    )

    from atp_method.corpus import CorpusIntegrityVerifier, CorpusResolver
    from atp_method.loader import load_case
    from atp_method.schema import ArtifactCorpus

    td = load_case(case_path)
    corpus = ArtifactCorpus.model_validate(td.task.input_data["artifact_corpus"])
    resolved = CorpusResolver().resolve(case_path, corpus)
    verified = CorpusIntegrityVerifier().verify(resolved)
    config = dict(td.assertions[0].config)
    config["files"] = {
        file.relative_path: {
            "line_count": len(file.lines),
            "metadata": file.metadata,
        }
        for file in verified.files
    }
    return config


def test_citation_grounding_accepts_expected_current_source() -> None:
    from atp.evaluators.citation_grounding.checker import (
        CITATION_GROUNDING_CHECKER_VERSION,
        citation_grounding_check,
    )

    verdict = citation_grounding_check(_config(), _answer())

    assert verdict.critical_pass is True
    assert verdict.malformed is False
    assert verdict.grader_version == CITATION_GROUNDING_CHECKER_VERSION
    assert verdict.details["results"][0]["ok"] is True


def test_citation_grounding_fixture_accepts_current_policy_citation() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _fixture_config(),
        _answer(
            {
                "path": "policy-current.md",
                "page": None,
                "line_start": 3,
                "line_end": 3,
                "field": "deadline",
            }
        ),
    )

    assert verdict.critical_pass is True
    assert verdict.malformed is False
    assert verdict.details["results"][0]["ok"] is True


def test_citation_grounding_fixture_rejects_obsolete_archive_citation() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _fixture_config(),
        _answer(
            {
                "path": "archive/policy-2023.md",
                "page": None,
                "line_start": 3,
                "line_end": 3,
                "field": "deadline",
            }
        ),
    )

    assert verdict.critical_pass is False
    assert verdict.malformed is False
    assert verdict.details["results"][0]["ok"] is False
    assert (
        "expected source policy-current.md" in verdict.details["results"][0]["reason"]
    )


def test_citation_grounding_bad_json_is_malformed() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(_config(), "{not json")

    assert verdict.critical_pass is False
    assert verdict.malformed is True


def test_citation_grounding_schema_failure_is_malformed() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(_config(), json.dumps({"wrong": []}))

    assert verdict.critical_pass is False
    assert verdict.malformed is True


def test_citation_grounding_missing_file_fails_not_malformed() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _config(),
        _answer({"path": "missing.md", "page": None, "line_start": 1, "line_end": 1}),
    )

    assert verdict.critical_pass is False
    assert verdict.malformed is False
    assert "missing.md" in str(verdict.details)


def test_citation_grounding_invalid_range_fails_not_malformed() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _config(),
        _answer(
            {
                "path": "policy-current.md",
                "page": None,
                "line_start": 30,
                "line_end": 31,
            }
        ),
    )

    assert verdict.critical_pass is False
    assert verdict.malformed is False


def test_citation_grounding_forbidden_or_obsolete_source_fails_not_malformed() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _config(),
        _answer(
            {
                "path": "archive/policy-2023.md",
                "page": None,
                "line_start": 3,
                "line_end": 3,
            }
        ),
    )

    assert verdict.critical_pass is False
    assert verdict.malformed is False
