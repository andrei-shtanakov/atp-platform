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


def _answer_without_deadline_citation() -> str:
    return json.dumps(
        {
            "requirements": [
                {
                    "obligation": "raw answer text must stay out of diagnostics",
                    "actor": "vendors",
                    "deadline": "within 30 days of onboarding",
                    "citations": {},
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


def test_citation_grounding_source_mismatch_includes_bounded_diagnostics() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _config(),
        _answer(
            {
                "path": "archive/policy-2023.md",
                "page": None,
                "line_start": 14,
                "line_end": 14,
            }
        ),
    )

    result = verdict.details["results"][0]
    assert verdict.critical_pass is False
    assert result["ok"] is False
    assert result["path"].endswith(".path")
    assert result["field"] == "path"
    assert result["expected_value"] == "policy-current.md"
    assert result["received_value"] == "archive/policy-2023.md"


def test_citation_grounding_page_mismatch_includes_bounded_diagnostics() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    config = _config()
    config["expected"][0]["page"] = 2
    verdict = citation_grounding_check(config, _answer())

    result = verdict.details["results"][0]
    assert verdict.critical_pass is False
    assert result["ok"] is False
    assert result["path"].endswith(".page")
    assert result["field"] == "page"
    assert result["expected_value"] == 2
    assert result["received_value"] is None


def test_citation_grounding_line_range_mismatch_includes_compact_range() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(
        _config(),
        _answer(
            {
                "path": "policy-current.md",
                "page": None,
                "line_start": 13,
                "line_end": 15,
            }
        ),
    )

    result = verdict.details["results"][0]
    assert verdict.critical_pass is False
    assert result["ok"] is False
    assert result["field"] == "line_range" or "line_range" in result["path"]
    assert result["expected_value"] == {"line_start": 14, "line_end": 14}
    assert result["received_value"] == {"line_start": 13, "line_end": 15}


def test_citation_grounding_missing_output_path_diagnostic_is_bounded() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(_config(), _answer_without_deadline_citation())

    result = verdict.details["results"][0]
    assert verdict.critical_pass is False
    assert verdict.malformed is False
    assert result["ok"] is False
    assert result["path"] == "$.requirements[0].citations.deadline"
    assert result["field"] == "deadline"
    assert result["expected_value"] == "citation object"
    assert result["received_value"] == "missing"
    assert "raw answer text must stay out of diagnostics" not in str(result)


def test_citation_grounding_malformed_output_shape_has_no_results_list() -> None:
    from atp.evaluators.citation_grounding.checker import citation_grounding_check

    verdict = citation_grounding_check(_config(), _answer("not an object"))

    assert verdict.critical_pass is False
    assert verdict.malformed is True
    assert "results" not in verdict.details


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
