"""Schema tests for read-only artifact corpus cases."""

from copy import deepcopy

import pytest
from pydantic import ValidationError


def _corpus() -> dict:
    return {
        "id": "fabricated-deadline-clean-corpus",
        "root": (
            "method/cases/req-extraction/assets/fabricated-deadline-clean-corpus-001"
        ),
        "include": ["**/*.md", "**/*.txt"],
        "exclude": ["README.md"],
        "digest": {
            "algorithm": "sha256",
            "normalization": "lf",
            "manifest_path": "manifest.sha256",
        },
        "metadata_path": "corpus.meta.yaml",
    }


def _case() -> dict:
    return {
        "id": "case-demo-001",
        "version": 1,
        "family": "demo",
        "status": "active",
        "suite_type": "probe",
        "capability": "calibration",
        "construction_axis": "information_conditions",
        "axis_level": "clean",
        "instruction": "Extract requirements and cite source lines.",
        "environment": {"tools": ["file_read"], "side_effects": "none"},
        "expected_failure_mode": "cites an obsolete source",
        "grader": {
            "type": "programmatic",
            "checker": "citation_grounding",
            "critical_check": "ground citations in current corpus files",
            "scoring": "fail if critical_check fails",
            "config": {
                "artifact_name": "answer",
                "corpus_id": "fabricated-deadline-clean-corpus",
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
            },
        },
        "output_contract": {
            "artifact_name": "answer",
            "schema": {
                "type": "object",
                "required": ["requirements"],
                "properties": {"requirements": {"type": "array"}},
            },
        },
        "run_mode": "read_only_corpus",
        "artifact_corpus": _corpus(),
        "provenance": {"author": "test", "created": "2026-06-20"},
    }


def test_corpus_models_accept_spec_example_paths() -> None:
    from atp_method.schema import ArtifactCorpus, CorpusDigest

    corpus = ArtifactCorpus.model_validate(_corpus())

    assert corpus.id == "fabricated-deadline-clean-corpus"
    assert corpus.root.endswith("fabricated-deadline-clean-corpus-001")
    assert corpus.include == ["**/*.md", "**/*.txt"]
    assert corpus.exclude == ["README.md"]
    assert corpus.digest == CorpusDigest(
        algorithm="sha256",
        normalization="lf",
        manifest_path="manifest.sha256",
    )
    assert corpus.metadata_path == "corpus.meta.yaml"


def test_artifact_corpus_under_text_out_is_rejected() -> None:
    from atp_method.schema import AgentEvalCase

    doc = _case()
    doc["run_mode"] = "text_out"

    with pytest.raises(ValidationError, match="artifact_corpus"):
        AgentEvalCase.model_validate(doc)


def test_read_only_corpus_without_artifact_corpus_is_rejected() -> None:
    from atp_method.schema import AgentEvalCase

    doc = _case()
    del doc["artifact_corpus"]

    with pytest.raises(ValidationError, match="artifact_corpus"):
        AgentEvalCase.model_validate(doc)


def test_read_only_corpus_with_valid_corpus_validates_as_wired_mode() -> None:
    from atp_method.schema import WIRED_RUN_MODES, AgentEvalCase

    case = AgentEvalCase.model_validate(_case())

    assert "read_only_corpus" in WIRED_RUN_MODES
    assert case.run_mode == "read_only_corpus"
    assert case.artifact_corpus is not None
    assert case.artifact_corpus.id == "fabricated-deadline-clean-corpus"


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        (("root",), "/tmp/corpus"),
        (("root",), "../corpus"),
        (("root",), "~/corpus"),
        (("root",), "safe//empty"),
        (("root",), "safe/\x00bad"),
        (("digest", "manifest_path"), "/tmp/manifest.sha256"),
        (("digest", "manifest_path"), "nested/../manifest.sha256"),
        (("metadata_path",), "~/corpus.meta.yaml"),
        (("include", 0), "../*.md"),
        (("exclude", 0), "archive/\x00*.md"),
    ],
)
def test_corpus_paths_reject_unsafe_values(
    field_path: tuple[str | int, ...], bad_value: str
) -> None:
    from atp_method.schema import ArtifactCorpus

    corpus = _corpus()
    target = corpus
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = bad_value  # type: ignore[index]

    with pytest.raises(ValidationError, match="path|relative|unsafe|absolute"):
        ArtifactCorpus.model_validate(corpus)


@pytest.mark.parametrize(
    ("key", "value"),
    [("algorithm", "sha1"), ("normalization", "raw")],
)
def test_digest_rejects_unsupported_settings(key: str, value: str) -> None:
    from atp_method.schema import CorpusDigest

    digest = deepcopy(_corpus()["digest"])
    digest[key] = value

    with pytest.raises(ValidationError):
        CorpusDigest.model_validate(digest)


def test_duplicate_tools_rejected_for_corpus_case_before_wiring() -> None:
    from atp_method.schema import AgentEvalCase

    doc = _case()
    doc["environment"]["tools"] = ["file_read", "file_read"]

    with pytest.raises(ValidationError, match="duplicates"):
        AgentEvalCase.model_validate(doc)


def test_citation_grounding_requires_non_empty_expected_config() -> None:
    from atp_method.schema import Grader

    with pytest.raises(ValidationError, match="expected"):
        Grader(
            type="programmatic",
            checker="citation_grounding",
            critical_check="ground citations",
            scoring="fail if critical fails",
            config={"artifact_name": "answer", "expected": []},
        )
