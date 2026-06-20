"""Tests for the shared capability envelope module (Phase A-2)."""

import pytest

from atp_method.envelopes import (
    DEFAULT_MODEL,
    REVIEW_ENVELOPE,
    build_prompt,
    get_envelope,
)


def test_default_model_is_pinned() -> None:
    assert DEFAULT_MODEL == "claude-opus-4-8"


def test_get_envelope_review() -> None:
    assert get_envelope("review") is REVIEW_ENVELOPE
    assert "{task}" in get_envelope("review")


def test_get_envelope_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_envelope("nope")


def test_build_prompt_inlines_task_and_artifacts() -> None:
    # Artifacts are delivered the way the loader emits them: under
    # task.input_data["artifacts"]. The ATP `Context` model has no artifacts
    # field, so reading context.artifacts (the old path) always came up empty.
    request = {
        "task": {
            "description": "Review the diff",
            "input_data": {"artifacts": [{"id": "diff", "content": "x = 1"}]},
        },
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert "Review the diff" in prompt
    assert "--- diff ---" in prompt
    assert "x = 1" in prompt


def test_build_prompt_tolerates_missing_fields() -> None:
    assert isinstance(build_prompt({}, "{task}"), str)


def test_build_prompt_delivers_loader_artifacts_end_to_end() -> None:
    """Regression: the diff + rules a real case carries must reach the model.

    Mirrors the live wiring (loader -> TestDefinition.task.input_data ->
    ATPRequest.model_dump_json -> shim build_prompt). Guards against the
    artifact-path drift that made the paid pipe-check review an empty diff.
    """
    import json
    from pathlib import Path

    from atp.protocol.models import ATPRequest, Task

    from atp_method.loader import load_suite

    # Resolve from this file, not cwd: tests/test_envelopes.py -> parents[3].
    repo_root = Path(__file__).resolve().parents[3]
    suite = load_suite(str(repo_root / "method" / "cases" / "code-review"))
    td = next(t for t in suite.tests if t.id == "case-code-review-sqli-clean-001")
    req = ATPRequest(
        task_id=td.id,
        task=Task(description=td.task.description, input_data=td.task.input_data),
    )
    req_json = json.loads(req.model_dump_json())

    prompt = build_prompt(req_json, get_envelope("review"))
    # The compliant query (diff) and the SEC-011 rule (kb-rules) both arrive.
    assert "SELECT" in prompt
    assert "SEC-011" in prompt


def test_build_prompt_uses_format_instruction_when_present() -> None:
    request = {
        "task": {
            "description": "Extract requirements",
            "input_data": {
                "artifacts": [{"id": "doc", "content": "Vendor must submit."}],
                "output_contract": {
                    "artifact_name": "answer",
                    "format_instruction": "Return ONLY JSON {requirements:[...]}",
                },
            },
        }
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert "Return ONLY JSON" in prompt
    assert "Response JSON Schema:" not in prompt
    assert "Vendor must submit." in prompt
    assert "senior code reviewer" not in prompt


def test_build_prompt_includes_response_schema_with_format_instruction() -> None:
    request = {
        "task": {
            "description": "Extract requirements",
            "input_data": {
                "artifacts": [{"id": "doc", "content": "Vendor must submit."}],
                "output_contract": {
                    "artifact_name": "answer",
                    "format_instruction": "Return ONLY JSON.",
                    "schema": {
                        "type": "object",
                        "required": ["requirements"],
                        "properties": {
                            "requirements": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            },
        }
    }

    prompt = build_prompt(request, get_envelope("review"))

    assert "Return ONLY JSON." in prompt
    assert "Response JSON Schema:" in prompt
    assert '"requirements"' in prompt
    assert '"type": "object"' in prompt
    assert "Vendor must submit." in prompt
    assert "senior code reviewer" not in prompt


def test_build_prompt_falls_back_to_review_without_contract() -> None:
    request = {"task": {"description": "Review", "input_data": {"artifacts": []}}}
    prompt = build_prompt(request, get_envelope("review"))
    assert "senior code reviewer" in prompt


def test_build_prompt_contract_without_instruction_uses_review() -> None:
    request = {
        "task": {
            "description": "Review",
            "input_data": {
                "artifacts": [],
                "output_contract": {"artifact_name": "a"},
            },
        }
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert "senior code reviewer" in prompt


def test_build_prompt_includes_corpus_id_and_paths_without_file_contents() -> None:
    request = {
        "task": {
            "description": "Extract requirements with citations",
            "input_data": {
                "run_mode": "read_only_corpus",
                "artifact_corpus": {
                    "id": "req-corpus",
                    "files": ["policy-current.md", "archive/policy-2023.md"],
                },
                "output_contract": {
                    "artifact_name": "answer",
                    "format_instruction": "Return JSON with citations.",
                },
            },
        }
    }

    prompt = build_prompt(request, get_envelope("review"))

    assert "req-corpus" in prompt
    assert "policy-current.md" in prompt
    assert "archive/policy-2023.md" in prompt
    assert "within 30 days of onboarding" not in prompt


def test_build_prompt_lists_fixture_corpus_paths_without_inlining_fixture_text() -> (
    None
):
    import json
    from pathlib import Path

    from atp.protocol.models import ATPRequest, Task

    from atp_method.corpus import CorpusIntegrityVerifier, CorpusResolver
    from atp_method.loader import load_case
    from atp_method.schema import ArtifactCorpus

    repo_root = Path(__file__).resolve().parents[3]
    case_path = (
        repo_root
        / "method"
        / "cases"
        / "req-extraction"
        / "case-req-extraction-fabricated-deadline-corpus-clean-001.yaml"
    )
    td = load_case(case_path)
    corpus = ArtifactCorpus.model_validate(td.task.input_data["artifact_corpus"])
    resolved = CorpusResolver().resolve(case_path, corpus)
    verified = CorpusIntegrityVerifier().verify(resolved)
    input_data = dict(td.task.input_data)
    artifact_corpus = dict(input_data["artifact_corpus"])
    artifact_corpus["files"] = [file.relative_path for file in verified.files]
    input_data["artifact_corpus"] = artifact_corpus
    req = ATPRequest(
        task_id=td.id,
        task=Task(description=td.task.description, input_data=input_data),
    )

    prompt = build_prompt(json.loads(req.model_dump_json()), get_envelope("review"))

    assert "fabricated-deadline-clean-corpus" in prompt
    assert "Response JSON Schema:" in prompt
    assert "citations.deadline as a single citation object" in prompt
    assert '"citations"' in prompt
    assert '"deadline"' in prompt
    assert "policy-current.md" in prompt
    assert "vendor-addendum.md" in prompt
    assert "archive/policy-2023.md" in prompt
    assert "within 30 days of onboarding" not in prompt
    assert "within 45 days of onboarding" not in prompt
