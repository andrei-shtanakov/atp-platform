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
