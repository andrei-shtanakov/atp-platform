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
    request = {
        "task": {"description": "Review the diff"},
        "context": {"artifacts": [{"id": "diff", "content": "x = 1"}]},
    }
    prompt = build_prompt(request, get_envelope("review"))
    assert "Review the diff" in prompt
    assert "--- diff ---" in prompt
    assert "x = 1" in prompt


def test_build_prompt_tolerates_missing_fields() -> None:
    assert isinstance(build_prompt({}, "{task}"), str)
