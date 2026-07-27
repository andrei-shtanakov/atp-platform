"""Regression guard for the assertion-type vocabulary.

Moving this table out of `EvaluatorRegistry` dropped ten entries — the
filesystem and style sub-assertions — because it was transcribed from a
truncated read and then "verified" by counting the new list against itself.
The consequence would have been silent: suite upload warning on valid
assertion types, and the registry refusing to resolve them.

So the expected set is written out in full here. A golden list is tedious on
purpose: it cannot be satisfied by the same mistake that produced it, and
dropping an entry fails loudly instead of narrowing the platform's surface.
"""

from __future__ import annotations

import pytest

from atp.evaluation.vocabulary import (
    ASSERTION_TO_EVALUATOR,
    CALLS_EXTERNAL_SERVICE,
    DETERMINISTIC_EVALUATORS,
    EXECUTES_UNTRUSTED_INPUT,
    deterministic_assertion_types,
    known_assertion_types,
)

#: Every assertion type the platform accepted before the vocabulary moved into
#: atp-core. Extracted from the pre-change EvaluatorRegistry.
EXPECTED: dict[str, str] = {
    "artifact_exists": "artifact",
    "contains": "artifact",
    "schema": "artifact",
    "sections": "artifact",
    "behavior": "behavior",
    "must_use_tools": "behavior",
    "max_tool_calls": "behavior",
    "min_tool_calls": "behavior",
    "no_errors": "behavior",
    "forbidden_tools": "behavior",
    "llm_eval": "llm_judge",
    "code_exec": "code_exec",
    "pytest": "code_exec",
    "npm": "code_exec",
    "custom_command": "code_exec",
    "lint": "code_exec",
    "security": "security",
    "factuality": "factuality",
    "performance": "performance",
    "style": "style",
    "style_rules": "style",
    "tone": "style",
    "readability": "style",
    "passive_voice": "style",
    "sentence_length": "style",
    "file_exists": "filesystem",
    "file_not_exists": "filesystem",
    "file_contains": "filesystem",
    "file_count": "filesystem",
    "dir_exists": "filesystem",
    "composite": "composite",
    "findings_match": "findings_match",
}


def test_vocabulary_matches_the_pre_extraction_surface() -> None:
    """The refactor must not narrow or widen what suites may declare."""
    assert dict(ASSERTION_TO_EVALUATOR) == EXPECTED


def test_registry_resolves_every_known_assertion_type() -> None:
    """A mapping to an evaluator name nobody registered is a dead entry."""
    from atp.evaluators import EvaluatorRegistry

    registry = EvaluatorRegistry()
    unresolvable = [
        assertion
        for assertion in known_assertion_types()
        if registry.get_evaluator_for_assertion(assertion) is None
    ]
    assert not unresolvable


def test_registry_exposes_exactly_the_vocabulary() -> None:
    """Registry and vocabulary must not drift apart in either direction."""
    from atp.evaluators import EvaluatorRegistry

    assert set(EvaluatorRegistry().list_assertion_types()) == known_assertion_types()


class TestBehaviourClassification:
    """Classification decides what a server-side policy may run."""

    def test_every_evaluator_is_classified(self) -> None:
        """An unclassified evaluator would silently count as safe."""
        all_evaluators = set(ASSERTION_TO_EVALUATOR.values())
        classified = (
            DETERMINISTIC_EVALUATORS | EXECUTES_UNTRUSTED_INPUT | CALLS_EXTERNAL_SERVICE
        )
        assert all_evaluators == classified

    @pytest.mark.parametrize(
        "unsafe",
        [
            EXECUTES_UNTRUSTED_INPUT,
            CALLS_EXTERNAL_SERVICE,
        ],
    )
    def test_classes_do_not_overlap(self, unsafe: frozenset[str]) -> None:
        assert not DETERMINISTIC_EVALUATORS & unsafe

    @pytest.mark.parametrize(
        "assertion",
        ["file_exists", "file_not_exists", "file_contains", "file_count", "dir_exists"],
    )
    def test_filesystem_assertions_are_deterministic_now(self, assertion: str) -> None:
        """ADR-008 track B: the root is granted by the composition, not the suite.

        The old exclusion was about *where* the evaluator looked, not what it
        did: it took `workspace_path` from the suite, so on the server it
        answered questions about the server's own disk. It now inspects only
        the directory it was handed — the artifact sandbox — which is the
        submission and nothing else. The behavioural proof is in
        `tests/unit/evaluators/test_filesystem.py`.
        """
        assert assertion in deterministic_assertion_types()

    def test_composite_is_deterministic_again(self) -> None:
        """ADR-008 track A: its leaves go through the resolver it is handed.

        It stopped reaching for the global registry, so nesting an excluded
        assertion under it no longer escapes the policy — the behavioural
        proof is in `tests/unit/evaluators/test_composite.py`.
        """
        assert "composite" in deterministic_assertion_types()

    @pytest.mark.parametrize("assertion", ["pytest", "code_exec", "npm", "lint"])
    def test_executing_assertions_are_not_deterministic(self, assertion: str) -> None:
        assert assertion not in deterministic_assertion_types()

    @pytest.mark.parametrize("assertion", ["llm_eval", "factuality"])
    def test_network_assertions_are_not_deterministic(self, assertion: str) -> None:
        assert assertion not in deterministic_assertion_types()

    @pytest.mark.parametrize(
        "assertion", ["tone", "contains", "findings_match", "no_errors"]
    )
    def test_inspection_only_assertions_are_deterministic(self, assertion: str) -> None:
        """The ten entries dropped by the bad extraction live in this class."""
        assert assertion in deterministic_assertion_types()
