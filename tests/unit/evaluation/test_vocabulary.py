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
    DELEGATES_TO_REGISTRY,
    DETERMINISTIC_EVALUATORS,
    EXECUTES_UNTRUSTED_INPUT,
    READS_HOST_FILESYSTEM,
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
            DETERMINISTIC_EVALUATORS
            | EXECUTES_UNTRUSTED_INPUT
            | CALLS_EXTERNAL_SERVICE
            | READS_HOST_FILESYSTEM
            | DELEGATES_TO_REGISTRY
        )
        assert all_evaluators == classified

    @pytest.mark.parametrize(
        "unsafe",
        [
            EXECUTES_UNTRUSTED_INPUT,
            CALLS_EXTERNAL_SERVICE,
            READS_HOST_FILESYSTEM,
            DELEGATES_TO_REGISTRY,
        ],
    )
    def test_classes_do_not_overlap(self, unsafe: frozenset[str]) -> None:
        assert not DETERMINISTIC_EVALUATORS & unsafe

    @pytest.mark.parametrize(
        "assertion",
        ["file_exists", "file_not_exists", "file_contains", "file_count", "dir_exists"],
    )
    def test_host_filesystem_assertions_are_not_deterministic(
        self, assertion: str
    ) -> None:
        """`workspace_path` comes from the suite, so the server's disk is the target."""
        assert assertion not in deterministic_assertion_types()

    def test_delegating_assertions_are_not_deterministic(self) -> None:
        """`composite` resolves its leaves itself, so it can nest an excluded one."""
        assert "composite" not in deterministic_assertion_types()

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
