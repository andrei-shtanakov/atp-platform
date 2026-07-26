"""What each execution context is allowed to run.

The allowlist is a security control, so these tests are written to fail loudly
if it ever widens by accident — including through the indirect route of
someone adding an executing evaluator to the vocabulary without classifying
it, which would otherwise make it permitted by default.
"""

from __future__ import annotations

import pytest

from atp.evaluation import TRUSTED_LOCAL, UNTRUSTED_SUBMISSION
from atp.evaluation.vocabulary import (
    ASSERTION_TO_EVALUATOR,
    CALLS_EXTERNAL_SERVICE,
    EXECUTES_UNTRUSTED_INPUT,
)

#: Exactly what a submission may have evaluated. Written out rather than
#: sampled: the allowlist is a security control, and a control that widens
#: without a test failing is not a control. Adding a deterministic evaluator
#: is a deliberate act, so updating this set is the deliberate part.
PERMITTED = {
    "artifact_exists",
    "behavior",
    "composite",
    "contains",
    "dir_exists",
    "file_contains",
    "file_count",
    "file_exists",
    "file_not_exists",
    "findings_match",
    "forbidden_tools",
    "max_tool_calls",
    "min_tool_calls",
    "must_use_tools",
    "no_errors",
    "passive_voice",
    "performance",
    "readability",
    "schema",
    "sections",
    "security",
    "sentence_length",
    "style",
    "style_rules",
    "tone",
}


class TestServerPolicy:
    """The benchmark plane runs on input from a self-service token holder."""

    def test_the_allowlist_is_exactly_this(self) -> None:
        """Fails on any widening, including one arriving through the vocabulary."""
        assert UNTRUSTED_SUBMISSION.allowed_assertion_types == PERMITTED

    @pytest.mark.parametrize(
        "assertion", sorted(EXECUTES_UNTRUSTED_INPUT | CALLS_EXTERNAL_SERVICE)
    )
    def test_no_evaluator_class_that_executes_or_calls_out_is_permitted(
        self, assertion: str
    ) -> None:
        """Checked by evaluator class, so a new assertion alias cannot sneak in."""
        for assertion_type, evaluator in ASSERTION_TO_EVALUATOR.items():
            if evaluator == assertion:
                assert UNTRUSTED_SUBMISSION.permits(assertion_type) is False

    @pytest.mark.parametrize(
        "assertion",
        [
            "pytest",
            "npm",
            "lint",
            "code_exec",
            "custom_command",
            "llm_eval",
            "factuality",
        ],
    )
    def test_named_dangerous_assertions_are_refused(self, assertion: str) -> None:
        """Spelled out as well as derived: the derivation could itself break."""
        assert UNTRUSTED_SUBMISSION.permits(assertion) is False

    @pytest.mark.parametrize(
        "assertion",
        [
            "contains",
            "schema",
            "behavior",
            "file_exists",
            "tone",
            "findings_match",
            "security",
            "performance",
        ],
    )
    def test_inspection_only_assertions_are_permitted(self, assertion: str) -> None:
        """Refusing these would leave the plane with nothing but completion."""
        assert UNTRUSTED_SUBMISSION.permits(assertion) is True

    def test_unknown_assertion_type_is_not_permitted(self) -> None:
        """An allowlist answers 'no' to what it has never heard of."""
        assert UNTRUSTED_SUBMISSION.permits("something_invented_later") is False

    def test_the_allowlist_is_not_empty(self) -> None:
        """An empty allowlist would silently reduce the plane to completion."""
        assert UNTRUSTED_SUBMISSION.allowed_assertion_types
        assert len(UNTRUSTED_SUBMISSION.allowed_assertion_types) > 10


class TestPolicyIsServerOwned:
    """Nothing in a submission may widen what runs."""

    def test_policy_is_immutable(self) -> None:
        with pytest.raises(Exception):
            UNTRUSTED_SUBMISSION.name = "trusted_local"  # type: ignore[misc]

    def test_allowlist_is_a_frozenset(self) -> None:
        """A mutable set could be widened by a caller holding a reference."""
        assert isinstance(UNTRUSTED_SUBMISSION.allowed_assertion_types, frozenset)


class TestCliPolicy:
    """The CLI runs the operator's own suite; withholding would only obstruct."""

    def test_nothing_is_withheld(self) -> None:
        assert TRUSTED_LOCAL.allowed_assertion_types is None

    @pytest.mark.parametrize("assertion", ["pytest", "llm_eval", "contains"])
    def test_everything_is_permitted(self, assertion: str) -> None:
        assert TRUSTED_LOCAL.permits(assertion) is True

    def test_the_two_policies_are_not_the_same_object(self) -> None:
        """A shared instance would make one context's change hit the other."""
        assert TRUSTED_LOCAL is not UNTRUSTED_SUBMISSION
