"""Composite under a policy: what a refused leaf does to the operator above it.

ADR-008 track A. Two things are being proved here, and the second is the one
that took the ADR a revision to get right:

1. A leaf the policy withholds is never resolved, so `composite` cannot be
   used to smuggle `pytest` past the server allowlist.
2. That withheld leaf is `UNEVALUATED`, not `False` — and the operators
   propagate it by Kleene logic rather than collapsing it to a failure.

The second matters because collapsing is the tempting shortcut and it is
wrong in a specific, quiet way: it publishes "assessed and bad" for something
nobody assessed, which is exactly the confusion the benchmark score contract
exists to remove.
"""

from __future__ import annotations

from typing import Any

import pytest

from atp.core.results import EvalCheck, EvalResult
from atp.evaluation import (
    UNTRUSTED_SUBMISSION,
    AssertionUnevaluated,
    FilteredResolver,
)
from atp.evaluators.composite import CompositeEvaluator, Verdict
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StubEvaluator:
    """Returns a fixed verdict and score."""

    def __init__(self, passed: bool, score: float) -> None:
        self._passed = passed
        self._score = score

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[Any],
        assertion: Assertion,
    ) -> EvalResult:
        return EvalResult(
            evaluator="stub",
            checks=[
                EvalCheck(name=assertion.type, passed=self._passed, score=self._score)
            ],
        )


class StubRegistry:
    """Knows every type the tests use, permitted or not."""

    def __init__(self, evaluators: dict[str, StubEvaluator]) -> None:
        self._evaluators = evaluators
        self.asked: list[str] = []

    def create_for_assertion(self, assertion_type: str) -> StubEvaluator:
        self.asked.append(assertion_type)
        if assertion_type not in self._evaluators:
            raise LookupError(f"no evaluator for '{assertion_type}'")
        return self._evaluators[assertion_type]


PASSING = StubEvaluator(True, 1.0)
FAILING = StubEvaluator(False, 0.0)
MIDDLING = StubEvaluator(True, 0.6)


def task() -> TestDefinition:
    return TestDefinition(
        id="t1",
        name="t1",
        task=TaskDefinition(description="do the thing"),
        constraints=Constraints(),
    )


def response() -> ATPResponse:
    return ATPResponse(task_id="t1", status=ResponseStatus.COMPLETED, artifacts=[])


def leaf(assertion_type: str) -> dict[str, Any]:
    return {"type": assertion_type, "config": {}}


def composite_for(
    evaluators: dict[str, StubEvaluator],
) -> tuple[CompositeEvaluator, StubRegistry]:
    """A composite restricted by the real server policy."""
    registry = StubRegistry(evaluators)
    return CompositeEvaluator(
        FilteredResolver(registry, UNTRUSTED_SUBMISSION)
    ), registry


async def run(evaluator: CompositeEvaluator, config: dict[str, Any]) -> EvalResult:
    return await evaluator.evaluate(
        task(), response(), [], Assertion(type="composite", config=config)
    )


# ----------------------------------------------------------------------
# The policy actually reaches the leaves
# ----------------------------------------------------------------------


class TestLeavesAreFiltered:
    async def test_a_withheld_leaf_is_never_constructed(self) -> None:
        """The hole this closes: `composite` wrapping `pytest`."""
        evaluator, registry = composite_for({"pytest": PASSING})
        with pytest.raises(AssertionUnevaluated):
            await run(evaluator, {"operator": "and", "conditions": [leaf("pytest")]})
        assert registry.asked == []

    async def test_a_withheld_leaf_does_not_fail_the_conjunction(self) -> None:
        """It is unknown, so the AND is unknown — not false."""
        evaluator, _ = composite_for({"contains": PASSING, "pytest": PASSING})
        with pytest.raises(AssertionUnevaluated) as exc:
            await run(
                evaluator,
                {"operator": "and", "conditions": [leaf("contains"), leaf("pytest")]},
            )
        assert "pytest" in str(exc.value)

    async def test_nesting_does_not_escape_the_policy(self) -> None:
        """A composite inside a composite is bound as its parent was."""
        evaluator, registry = composite_for({"contains": PASSING, "pytest": PASSING})
        with pytest.raises(AssertionUnevaluated):
            await run(
                evaluator,
                {
                    "operator": "and",
                    "conditions": [
                        {"operator": "or", "conditions": [leaf("pytest")]},
                    ],
                },
            )
        assert "pytest" not in registry.asked

    async def test_an_unbound_composite_evaluates_nothing(self) -> None:
        """Fail-closed: no resolver means no leaves, not a global fallback."""
        with pytest.raises(AssertionUnevaluated):
            await run(
                CompositeEvaluator(),
                {"operator": "and", "conditions": [leaf("contains")]},
            )


# ----------------------------------------------------------------------
# Kleene propagation
# ----------------------------------------------------------------------


class TestPropagation:
    async def test_and_lets_a_real_failure_decide(self) -> None:
        """A conjunction with one genuine failure is false whatever the rest is.

        Refusing to say so would be its own dishonesty — the unknown only wins
        where it could change the answer.
        """
        evaluator, _ = composite_for({"contains": FAILING, "pytest": PASSING})
        result = await run(
            evaluator,
            {"operator": "and", "conditions": [leaf("contains"), leaf("pytest")]},
        )
        assert result.passed is False

    async def test_or_lets_a_real_pass_decide(self) -> None:
        evaluator, _ = composite_for({"contains": PASSING, "pytest": PASSING})
        result = await run(
            evaluator,
            {"operator": "or", "conditions": [leaf("contains"), leaf("pytest")]},
        )
        assert result.passed is True

    async def test_or_of_fail_and_unknown_is_unknown(self) -> None:
        """The unknown could have been the one that passed."""
        evaluator, _ = composite_for({"contains": FAILING, "pytest": PASSING})
        with pytest.raises(AssertionUnevaluated):
            await run(
                evaluator,
                {"operator": "or", "conditions": [leaf("contains"), leaf("pytest")]},
            )

    async def test_not_of_unknown_is_unknown(self) -> None:
        """The negation of a thing nobody measured is still unmeasured."""
        evaluator, _ = composite_for({"pytest": PASSING})
        with pytest.raises(AssertionUnevaluated):
            await run(
                evaluator,
                {
                    "operator": "and",
                    "conditions": [
                        {"operator": "not", "condition": leaf("pytest")},
                    ],
                },
            )

    async def test_not_of_a_real_verdict_still_inverts(self) -> None:
        evaluator, _ = composite_for({"contains": FAILING})
        result = await run(
            evaluator,
            {
                "operator": "and",
                "conditions": [{"operator": "not", "condition": leaf("contains")}],
            },
        )
        assert result.passed is True


class TestThresholdInterval:
    async def test_threshold_is_decided_when_bounds_agree(self) -> None:
        """0.6 >= 0.5 is true at both ends when the leaf was measured."""
        evaluator, _ = composite_for({"contains": MIDDLING})
        result = await run(
            evaluator,
            {
                "operator": "and",
                "conditions": [
                    {
                        "operator": "threshold",
                        "value": 0.5,
                        "comparator": ">=",
                        "condition": leaf("contains"),
                    }
                ],
            },
        )
        assert result.passed is True

    async def test_threshold_over_an_unknown_is_unknown(self) -> None:
        """`score >= 0.8` has no answer when the score is anywhere in [0, 1]."""
        evaluator, _ = composite_for({"pytest": MIDDLING})
        with pytest.raises(AssertionUnevaluated):
            await run(
                evaluator,
                {
                    "operator": "and",
                    "conditions": [
                        {
                            "operator": "threshold",
                            "value": 0.8,
                            "comparator": ">=",
                            "condition": leaf("pytest"),
                        }
                    ],
                },
            )

    async def test_threshold_an_unknown_cannot_change_is_decided(self) -> None:
        """`score >= 0.0` is true across the whole interval, so it is answerable.

        This is the case a naive "any unknown means unknown" rule gets wrong,
        and it is why the decision is made over bounds rather than by counting
        unknowns.
        """
        evaluator, _ = composite_for({"pytest": MIDDLING})
        result = await run(
            evaluator,
            {
                "operator": "and",
                "conditions": [
                    {
                        "operator": "threshold",
                        "value": 0.0,
                        "comparator": ">=",
                        "condition": leaf("pytest"),
                    }
                ],
            },
        )
        assert result.passed is True


class TestPublishedScore:
    async def test_a_fully_measured_composite_reports_its_average(self) -> None:
        """Unchanged behaviour when nothing is unknown."""
        evaluator, _ = composite_for({"contains": MIDDLING, "behavior": PASSING})
        result = await run(
            evaluator,
            {"operator": "and", "conditions": [leaf("contains"), leaf("behavior")]},
        )
        assert result.checks[0].score == pytest.approx(0.8)

    async def test_a_verdict_held_despite_unknowns_reports_the_verdict(self) -> None:
        """Not an average over a number that was never measured.

        The AND is genuinely false, so 0.0 is the honest score; averaging the
        unknown leaf's [0, 1] into it would invent a measurement.
        """
        evaluator, _ = composite_for({"contains": FAILING, "pytest": PASSING})
        result = await run(
            evaluator,
            {"operator": "and", "conditions": [leaf("contains"), leaf("pytest")]},
        )
        assert result.checks[0].score == 0.0
        assert result.checks[0].details is not None
        assert result.checks[0].details["score_bounds"] != [0.0, 0.0]


def test_verdict_has_exactly_three_values() -> None:
    """A fourth would need propagation rules nobody has written."""
    assert {v.value for v in Verdict} == {"pass", "fail", "unevaluated"}
