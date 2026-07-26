"""Characterization + policy tests for the extracted evaluation pipeline.

The behaviours locked here are the ones the inline CLI implementation had
(`atp/cli/main.py`, pre-extraction): guardrails short-circuit the whole test,
a failing evaluator is a warning rather than a crash, and the assertion's
`critical` flag is propagated onto the result so scoring can hard-fail.

Everything else is new and exists because the server plane runs on untrusted
submissions: a policy decides what may run, a disallowed evaluator is never
constructed, an unknown assertion type is reported rather than guessed at, and
the outcome distinguishes "assessed" from "not assessed" so an unscored
submission cannot read as zero quality.
"""

from __future__ import annotations

from typing import Any

import pytest

from atp.core.results import EvalCheck, EvalResult
from atp.evaluation import (
    EvaluationPipeline,
    EvaluationPolicy,
    SkipReason,
)
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse


def _test_def(*assertions: Assertion) -> TestDefinition:
    """A minimal test definition carrying the given assertions."""
    return TestDefinition(
        id="t1",
        name="t1",
        task=TaskDefinition(description="do the thing"),
        assertions=list(assertions),
    )


def _response() -> ATPResponse:
    """A minimal completed response."""
    return ATPResponse(task_id="t1", status="completed", artifacts=[])


class RecordingEvaluator:
    """Evaluator that records being run and returns a passing result."""

    def __init__(self, name: str = "rec") -> None:
        self.name = name
        self.calls = 0

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[Any],
        assertion: Assertion,
    ) -> EvalResult:
        self.calls += 1
        # `passed`/`score` are computed from checks, not fields: passing them
        # as kwargs only worked because pydantic dropped the extras.
        return EvalResult(
            evaluator=self.name,
            checks=[EvalCheck(name="ok", passed=True, score=1.0)],
        )


class ExplodingEvaluator:
    """Evaluator that fails the way a broken one does in production."""

    async def evaluate(self, *args: Any, **kwargs: Any) -> EvalResult:
        raise RuntimeError("boom")


class Resolver:
    """Test double for the platform registry."""

    def __init__(self, **evaluators: Any) -> None:
        self._evaluators = evaluators
        self.resolved: list[str] = []

    def create_for_assertion(self, assertion_type: str) -> Any:
        self.resolved.append(assertion_type)
        if assertion_type not in self._evaluators:
            raise ValueError(f"no evaluator for assertion type: {assertion_type}")
        return self._evaluators[assertion_type]


pytestmark = pytest.mark.anyio

TRUSTED = EvaluationPolicy(name="trusted_local")
RESTRICTED = EvaluationPolicy(
    name="untrusted_submission", allowed_assertion_types=frozenset({"contains"})
)


class TestCharacterizationOfPriorBehaviour:
    """What the inline CLI implementation did, now locked in one place."""

    async def test_assertions_are_evaluated_and_collected(self) -> None:
        evaluator = RecordingEvaluator()
        pipeline = EvaluationPipeline(Resolver(contains=evaluator), TRUSTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains")), _response(), []
        )
        assert evaluator.calls == 1
        assert len(outcome.results) == 1
        assert outcome.applied == ["contains"]

    async def test_critical_flag_is_propagated_onto_the_result(self) -> None:
        """Scoring hard-fails a test on a failed critical assertion."""
        pipeline = EvaluationPipeline(Resolver(contains=RecordingEvaluator()), TRUSTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains", critical=True)), _response(), []
        )
        assert outcome.results[0].critical is True

    async def test_a_failing_evaluator_does_not_abort_the_others(self) -> None:
        pipeline = EvaluationPipeline(
            Resolver(contains=RecordingEvaluator(), schema=ExplodingEvaluator()),
            TRUSTED,
        )
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="schema"), Assertion(type="contains")),
            _response(),
            [],
        )
        assert outcome.applied == ["contains"]
        assert outcome.skipped[0].reason == SkipReason.EVALUATOR_ERROR

    async def test_guardrail_skips_the_whole_test(self) -> None:
        evaluator = RecordingEvaluator()
        pipeline = EvaluationPipeline(
            Resolver(contains=evaluator),
            TRUSTED,
            guardrail=lambda task, response: "budget exceeded",
        )
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains")), _response(), []
        )
        assert evaluator.calls == 0
        assert outcome.results == []
        assert outcome.skipped[0].reason == SkipReason.GUARDRAIL
        assert outcome.skipped[0].detail == "budget exceeded"

    async def test_no_assertions_is_not_an_error(self) -> None:
        outcome = await EvaluationPipeline(Resolver(), TRUSTED).evaluate(
            _test_def(), _response(), []
        )
        assert outcome.results == [] and outcome.skipped == []


class TestPolicyEnforcement:
    """The server plane decides what may run — not the submission."""

    async def test_disallowed_evaluator_is_never_constructed(self) -> None:
        """Refusing after construction would defeat the point of refusing."""
        resolver = Resolver(code_exec=RecordingEvaluator())
        pipeline = EvaluationPipeline(resolver, RESTRICTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="code_exec")), _response(), []
        )
        assert resolver.resolved == []
        assert outcome.skipped[0].reason == SkipReason.NOT_ALLOWED

    async def test_allowed_evaluator_still_runs_under_a_restrictive_policy(
        self,
    ) -> None:
        evaluator = RecordingEvaluator()
        pipeline = EvaluationPipeline(Resolver(contains=evaluator), RESTRICTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains")), _response(), []
        )
        assert evaluator.calls == 1 and outcome.applied == ["contains"]

    async def test_unknown_assertion_type_is_reported_not_guessed(self) -> None:
        pipeline = EvaluationPipeline(Resolver(), TRUSTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="no_such_thing")), _response(), []
        )
        assert outcome.results == []
        assert outcome.skipped[0].reason == SkipReason.UNSUPPORTED

    async def test_mixed_suite_records_both_sides(self) -> None:
        """A consumer must see what was assessed and what was not."""
        pipeline = EvaluationPipeline(
            Resolver(contains=RecordingEvaluator(), code_exec=RecordingEvaluator()),
            RESTRICTED,
        )
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains"), Assertion(type="code_exec")),
            _response(),
            [],
        )
        assert outcome.applied == ["contains"]
        assert [s.assertion_type for s in outcome.skipped] == ["code_exec"]


class TestQualitySignal:
    """Absence of assessment must not look like an assessment of zero."""

    async def test_nothing_ran_means_no_quality_signal(self) -> None:
        pipeline = EvaluationPipeline(Resolver(), RESTRICTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="code_exec")), _response(), []
        )
        assert outcome.quality_evaluated is False

    async def test_one_real_evaluator_is_enough(self) -> None:
        pipeline = EvaluationPipeline(Resolver(contains=RecordingEvaluator()), TRUSTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains")), _response(), []
        )
        assert outcome.quality_evaluated is True

    async def test_an_evaluator_that_errored_does_not_count_as_assessed(self) -> None:
        pipeline = EvaluationPipeline(Resolver(contains=ExplodingEvaluator()), TRUSTED)
        outcome = await pipeline.evaluate(
            _test_def(Assertion(type="contains")), _response(), []
        )
        assert outcome.quality_evaluated is False


class TestArtifactContextIsInjected:
    """Filesystem policy belongs to the caller, not to core."""

    async def test_context_wraps_evaluation_and_always_exits(self) -> None:
        import contextlib

        events: list[str] = []

        @contextlib.contextmanager
        def tracking(response: ATPResponse):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        pipeline = EvaluationPipeline(
            Resolver(contains=ExplodingEvaluator()), TRUSTED, artifacts=tracking
        )
        await pipeline.evaluate(_test_def(Assertion(type="contains")), _response(), [])
        assert events == ["enter", "exit"]

    async def test_context_is_not_entered_when_the_guardrail_skips(self) -> None:
        import contextlib

        entered = False

        @contextlib.contextmanager
        def tracking(response: ATPResponse):
            nonlocal entered
            entered = True
            yield

        pipeline = EvaluationPipeline(
            Resolver(),
            TRUSTED,
            guardrail=lambda task, response: "nope",
            artifacts=tracking,
        )
        await pipeline.evaluate(_test_def(Assertion(type="contains")), _response(), [])
        assert entered is False


@pytest.mark.parametrize(
    ("assertion_type", "expected_deterministic"),
    [
        ("contains", True),
        ("behavior", True),
        ("findings_match", True),
        ("pytest", False),
        ("code_exec", False),
        ("llm_eval", False),
        ("factuality", False),
    ],
)
def test_vocabulary_classifies_by_what_the_evaluator_does(
    assertion_type: str, expected_deterministic: bool
) -> None:
    """Classification is by behaviour, so a new executing evaluator is caught."""
    from atp.evaluation import deterministic_assertion_types

    assert (assertion_type in deterministic_assertion_types()) is expected_deterministic
