"""The benchmark plane's score, and the label that says what it is.

The table these tests encode was ratified before the implementation existed
(`docs/superpowers/plans/2026-07-26-step6-score-semantics.md`), so the contract
is not a description of whatever the code turned out to do:

| successful quality evaluators | kind                  | quality_signal | components |
|-------------------------------|-----------------------|----------------|------------|
| 0                             | completion_rate       | false          | {}         |
| >=1                           | aggregated_evaluation | true           | measured   |
| some applied, some skipped    | aggregated_evaluation | true           | measured   |

The row that matters most is the first one, because it is the row a wired
server keeps landing on: no assertions, an incomplete response, or every
assertion withheld by policy. Publishing `aggregated_evaluation` for any of
those would be a quality claim backed by nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from atp.core.results import EvalCheck, EvalResult
from atp.dashboard.benchmark.score_contract import (
    AGGREGATED_EVALUATION,
    COMPLETION_RATE,
)
from atp.dashboard.benchmark.scoring import (
    COMPLETION_SCORE,
    INCOMPLETE_SCORE,
    RecordStatus,
    derive_run_score_view,
    score_submission,
)
from atp.evaluation import (
    UNTRUSTED_SUBMISSION,
    EvaluationPolicy,
    EvaluatorLike,
    FilteredResolver,
    SkipReason,
)
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StubEvaluator:
    """An evaluator that returns a score chosen by the test."""

    def __init__(self, name: str, score: float, *, explode: bool = False) -> None:
        self._name = name
        self._score = score
        self._explode = explode

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[Any],
        assertion: Assertion,
    ) -> EvalResult:
        if self._explode:
            raise RuntimeError("evaluator blew up")
        return EvalResult(
            evaluator=self._name,
            checks=[
                EvalCheck(
                    name=assertion.type, passed=self._score >= 0.5, score=self._score
                )
            ],
        )


class StubRegistry:
    """Resolves the assertion types the test set up, and nothing else."""

    def __init__(self, evaluators: dict[str, StubEvaluator]) -> None:
        self._evaluators = evaluators

    def create_for_assertion(self, assertion_type: str) -> EvaluatorLike:
        if assertion_type not in self._evaluators:
            raise LookupError(f"no evaluator for '{assertion_type}'")
        return self._evaluators[assertion_type]


def resolver_for(
    evaluators: dict[str, StubEvaluator],
    policy: EvaluationPolicy = UNTRUSTED_SUBMISSION,
) -> FilteredResolver:
    return FilteredResolver(StubRegistry(evaluators), policy)


def make_test_def(*assertion_types: str) -> TestDefinition:
    return TestDefinition(
        id="t1",
        name="t1",
        task=TaskDefinition(description="do the thing"),
        assertions=[Assertion(type=t, config={}) for t in assertion_types],
    )


def completed(**kwargs: Any) -> ATPResponse:
    return ATPResponse(task_id="task-1", status="completed", **kwargs)


# ----------------------------------------------------------------------
# Per-task scoring
# ----------------------------------------------------------------------


class TestScoreSubmission:
    async def test_incomplete_response_is_not_evaluated(self) -> None:
        """An errored submission has nothing to assess.

        Running evaluators over it would report failed assertions as if the
        agent had produced something to fail on.
        """
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 1.0)}),
            make_test_def("contains"),
            ATPResponse(task_id="task-1", status="failed", error="boom"),
            [],
        )
        assert scored.score == INCOMPLETE_SCORE
        assert scored.quality_evaluated is False
        assert scored.records is None

    async def test_completion_only_deployment_scores_completion(self) -> None:
        """No resolver: exactly the behaviour the plane had before this step."""
        scored = await score_submission(
            None, make_test_def("contains"), completed(), []
        )
        assert scored.score == COMPLETION_SCORE
        assert scored.quality_evaluated is False
        assert scored.records is None

    async def test_suite_without_assertions_scores_completion(self) -> None:
        """A wired server still measures completion when nothing was asserted."""
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 1.0)}),
            make_test_def(),
            completed(),
            [],
        )
        assert scored.quality_evaluated is False
        assert scored.records is None

    async def test_missing_test_definition_scores_completion(self) -> None:
        """A submission for an index the suite lacks is malformed, not a 500."""
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 1.0)}),
            None,
            completed(),
            [],
        )
        assert scored.score == COMPLETION_SCORE
        assert scored.quality_evaluated is False

    async def test_applied_evaluator_produces_a_quality_score(self) -> None:
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 1.0)}),
            make_test_def("contains"),
            completed(),
            [],
        )
        assert scored.quality_evaluated is True
        assert scored.records is not None
        record = scored.records[0]
        assert record["status"] == RecordStatus.APPLIED
        assert record["assertion_type"] == "contains"
        assert record["evaluator"] == "artifact"
        assert record["score"] == 1.0

    async def test_a_low_score_is_recorded_as_a_low_score(self) -> None:
        """The failing direction has to work too, or the wiring proves nothing."""
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 0.0)}),
            make_test_def("contains"),
            completed(),
            [],
        )
        assert scored.quality_evaluated is True
        assert scored.score < COMPLETION_SCORE
        assert scored.records is not None
        assert scored.records[0]["passed"] is False

    async def test_policy_refusal_is_recorded_not_scored_zero(self) -> None:
        """A withheld evaluator must not look like a measurement of zero."""
        scored = await score_submission(
            resolver_for({"pytest": StubEvaluator("code_exec", 1.0)}),
            make_test_def("pytest"),
            completed(),
            [],
        )
        assert scored.quality_evaluated is False
        assert scored.score == COMPLETION_SCORE
        assert scored.records is not None
        assert scored.records[0]["status"] == RecordStatus.SKIPPED
        assert scored.records[0]["reason"] == SkipReason.NOT_ALLOWED

    async def test_forbidden_evaluator_is_never_constructed(self) -> None:
        """Policy is checked before resolution, so the class never loads."""
        asked: list[str] = []

        class SpyRegistry(StubRegistry):
            def create_for_assertion(self, assertion_type: str) -> EvaluatorLike:
                asked.append(assertion_type)
                return super().create_for_assertion(assertion_type)

        resolver = FilteredResolver(
            SpyRegistry({"pytest": StubEvaluator("code_exec", 1.0)}),
            UNTRUSTED_SUBMISSION,
        )
        await score_submission(resolver, make_test_def("pytest"), completed(), [])
        assert asked == []

    async def test_evaluator_crash_is_a_skip_with_a_reason(self) -> None:
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 1.0, explode=True)}),
            make_test_def("contains"),
            completed(),
            [],
        )
        assert scored.quality_evaluated is False
        assert scored.records is not None
        assert scored.records[0]["reason"] == SkipReason.EVALUATOR_ERROR

    async def test_unknown_assertion_type_is_a_skip_not_a_default_evaluator(
        self,
    ) -> None:
        """Scoring by an evaluator nobody chose is worse than not scoring."""
        scored = await score_submission(
            resolver_for({}),
            make_test_def("contains"),
            completed(),
            [],
        )
        assert scored.quality_evaluated is False
        assert scored.records is not None
        assert scored.records[0]["reason"] == SkipReason.UNSUPPORTED

    async def test_partial_evaluation_keeps_both_halves(self) -> None:
        scored = await score_submission(
            resolver_for({"contains": StubEvaluator("artifact", 1.0)}),
            make_test_def("contains", "pytest"),
            completed(),
            [],
        )
        assert scored.quality_evaluated is True
        assert scored.records is not None
        statuses = {r["assertion_type"]: r["status"] for r in scored.records}
        assert statuses == {
            "contains": RecordStatus.APPLIED,
            "pytest": RecordStatus.SKIPPED,
        }


# ----------------------------------------------------------------------
# Run-level semantics
# ----------------------------------------------------------------------


APPLIED = {
    "assertion_type": "contains",
    "status": RecordStatus.APPLIED,
    "evaluator": "artifact",
    "score": 1.0,
    "passed": True,
    "critical": False,
    "checks": [],
}
SKIPPED = {
    "assertion_type": "pytest",
    "status": RecordStatus.SKIPPED,
    "reason": SkipReason.NOT_ALLOWED,
    "detail": "policy 'untrusted_submission' does not permit this evaluator",
}


class TestRunSemantics:
    def test_no_evaluation_is_labelled_completion(self) -> None:
        """Row 1: the label a completion-only run has always deserved."""
        semantics, components = derive_run_score_view([None, None], tasks_total=2)
        assert semantics["kind"] == COMPLETION_RATE
        assert semantics["quality_signal"] is False
        assert components == {}

    def test_evaluated_run_is_labelled_aggregated(self) -> None:
        """Row 2."""
        semantics, components = derive_run_score_view([[APPLIED]], tasks_total=1)
        assert semantics["kind"] == AGGREGATED_EVALUATION
        assert semantics["quality_signal"] is True
        assert components == {"contains": 100.0}

    def test_only_skips_is_still_completion(self) -> None:
        """Requested but never applied is row 1, not row 2.

        This is the case a "capability implies quality" bug lands on: the
        suite asked, the server was wired, and still nothing ran.
        """
        semantics, components = derive_run_score_view([[SKIPPED]], tasks_total=1)
        assert semantics["kind"] == COMPLETION_RATE
        assert semantics["quality_signal"] is False
        assert components == {}

    def test_skipped_evaluator_never_becomes_a_zero_component(self) -> None:
        semantics, components = derive_run_score_view(
            [[APPLIED, SKIPPED]], tasks_total=1
        )
        assert "pytest" not in components
        assert components == {"contains": 100.0}
        assert semantics["coverage"]["assertions_skipped"] == [
            {
                "assertion_type": "pytest",
                "reason": SkipReason.NOT_ALLOWED,
                "count": 1,
            }
        ]

    def test_mixed_run_carries_a_caveat(self) -> None:
        """Invariant 2: a blend of two quantities never travels unlabelled."""
        semantics, _ = derive_run_score_view([[APPLIED], None], tasks_total=2)
        assert semantics["coverage"]["tasks_evaluated"] == 1
        assert semantics["coverage"]["tasks_completion_only"] == 1
        assert any(c.startswith("mixed_task_scores") for c in semantics["caveats"])

    def test_uniformly_evaluated_run_has_no_mixed_caveat(self) -> None:
        semantics, _ = derive_run_score_view([[APPLIED], [APPLIED]], tasks_total=2)
        assert not any(c.startswith("mixed_task_scores") for c in semantics["caveats"])

    def test_components_average_across_tasks(self) -> None:
        half = {**APPLIED, "score": 0.5}
        _, components = derive_run_score_view([[APPLIED], [half]], tasks_total=2)
        assert components == {"contains": 75.0}

    def test_components_are_keyed_by_assertion_type_not_evaluator(self) -> None:
        """Several assertion types share one evaluator; merging loses signal."""
        sections = {**APPLIED, "assertion_type": "sections", "score": 0.5}
        _, components = derive_run_score_view([[APPLIED, sections]], tasks_total=1)
        assert components == {"contains": 100.0, "sections": 50.0}

    def test_serialization_is_order_independent(self) -> None:
        """Invariant 8: a fixture diff means a semantic change."""
        sections = {**APPLIED, "assertion_type": "sections", "score": 0.5}
        forward, _ = derive_run_score_view([[APPLIED, sections]], tasks_total=1)
        backward, _ = derive_run_score_view([[sections, APPLIED]], tasks_total=1)
        assert forward == backward

    def test_coverage_counts_reflect_the_suite_not_the_submissions(self) -> None:
        semantics, _ = derive_run_score_view([[APPLIED]], tasks_total=5)
        assert semantics["coverage"]["tasks_total"] == 5
        assert semantics["coverage"]["tasks_submitted"] == 1

    def test_every_run_carries_a_schema_version(self) -> None:
        """A consumer that cannot version the meaning cannot use it safely."""
        semantics, _ = derive_run_score_view([[APPLIED]], tasks_total=1)
        assert semantics["schema_version"] == 1
