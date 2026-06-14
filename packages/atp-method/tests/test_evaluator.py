"""Tests for AgentEvalCaseEvaluator (critical_check + rubric grading).

The underlying LLM judge is faked, so no real model calls are made.
"""

import pytest
from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus
from atp.protocol.models import ArtifactFile

from atp_method.evaluators import AgentEvalCaseEvaluator
from atp_method.loader import METHOD_CRITICAL_CHECK, METHOD_RUBRIC


class FakeJudge(Evaluator):
    """A judge that returns a fixed score for every synthetic llm_eval."""

    def __init__(self, score: float) -> None:
        self._score = score

    @property
    def name(self) -> str:
        return "fake_judge"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list,
        assertion: Assertion,
    ) -> EvalResult:
        threshold = assertion.config.get("threshold", 0.7)
        return EvalResult(
            evaluator="fake_judge",
            checks=[
                EvalCheck(
                    name="llm_eval",
                    passed=self._score >= threshold,
                    score=self._score,
                    message="fake",
                )
            ],
        )


def _task() -> TestDefinition:
    return TestDefinition(
        id="case-x-001",
        name="x",
        task=TaskDefinition(description="do x"),
    )


def _response() -> ATPResponse:
    return ATPResponse(task_id="case-x-001", status=ResponseStatus.COMPLETED)


@pytest.mark.anyio
async def test_critical_check_passes_on_high_score() -> None:
    """A high judge score passes the critical check."""
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(1.0))
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={"check": "no fabricated value", "expected_failure_mode": "fabricates"},
    )
    result = await ev.evaluate(_task(), _response(), [], assertion)
    assert result.checks[0].name == "critical_check"
    assert result.checks[0].passed is True
    assert result.checks[0].score == 1.0


@pytest.mark.anyio
async def test_critical_check_fails_on_low_score() -> None:
    """A low judge score fails the critical check (which hard-gates upstream)."""
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(0.0))
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={"check": "no fabricated value", "expected_failure_mode": "fabricates"},
    )
    result = await ev.evaluate(_task(), _response(), [], assertion)
    assert result.checks[0].passed is False
    assert result.checks[0].score == 0.0


@pytest.mark.anyio
async def test_rubric_is_weighted() -> None:
    """The rubric score is the weight-weighted sum of per-criterion scores."""
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(0.8))
    assertion = Assertion(
        type=METHOD_RUBRIC,
        config={
            "rubric": [
                {"criterion": "c1", "weight": 0.3},
                {"criterion": "c2", "weight": 0.7},
            ]
        },
    )
    result = await ev.evaluate(_task(), _response(), [], assertion)
    # 0.3*0.8 + 0.7*0.8 = 0.8
    assert result.checks[0].score == pytest.approx(0.8)
    assert result.checks[0].passed is True
    assert len(result.checks[0].details["items"]) == 2


@pytest.mark.anyio
async def test_empty_rubric_passes() -> None:
    """A rubric assertion with no criteria is an inert pass."""
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(0.0))
    assertion = Assertion(type=METHOD_RUBRIC, config={"rubric": []})
    result = await ev.evaluate(_task(), _response(), [], assertion)
    assert result.checks[0].passed is True
    assert result.checks[0].score == 1.0


@pytest.mark.anyio
async def test_unsupported_assertion_type() -> None:
    """An unknown assertion type yields a failed check, not a crash."""
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(1.0))
    assertion = Assertion(type="something_else", config={})
    result = await ev.evaluate(_task(), _response(), [], assertion)
    assert result.checks[0].passed is False


class RecordingJudge(Evaluator):
    """Captures the prompts it is asked to grade; can simulate a judge failure."""

    def __init__(self, score: float = 1.0, return_empty: bool = False) -> None:
        self.prompts: list[str] = []
        self._score = score
        self._return_empty = return_empty

    @property
    def name(self) -> str:
        return "recording"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list,
        assertion: Assertion,
    ) -> EvalResult:
        self.prompts.append(assertion.config.get("prompt", ""))
        if self._return_empty:
            return EvalResult(evaluator="recording", checks=[])
        return EvalResult(
            evaluator="recording",
            checks=[EvalCheck(name="x", passed=True, score=self._score)],
        )


@pytest.mark.anyio
async def test_rubric_normalizes_by_weight_sum() -> None:
    """Weights that don't sum to 1 are normalized, not clamped."""
    ev = AgentEvalCaseEvaluator(judge=FakeJudge(0.6))
    assertion = Assertion(
        type=METHOD_RUBRIC,
        config={
            "rubric": [
                {"criterion": "c1", "weight": 1.0},
                {"criterion": "c2", "weight": 1.0},
            ]
        },
    )
    result = await ev.evaluate(_task(), _response(), [], assertion)
    # (1*0.6 + 1*0.6) / 2 = 0.6 — not clamped to 1.0
    assert result.checks[0].score == pytest.approx(0.6)


@pytest.mark.anyio
async def test_rubric_judge_failure_flagged() -> None:
    """A per-criterion judge failure is recorded in details, not silent."""
    ev = AgentEvalCaseEvaluator(judge=RecordingJudge(return_empty=True))
    assertion = Assertion(
        type=METHOD_RUBRIC, config={"rubric": [{"criterion": "c1", "weight": 1.0}]}
    )
    result = await ev.evaluate(_task(), _response(), [], assertion)
    item = result.checks[0].details["items"][0]
    assert item["judge_failed"] is True
    assert item["score"] == 0.0


@pytest.mark.anyio
async def test_critical_prompt_includes_gold() -> None:
    """When the case supplies a gold reference, it reaches the judge prompt."""
    judge = RecordingJudge(score=1.0)
    ev = AgentEvalCaseEvaluator(judge=judge)
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={
            "check": "c",
            "expected_failure_mode": "f",
            "gold": "the reference answer",
        },
    )
    await ev.evaluate(_task(), _response(), [], assertion)
    assert any("the reference answer" in p for p in judge.prompts)


class BombJudge(Evaluator):
    """Judge that raises if called — used to assert the matcher path never calls it."""

    @property
    def name(self) -> str:
        return "bomb"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list,
        assertion: Assertion,
    ) -> EvalResult:
        raise AssertionError("Judge must NOT be called for findings_match grader_type")


@pytest.mark.anyio
async def test_findings_match_critical_uses_matcher_not_judge() -> None:
    """findings_match grader_type resolved deterministically; judge is never called."""
    ev = AgentEvalCaseEvaluator(judge=BombJudge())
    artifact = ArtifactFile(
        path="findings.json",
        content='[{"rule_id":"cwe-89","anchor":"f\\"SELECT","severity":"critical"}]',
    )
    response = ATPResponse(
        task_id="case-x-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[artifact],
    )
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={
            "check": "flag the injection",
            "grader_type": "findings_match",
            "expected_findings": [
                {
                    "rule_ids": ["SEC-011", "cwe-89"],
                    "anchor": 'f"SELECT',
                    "severity": "critical",
                }
            ],
            "must_not_flag": [],
        },
    )
    result = await ev.evaluate(_task(), response, [], assertion)
    assert result.checks[0].name == "critical_check"
    assert result.checks[0].passed is True


@pytest.mark.anyio
async def test_findings_match_malformed_is_distinct_from_missed() -> None:
    """A finding missing the required severity is malformed (not a silent miss):
    critical_check fails AND details.malformed is True for the routing signal."""
    ev = AgentEvalCaseEvaluator(judge=BombJudge())
    response = ATPResponse(
        task_id="case-x-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="findings.json",
                content='[{"rule_id":"cwe-89","anchor":"f\\"SELECT"}]',
            )
        ],
    )
    assertion = Assertion(
        type=METHOD_CRITICAL_CHECK,
        critical=True,
        config={
            "check": "flag the injection",
            "grader_type": "findings_match",
            "expected_findings": [
                {"rule_ids": ["cwe-89"], "anchor": 'f"SELECT', "severity": "critical"}
            ],
            "must_not_flag": [],
        },
    )
    result = await ev.evaluate(_task(), response, [], assertion)
    check = result.checks[0]
    assert check.passed is False
    assert check.details is not None
    assert check.details["malformed"] is True
    assert "malformed" in (check.message or "")
