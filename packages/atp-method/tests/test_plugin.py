"""Plugin registration + end-to-end wiring tests."""

from pathlib import Path

import pytest
from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.evaluators.registry import get_registry
from atp.loader import get_suite_source_registry
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus
from atp.scoring.aggregator import ScoreAggregator

from atp_method.evaluators import AgentEvalCaseEvaluator
from atp_method.loader import METHOD_CRITICAL_CHECK, is_agent_eval_case, load_suite
from atp_method.plugin import register


class _FailingJudge(Evaluator):
    """Judge that always scores 0.0 (the trap is triggered)."""

    @property
    def name(self) -> str:
        return "failing"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list,
        assertion: Assertion,
    ) -> EvalResult:
        return EvalResult(
            evaluator="failing",
            checks=[EvalCheck(name="x", passed=False, score=0.0)],
        )


def test_register_wires_evaluator_and_source() -> None:
    """register() makes the assertion types resolve and the source format known."""
    register()
    reg = get_registry()
    assert (
        reg.get_evaluator_for_assertion(METHOD_CRITICAL_CHECK).__name__
        == "AgentEvalCaseEvaluator"
    )
    assert "agent_eval_case" in get_suite_source_registry().names()


def test_detector_matches_example_and_rejects_native(
    clean_case_path: Path, tmp_path: Path
) -> None:
    """is_agent_eval_case detects a case file/dir and rejects a native suite."""
    assert is_agent_eval_case(clean_case_path) is True
    assert is_agent_eval_case(clean_case_path.parent) is True
    native = tmp_path / "native.yaml"
    native.write_text("test_suite: s\ntests: []\n")
    assert is_agent_eval_case(native) is False


def test_load_suite_file_and_sweep(
    clean_case_path: Path, example_cases_dir: Path
) -> None:
    """A single file loads one test; a directory loads the whole sweep."""
    assert len(load_suite(clean_case_path).tests) == 1
    # The req-extraction sweep: four text-out severity cases plus one corpus case.
    assert len(load_suite(example_cases_dir).tests) == 5


@pytest.mark.anyio
async def test_end_to_end_load_evaluate_hard_gate(clean_case_path: Path) -> None:
    """Load via the source registry, evaluate the critical check, hard-gate fails.

    Exercises the real chain: source registry → loaded TestSuite → the assertion
    types the loader emits → AgentEvalCaseEvaluator → ScoreAggregator hard gate.
    """
    register()
    source_loader = get_suite_source_registry().find_loader(clean_case_path)
    assert source_loader is not None
    suite = source_loader(clean_case_path)
    test = suite.tests[0]
    critical = next(a for a in test.assertions if a.type == METHOD_CRITICAL_CHECK)

    evaluator = AgentEvalCaseEvaluator(judge=_FailingJudge())
    response = ATPResponse(task_id=test.id, status=ResponseStatus.COMPLETED)
    result = await evaluator.evaluate(test, response, [], critical)
    # The CLI propagates the assertion's critical flag onto the result.
    result.critical = critical.critical

    scored = ScoreAggregator().score_test_result(test.id, [result])
    assert scored.passed is False
    assert scored.score == 0.0
