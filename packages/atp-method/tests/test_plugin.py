"""Plugin registration + end-to-end wiring tests."""

import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest
from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.evaluators.registry import get_registry
from atp.loader import get_suite_source_registry
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol import ATPRequest, ATPResponse, ResponseStatus, Task
from atp.runner.preparation import PreparedRequest, get_request_preparer
from atp.scoring.aggregator import ScoreAggregator

from atp_method.evaluators import AgentEvalCaseEvaluator
from atp_method.loader import (
    METHOD_CRITICAL_CHECK,
    METHOD_RUBRIC,
    is_agent_eval_case,
    load_suite,
)
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


def test_register_does_not_import_corpus_runtime() -> None:
    """register() avoids optional mock-server runtime dependencies."""
    sys.modules.pop("atp_method.runtime", None)

    register()

    assert "atp_method.runtime" not in sys.modules


def test_register_wires_evaluator_source_and_corpus_preparer() -> None:
    """register() wires assertion types, source format, and corpus preparation."""
    register()
    reg = get_registry()
    assert (
        reg.get_evaluator_for_assertion(METHOD_CRITICAL_CHECK).__name__
        == "AgentEvalCaseEvaluator"
    )
    assert (
        reg.get_evaluator_for_assertion(METHOD_RUBRIC).__name__
        == "AgentEvalCaseEvaluator"
    )
    assert "agent_eval_case" in get_suite_source_registry().names()
    assert get_request_preparer("corpus") is not None


@pytest.mark.anyio
async def test_registered_corpus_preparer_imports_runtime_when_prepare_called() -> None:
    """The corpus preparer imports CorpusRunPreparer lazily inside prepare()."""
    sys.modules.pop("atp_method.runtime", None)
    register()
    preparer = get_request_preparer("corpus")
    assert preparer is not None
    assert "atp_method.runtime" not in sys.modules

    class _RuntimePreparer:
        async def prepare(
            self, test: TestDefinition, request: ATPRequest
        ) -> PreparedRequest:
            request.metadata = {"runtime_preparer_called": True}
            return PreparedRequest(request=request)

    runtime_module = types.ModuleType("atp_method.runtime")
    runtime_module.CorpusRunPreparer = _RuntimePreparer
    sys.modules["atp_method.runtime"] = runtime_module

    test = TestDefinition(
        id="t1",
        name="corpus",
        task=TaskDefinition(description="extract"),
    )
    request = ATPRequest(task_id="t1", task=Task(description="extract"))

    prepared = await preparer.prepare(test, request)

    assert prepared.request.metadata == {"runtime_preparer_called": True}


@pytest.mark.anyio
async def test_registered_corpus_preparer_missing_runtime_dep_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing mock-server dependencies are explained when corpus prepare runs."""
    sys.modules.pop("atp_method.runtime", None)
    register()
    preparer = get_request_preparer("corpus")
    assert preparer is not None
    assert "atp_method.runtime" not in sys.modules

    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "atp_method.runtime":
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import(name, globals, locals, fromlist, level)

    def fail_runtime_import_module(name: str, package: str | None = None):
        if name == "atp_method.runtime":
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", fail_runtime_import)
    monkeypatch.setattr(importlib, "import_module", fail_runtime_import_module)

    test = TestDefinition(
        id="t1",
        name="corpus",
        task=TaskDefinition(description="extract"),
    )
    request = ATPRequest(task_id="t1", task=Task(description="extract"))

    with pytest.raises(
        RuntimeError,
        match="corpus-backed method cases require.*fastapi.*uvicorn",
    ):
        await preparer.prepare(test, request)


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
    # The req-extraction sweep: 13 inline text-out cases (deadline/actor/condition
    # trap families across a clean..very_severe axis) plus the 4-case
    # read-only corpus severity ladder.
    assert len(load_suite(example_cases_dir).tests) == 17


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
