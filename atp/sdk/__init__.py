"""ATP Python SDK for programmatic test execution.

Provides a clean Python API for running ATP test suites, evaluating
agent responses, and loading test definitions without going through
the CLI. Designed for use in Jupyter notebooks, custom pipelines,
and programmatic integrations.

Example usage::

    import atp.sdk as sdk

    # Load and run a test suite
    suite = sdk.load_suite("tests/my_suite.yaml")
    results = await sdk.arun(suite, adapter="http", adapter_config={"base_url": "..."})

    # Evaluate a single response
    eval_results = await sdk.aevaluate(response, test, assertions)
"""

import asyncio
from pathlib import Path
from typing import Any

from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.adapters.registry import create_adapter
from atp.evaluators.base import EvalResult
from atp.evaluators.registry import get_registry as get_evaluator_registry
from atp.loader.loader import TestLoader
from atp.loader.models import Assertion, TestDefinition, TestSuite
from atp.protocol import ATPEvent, ATPResponse
from atp.runner.models import SuiteResult, TestResult
from atp.runner.orchestrator import TestOrchestrator
from atp.scoring.aggregator import ScoreAggregator
from atp.scoring.models import ScoredTestResult

__all__ = [
    # Functions
    "load_suite",
    "load_suite_string",
    "run",
    "arun",
    "evaluate",
    "aevaluate",
    "score",
    # Re-exported types
    "TestSuite",
    "TestDefinition",
    "Assertion",
    "ATPResponse",
    "ATPEvent",
    "EvalResult",
    "SuiteResult",
    "TestResult",
    "ScoredTestResult",
    "AgentAdapter",
    "AdapterConfig",
]


def load_suite(
    path: str | Path,
    env: dict[str, str] | None = None,
) -> TestSuite:
    """Load a test suite from a YAML file.

    Args:
        path: Path to the test suite YAML file.
        env: Custom environment variables for substitution.

    Returns:
        Parsed and validated TestSuite object.

    Raises:
        atp.core.exceptions.ParseError: If YAML parsing fails.
        atp.core.exceptions.ValidationError: If validation fails.
    """
    loader = TestLoader(env=env)
    return loader.load_file(path)


def load_suite_string(
    content: str,
    env: dict[str, str] | None = None,
) -> TestSuite:
    """Load a test suite from a YAML string.

    Args:
        content: YAML content as a string.
        env: Custom environment variables for substitution.

    Returns:
        Parsed and validated TestSuite object.

    Raises:
        atp.core.exceptions.ParseError: If YAML parsing fails.
        atp.core.exceptions.ValidationError: If validation fails.
    """
    loader = TestLoader(env=env)
    return loader.load_string(content)


def _resolve_adapter(
    adapter: str | AgentAdapter,
    adapter_config: dict[str, Any] | AdapterConfig | None = None,
) -> AgentAdapter:
    """Resolve an adapter from a type string or instance.

    Args:
        adapter: Adapter type string (e.g., "http", "cli") or an
            AgentAdapter instance.
        adapter_config: Configuration for the adapter. Ignored if
            adapter is already an instance.

    Returns:
        An AgentAdapter instance.
    """
    if isinstance(adapter, AgentAdapter):
        return adapter
    config = adapter_config if adapter_config is not None else {}
    return create_adapter(adapter, config)


async def arun(
    suite: TestSuite | str | Path,
    adapter: str | AgentAdapter,
    adapter_config: dict[str, Any] | AdapterConfig | None = None,
    agent_name: str = "default",
    runs_per_test: int = 1,
    fail_fast: bool = False,
    parallel_tests: bool = False,
    max_parallel_tests: int = 5,
    tag_filter: str | None = None,
    env: dict[str, str] | None = None,
) -> SuiteResult:
    """Run a test suite asynchronously.

    Args:
        suite: A TestSuite object, or path to a YAML file.
        adapter: Adapter type string (e.g., "http", "cli") or an
            AgentAdapter instance.
        adapter_config: Configuration dict or AdapterConfig for the
            adapter. Ignored if adapter is already an instance.
        agent_name: Name of the agent being tested.
        runs_per_test: Number of times to run each test.
        fail_fast: Stop on first test failure.
        parallel_tests: Run tests in parallel.
        max_parallel_tests: Maximum number of parallel tests.
        tag_filter: Filter tests by tags (e.g., "smoke,!slow").
        env: Custom environment variables for suite loading.

    Returns:
        SuiteResult with all test results.
    """
    if isinstance(suite, (str, Path)):
        suite = load_suite(suite, env=env)

    if tag_filter:
        suite = suite.filter_by_tags(tag_filter)

    resolved_adapter = _resolve_adapter(adapter, adapter_config)

    async with TestOrchestrator(
        adapter=resolved_adapter,
        runs_per_test=runs_per_test,
        fail_fast=fail_fast,
        parallel_tests=parallel_tests,
        max_parallel_tests=max_parallel_tests,
    ) as orchestrator:
        return await orchestrator.run_suite(suite, agent_name, runs_per_test)


def run(
    suite: TestSuite | str | Path,
    adapter: str | AgentAdapter,
    adapter_config: dict[str, Any] | AdapterConfig | None = None,
    agent_name: str = "default",
    runs_per_test: int = 1,
    fail_fast: bool = False,
    parallel_tests: bool = False,
    max_parallel_tests: int = 5,
    tag_filter: str | None = None,
    env: dict[str, str] | None = None,
) -> SuiteResult:
    """Run a test suite synchronously.

    Convenience wrapper around :func:`arun` that manages the event
    loop. Not suitable for use inside an already-running event loop
    (e.g., Jupyter). Use :func:`arun` instead in those cases.

    See :func:`arun` for parameter documentation.
    """
    return asyncio.run(
        arun(
            suite=suite,
            adapter=adapter,
            adapter_config=adapter_config,
            agent_name=agent_name,
            runs_per_test=runs_per_test,
            fail_fast=fail_fast,
            parallel_tests=parallel_tests,
            max_parallel_tests=max_parallel_tests,
            tag_filter=tag_filter,
            env=env,
        )
    )


async def aevaluate(
    response: ATPResponse,
    test: TestDefinition,
    assertions: list[Assertion] | None = None,
    trace: list[ATPEvent] | None = None,
) -> list[EvalResult]:
    """Evaluate an agent response against assertions asynchronously.

    Args:
        response: The ATP response from the agent.
        test: The test definition containing task details.
        assertions: Assertions to evaluate against. If None, uses
            the assertions from the test definition.
        trace: List of ATP events from execution. Defaults to empty.

    Returns:
        List of EvalResult, one per assertion evaluated.
    """
    assertions_to_use = assertions if assertions is not None else test.assertions
    trace_to_use = trace if trace is not None else []
    registry = get_evaluator_registry()

    results: list[EvalResult] = []
    for assertion in assertions_to_use:
        evaluator = registry.create_for_assertion(assertion.type)
        result = await evaluator.evaluate(test, response, trace_to_use, assertion)
        results.append(result)

    return results


def evaluate(
    response: ATPResponse,
    test: TestDefinition,
    assertions: list[Assertion] | None = None,
    trace: list[ATPEvent] | None = None,
) -> list[EvalResult]:
    """Evaluate an agent response against assertions synchronously.

    Convenience wrapper around :func:`aevaluate`.
    See :func:`aevaluate` for parameter documentation.
    """
    return asyncio.run(
        aevaluate(
            response=response,
            test=test,
            assertions=assertions,
            trace=trace,
        )
    )


def score(
    eval_results: list[EvalResult],
    test_id: str,
    response: ATPResponse | None = None,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
) -> ScoredTestResult:
    """Score evaluation results into a composite test score.

    Args:
        eval_results: List of evaluation results to score.
        test_id: Identifier for the test being scored.
        response: Optional ATP response with metrics for
            efficiency and cost scoring.
        max_steps: Maximum expected steps for efficiency scoring.
        max_tokens: Maximum expected tokens for cost scoring.
        max_cost_usd: Maximum expected cost in USD.

    Returns:
        ScoredTestResult with final score and breakdown.
    """
    aggregator = ScoreAggregator()
    return aggregator.score_test_result(
        test_id=test_id,
        eval_results=eval_results,
        response=response,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
    )
