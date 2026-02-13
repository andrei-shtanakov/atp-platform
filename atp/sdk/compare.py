"""Multi-model comparison mode for ATP SDK.

Run the same test suite against multiple adapter configurations and
produce a side-by-side comparison of results.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atp.adapters.base import AgentAdapter
from atp.loader.models import TestSuite
from atp.runner.models import SuiteResult

from . import _resolve_adapter, load_suite


class ModelConfig(BaseModel):
    """Configuration for a single model/adapter to compare."""

    name: str = Field(..., description="Display name for this config")
    adapter: str = Field(..., description="Adapter type (e.g., 'http')")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter configuration",
    )


class TestComparison(BaseModel):
    """Comparison results for a single test across models."""

    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    scores: dict[str, float | None] = Field(
        default_factory=dict,
        description="Score per model name (None if test errored)",
    )
    passed: dict[str, bool] = Field(
        default_factory=dict,
        description="Pass/fail per model name",
    )
    durations: dict[str, float | None] = Field(
        default_factory=dict,
        description="Duration in seconds per model name",
    )


class ComparisonResult(BaseModel):
    """Complete comparison across multiple models."""

    suite_name: str = Field(..., description="Test suite name")
    models: list[str] = Field(..., description="List of model names compared")
    tests: list[TestComparison] = Field(
        default_factory=list,
        description="Per-test comparison data",
    )
    suite_results: dict[str, SuiteResult] = Field(
        default_factory=dict,
        description="Full SuiteResult per model name",
    )
    summary: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Summary statistics per model",
    )

    @property
    def best_model(self) -> str | None:
        """Return the model name with the highest overall success rate."""
        if not self.summary:
            return None
        return max(
            self.summary,
            key=lambda m: self.summary[m].get("success_rate", 0.0),
        )


async def acompare(
    suite: TestSuite | str | Path,
    configs: Sequence[ModelConfig | dict[str, Any]],
    runs_per_test: int = 1,
    fail_fast: bool = False,
    tag_filter: str | None = None,
    env: dict[str, str] | None = None,
) -> ComparisonResult:
    """Run the same suite against multiple model configs and compare.

    Args:
        suite: A TestSuite object, or path to a YAML file.
        configs: List of ModelConfig objects or dicts with keys
            ``name``, ``adapter``, ``config``.
        runs_per_test: Number of times to run each test.
        fail_fast: Stop on first test failure per model.
        tag_filter: Filter tests by tags (e.g., "smoke,!slow").
        env: Custom environment variables for suite loading.

    Returns:
        ComparisonResult with side-by-side data.
    """
    if isinstance(suite, (str, Path)):
        suite = load_suite(suite, env=env)

    if tag_filter:
        suite = suite.filter_by_tags(tag_filter)

    resolved_configs = _resolve_configs(configs)

    suite_results: dict[str, SuiteResult] = {}
    for mc in resolved_configs:
        adapter = _resolve_adapter(mc.adapter, mc.config)
        result = await _run_for_model(
            suite=suite,
            adapter=adapter,
            model_name=mc.name,
            runs_per_test=runs_per_test,
            fail_fast=fail_fast,
        )
        suite_results[mc.name] = result

    return _build_comparison(
        suite_name=suite.test_suite,
        models=[mc.name for mc in resolved_configs],
        suite_results=suite_results,
    )


def compare(
    suite: TestSuite | str | Path,
    configs: Sequence[ModelConfig | dict[str, Any]],
    runs_per_test: int = 1,
    fail_fast: bool = False,
    tag_filter: str | None = None,
    env: dict[str, str] | None = None,
) -> ComparisonResult:
    """Run the same suite against multiple model configs (sync).

    Convenience wrapper around :func:`acompare`.
    See :func:`acompare` for parameter documentation.
    """
    return asyncio.run(
        acompare(
            suite=suite,
            configs=configs,
            runs_per_test=runs_per_test,
            fail_fast=fail_fast,
            tag_filter=tag_filter,
            env=env,
        )
    )


def _resolve_configs(
    configs: Sequence[ModelConfig | dict[str, Any]],
) -> list[ModelConfig]:
    """Normalize config inputs to ModelConfig objects."""
    resolved: list[ModelConfig] = []
    for c in configs:
        if isinstance(c, ModelConfig):
            resolved.append(c)
        else:
            resolved.append(ModelConfig(**c))
    return resolved


async def _run_for_model(
    suite: TestSuite,
    adapter: AgentAdapter,
    model_name: str,
    runs_per_test: int,
    fail_fast: bool,
) -> SuiteResult:
    """Run a suite for a single model config."""
    from atp.runner.orchestrator import TestOrchestrator

    async with TestOrchestrator(
        adapter=adapter,
        runs_per_test=runs_per_test,
        fail_fast=fail_fast,
    ) as orchestrator:
        return await orchestrator.run_suite(suite, model_name, runs_per_test)


def _build_comparison(
    suite_name: str,
    models: list[str],
    suite_results: dict[str, SuiteResult],
) -> ComparisonResult:
    """Build a ComparisonResult from individual suite results."""
    # Collect all test IDs across models (preserving order from first)
    all_test_ids: list[str] = []
    test_names: dict[str, str] = {}
    for model_name in models:
        sr = suite_results[model_name]
        for tr in sr.tests:
            if tr.test.id not in test_names:
                all_test_ids.append(tr.test.id)
                test_names[tr.test.id] = tr.test.name

    # Build per-test comparisons
    tests: list[TestComparison] = []
    for test_id in all_test_ids:
        tc = TestComparison(
            test_id=test_id,
            test_name=test_names[test_id],
        )
        for model_name in models:
            sr = suite_results[model_name]
            tr = _find_test_result(sr, test_id)
            if tr is not None:
                tc.passed[model_name] = tr.success
                tc.durations[model_name] = tr.duration_seconds
                # Use success rate as a simple score proxy
                if tr.total_runs > 0:
                    tc.scores[model_name] = round(
                        tr.successful_runs / tr.total_runs * 100, 2
                    )
                else:
                    tc.scores[model_name] = None
            else:
                tc.passed[model_name] = False
                tc.scores[model_name] = None
                tc.durations[model_name] = None
        tests.append(tc)

    # Build per-model summaries
    summary: dict[str, dict[str, Any]] = {}
    for model_name in models:
        sr = suite_results[model_name]
        summary[model_name] = {
            "total_tests": sr.total_tests,
            "passed_tests": sr.passed_tests,
            "failed_tests": sr.failed_tests,
            "success_rate": sr.success_rate,
            "duration_seconds": sr.duration_seconds,
        }

    return ComparisonResult(
        suite_name=suite_name,
        models=models,
        tests=tests,
        suite_results=suite_results,
        summary=summary,
    )


def _find_test_result(suite_result: SuiteResult, test_id: str) -> Any:
    """Find a TestResult by test_id in a SuiteResult."""
    for tr in suite_result.tests:
        if tr.test.id == test_id:
            return tr
    return None


def format_comparison_table(result: ComparisonResult) -> str:
    """Format comparison results as an ASCII table.

    Args:
        result: ComparisonResult to format.

    Returns:
        Formatted table string.
    """
    if not result.models or not result.tests:
        return "No comparison data available."

    lines: list[str] = []
    models = result.models

    # Header
    header = f"{'Test':<30}"
    for model in models:
        header += f" | {model:>15}"
    lines.append(header)
    lines.append("-" * len(header))

    # Test rows
    for tc in result.tests:
        row = f"{tc.test_name[:29]:<30}"
        for model in models:
            score = tc.scores.get(model)
            passed = tc.passed.get(model, False)
            if score is not None:
                marker = "PASS" if passed else "FAIL"
                row += f" | {score:>8.1f} {marker:>5}"
            else:
                row += f" | {'N/A':>14}"
        lines.append(row)

    # Summary
    lines.append("-" * len(header))
    summary_row = f"{'Overall':.<30}"
    for model in models:
        s = result.summary.get(model, {})
        rate = s.get("success_rate", 0.0)
        summary_row += f" | {rate * 100:>10.1f}%    "
    lines.append(summary_row)

    # Best model
    best = result.best_model
    if best:
        lines.append(f"\nBest model: {best}")

    return "\n".join(lines)


def format_comparison_json(result: ComparisonResult) -> dict[str, Any]:
    """Format comparison results as a JSON-serializable dict.

    Args:
        result: ComparisonResult to format.

    Returns:
        Dict suitable for JSON serialization.
    """
    return {
        "suite_name": result.suite_name,
        "models": result.models,
        "best_model": result.best_model,
        "summary": result.summary,
        "tests": [
            {
                "test_id": tc.test_id,
                "test_name": tc.test_name,
                "scores": tc.scores,
                "passed": tc.passed,
                "durations": tc.durations,
            }
            for tc in result.tests
        ],
    }
