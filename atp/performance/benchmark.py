"""Benchmark suite for ATP performance testing.

Provides standardized benchmarks for:
- Test suite loading
- Test execution
- Adapter operations
- Evaluator execution
- Score aggregation
- Statistics calculation
"""

import asyncio
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atp.evaluators.base import EvalCheck, EvalResult
from atp.loader.loader import TestLoader
from atp.performance.cache import CachedTestLoader, TestSuiteCache
from atp.performance.profiler import Profiler
from atp.protocol import ATPResponse, Metrics, ResponseStatus
from atp.scoring.aggregator import ScoreAggregator
from atp.statistics.calculator import StatisticsCalculator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    std_seconds: float
    throughput: float  # iterations per second
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean_seconds * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "mean_ms": self.mean_ms,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "std_seconds": self.std_seconds,
            "throughput": self.throughput,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    start_time: float
    end_time: float
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }

    def format_report(self) -> str:
        """Format a human-readable report."""
        lines = [
            f"Benchmark Suite: {self.name}",
            "=" * 60,
            f"Total duration: {self.duration_seconds:.2f}s",
            "",
            f"{'Benchmark':<35} {'Iters':>6} {'Mean':>10} {'Min':>10} "
            f"{'Max':>10} {'ops/s':>10}",
            "-" * 60,
        ]

        for r in self.results:
            lines.append(
                f"{r.name:<35} {r.iterations:>6} "
                f"{r.mean_ms:>8.2f}ms {r.min_seconds * 1000:>8.2f}ms "
                f"{r.max_seconds * 1000:>8.2f}ms {r.throughput:>10.1f}"
            )

        lines.append("-" * 60)
        return "\n".join(lines)


class Benchmark:
    """
    Benchmark runner for ATP operations.

    Runs standardized benchmarks with warmup, multiple iterations,
    and statistical analysis.
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        min_iterations: int = 10,
        max_iterations: int = 100,
        target_seconds: float = 1.0,
    ) -> None:
        """
        Initialize benchmark runner.

        Args:
            warmup_iterations: Number of warmup iterations (not counted).
            min_iterations: Minimum iterations to run.
            max_iterations: Maximum iterations to run.
            target_seconds: Target total time to spend on each benchmark.
        """
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.target_seconds = target_seconds

    def run(
        self,
        name: str,
        func: Callable[[], Any],
        setup: Callable[[], None] | None = None,
        teardown: Callable[[], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """
        Run a synchronous benchmark.

        Args:
            name: Benchmark name.
            func: Function to benchmark.
            setup: Optional setup function called before each iteration.
            teardown: Optional teardown function called after each iteration.
            metadata: Optional metadata to include in results.

        Returns:
            BenchmarkResult with timing statistics.
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            if setup:
                setup()
            func()
            if teardown:
                teardown()

        # Run benchmark
        times: list[float] = []
        start_total = time.perf_counter()

        for i in range(self.max_iterations):
            if setup:
                setup()

            start = time.perf_counter()
            func()
            end = time.perf_counter()

            if teardown:
                teardown()

            times.append(end - start)

            # Check if we've reached target time and minimum iterations
            elapsed = time.perf_counter() - start_total
            if i >= self.min_iterations - 1 and elapsed >= self.target_seconds:
                break

        end_total = time.perf_counter()

        return self._calculate_result(name, times, end_total - start_total, metadata)

    async def run_async(
        self,
        name: str,
        func: Callable[[], Any],
        setup: Callable[[], None] | None = None,
        teardown: Callable[[], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """
        Run an async benchmark.

        Args:
            name: Benchmark name.
            func: Async function to benchmark.
            setup: Optional setup function.
            teardown: Optional teardown function.
            metadata: Optional metadata.

        Returns:
            BenchmarkResult with timing statistics.
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            if setup:
                setup()
            await func()
            if teardown:
                teardown()

        # Run benchmark
        times: list[float] = []
        start_total = time.perf_counter()

        for i in range(self.max_iterations):
            if setup:
                setup()

            start = time.perf_counter()
            await func()
            end = time.perf_counter()

            if teardown:
                teardown()

            times.append(end - start)

            elapsed = time.perf_counter() - start_total
            if i >= self.min_iterations - 1 and elapsed >= self.target_seconds:
                break

        end_total = time.perf_counter()

        return self._calculate_result(name, times, end_total - start_total, metadata)

    def _calculate_result(
        self,
        name: str,
        times: list[float],
        total_seconds: float,
        metadata: dict[str, Any] | None,
    ) -> BenchmarkResult:
        """Calculate benchmark statistics."""
        n = len(times)
        mean = sum(times) / n
        min_t = min(times)
        max_t = max(times)
        variance = sum((t - mean) ** 2 for t in times) / max(n - 1, 1)
        std = variance**0.5

        return BenchmarkResult(
            name=name,
            iterations=n,
            total_seconds=total_seconds,
            mean_seconds=mean,
            min_seconds=min_t,
            max_seconds=max_t,
            std_seconds=std,
            throughput=n / total_seconds if total_seconds > 0 else 0,
            metadata=metadata or {},
        )


def run_standard_benchmarks() -> BenchmarkSuite:
    """
    Run the standard ATP benchmark suite.

    Returns:
        BenchmarkSuite with all benchmark results.
    """
    benchmark = Benchmark()
    suite = BenchmarkSuite(
        name="ATP Standard Benchmarks",
        start_time=time.time(),
        end_time=0,
    )

    # Benchmark 1: Test suite loading (uncached)
    suite_yaml = _create_test_suite_yaml(num_tests=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(suite_yaml)
        suite_path = f.name

    def load_uncached():
        loader = TestLoader()
        loader.load_file(suite_path)

    result = benchmark.run(
        "test_suite_load_uncached",
        load_uncached,
        metadata={"num_tests": 10},
    )
    suite.results.append(result)

    # Benchmark 2: Test suite loading (cached)
    cache = TestSuiteCache()
    cached_loader = CachedTestLoader(cache=cache, use_shared_cache=False)

    # Warm the cache
    cached_loader.load_file(suite_path)

    def load_cached():
        cached_loader.load_file(suite_path)

    result = benchmark.run(
        "test_suite_load_cached",
        load_cached,
        metadata={"num_tests": 10},
    )
    suite.results.append(result)

    # Benchmark 3: Score aggregation
    eval_results = _create_sample_eval_results(num_checks=20)
    response = _create_sample_response()
    aggregator = ScoreAggregator()

    def aggregate_scores():
        aggregator.aggregate(eval_results, response)

    result = benchmark.run(
        "score_aggregation",
        aggregate_scores,
        metadata={"num_checks": 20},
    )
    suite.results.append(result)

    # Benchmark 4: Statistics calculation
    calculator = StatisticsCalculator()
    values = [float(i % 100) + 0.5 for i in range(100)]

    def calculate_stats():
        calculator.compute(values)

    result = benchmark.run(
        "statistics_calculation",
        calculate_stats,
        metadata={"num_values": 100},
    )
    suite.results.append(result)

    # Benchmark 5: Large suite loading
    large_suite_yaml = _create_test_suite_yaml(num_tests=100)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(large_suite_yaml)
        large_suite_path = f.name

    def load_large_suite():
        loader = TestLoader()
        loader.load_file(large_suite_path)

    result = benchmark.run(
        "large_suite_load",
        load_large_suite,
        metadata={"num_tests": 100},
    )
    suite.results.append(result)

    # Benchmark 6: Profiler overhead
    profiler = Profiler(enabled=True)

    def with_profiling():
        with profiler.profile("dummy_op"):
            pass

    result = benchmark.run(
        "profiler_overhead_enabled",
        with_profiling,
    )
    suite.results.append(result)

    profiler_disabled = Profiler(enabled=False)

    def without_profiling():
        with profiler_disabled.profile("dummy_op"):
            pass

    result = benchmark.run(
        "profiler_overhead_disabled",
        without_profiling,
    )
    suite.results.append(result)

    # Cleanup
    Path(suite_path).unlink(missing_ok=True)
    Path(large_suite_path).unlink(missing_ok=True)

    suite.end_time = time.time()
    return suite


async def run_async_benchmarks() -> BenchmarkSuite:
    """
    Run async ATP benchmarks.

    Returns:
        BenchmarkSuite with async benchmark results.
    """
    benchmark = Benchmark()
    suite = BenchmarkSuite(
        name="ATP Async Benchmarks",
        start_time=time.time(),
        end_time=0,
    )

    # Benchmark async profiler
    profiler = Profiler(enabled=True)

    async def with_async_profiling():
        async with profiler.profile_async("dummy_async_op"):
            await asyncio.sleep(0)

    result = await benchmark.run_async(
        "async_profiler_overhead",
        with_async_profiling,
    )
    suite.results.append(result)

    suite.end_time = time.time()
    return suite


def _create_test_suite_yaml(num_tests: int) -> str:
    """Create a test suite YAML string for benchmarking."""
    tests = []
    for i in range(num_tests):
        tests.append(f"""
  - id: test-{i:03d}
    name: Test {i}
    task:
      description: Test task {i}
      input_data:
        key: value{i}
    constraints:
      max_steps: 50
      timeout_seconds: 300
    assertions:
      - type: artifact_exists
        config:
          path: output_{i}.txt
    tags:
      - benchmark
      - test-{i % 5}""")

    return f"""test_suite: benchmark-suite
version: "1.0"
description: Benchmark test suite

defaults:
  timeout_seconds: 300
  runs_per_test: 1
  scoring:
    quality_weight: 0.4
    completeness_weight: 0.3
    efficiency_weight: 0.2
    cost_weight: 0.1

tests:
{"".join(tests)}
"""


def _create_sample_eval_results(num_checks: int) -> list[EvalResult]:
    """Create sample evaluation results for benchmarking."""
    checks = []
    for i in range(num_checks):
        checks.append(
            EvalCheck(
                name=f"check_{i}",
                passed=i % 3 != 0,
                score=0.5 + (i % 10) / 20,
                message=f"Check {i} result",
            )
        )

    return [
        EvalResult(evaluator="artifact", checks=checks[: num_checks // 2]),
        EvalResult(evaluator="behavior", checks=checks[num_checks // 2 :]),
    ]


def _create_sample_response() -> ATPResponse:
    """Create a sample ATP response for benchmarking."""
    return ATPResponse(
        task_id="benchmark-task",
        status=ResponseStatus.COMPLETED,
        metrics=Metrics(
            total_tokens=5000,
            total_steps=25,
            wall_time_seconds=10.5,
            cost_usd=0.05,
        ),
    )


# CLI entry point for running benchmarks
if __name__ == "__main__":
    print("Running ATP Standard Benchmarks...")
    results = run_standard_benchmarks()
    print(results.format_report())

    print("\nRunning ATP Async Benchmarks...")
    async_results = asyncio.run(run_async_benchmarks())
    print(async_results.format_report())
