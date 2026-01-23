"""Tests for the benchmark module."""

import time

import pytest

from atp.performance.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSuite,
    run_standard_benchmarks,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_mean_ms(self) -> None:
        """Test mean time in milliseconds."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            total_seconds=1.0,
            mean_seconds=0.1,
            min_seconds=0.05,
            max_seconds=0.15,
            std_seconds=0.02,
            throughput=10.0,
        )
        assert result.mean_ms == 100.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            name="test_bench",
            iterations=5,
            total_seconds=0.5,
            mean_seconds=0.1,
            min_seconds=0.08,
            max_seconds=0.12,
            std_seconds=0.01,
            throughput=10.0,
            metadata={"key": "value"},
        )
        d = result.to_dict()

        assert d["name"] == "test_bench"
        assert d["iterations"] == 5
        assert d["mean_ms"] == 100.0
        assert d["metadata"] == {"key": "value"}


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        suite = BenchmarkSuite(
            name="test_suite",
            start_time=100.0,
            end_time=110.0,
        )
        assert suite.duration_seconds == 10.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            name="bench1",
            iterations=10,
            total_seconds=1.0,
            mean_seconds=0.1,
            min_seconds=0.05,
            max_seconds=0.15,
            std_seconds=0.02,
            throughput=10.0,
        )
        suite = BenchmarkSuite(
            name="test_suite",
            start_time=100.0,
            end_time=101.0,
            results=[result],
        )
        d = suite.to_dict()

        assert d["name"] == "test_suite"
        assert len(d["results"]) == 1
        assert d["duration_seconds"] == 1.0

    def test_format_report(self) -> None:
        """Test report formatting."""
        result = BenchmarkResult(
            name="test_operation",
            iterations=100,
            total_seconds=1.0,
            mean_seconds=0.01,
            min_seconds=0.005,
            max_seconds=0.015,
            std_seconds=0.002,
            throughput=100.0,
        )
        suite = BenchmarkSuite(
            name="Test Suite",
            start_time=0.0,
            end_time=1.0,
            results=[result],
        )
        report = suite.format_report()

        assert "Test Suite" in report
        assert "test_operation" in report
        assert "100" in report  # iterations


class TestBenchmark:
    """Tests for Benchmark class."""

    def test_init(self) -> None:
        """Test benchmark initialization."""
        benchmark = Benchmark(
            warmup_iterations=5,
            min_iterations=20,
            max_iterations=50,
            target_seconds=2.0,
        )
        assert benchmark.warmup_iterations == 5
        assert benchmark.min_iterations == 20
        assert benchmark.max_iterations == 50
        assert benchmark.target_seconds == 2.0

    def test_run_sync(self) -> None:
        """Test running a synchronous benchmark."""
        benchmark = Benchmark(
            warmup_iterations=1,
            min_iterations=5,
            max_iterations=10,
            target_seconds=0.1,
        )

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)

        result = benchmark.run("test_sync", operation)

        assert result.name == "test_sync"
        assert result.iterations >= 5  # At least min_iterations
        assert result.mean_seconds > 0
        assert result.throughput > 0
        assert call_count >= 5 + 1  # min + warmup

    def test_run_with_setup_teardown(self) -> None:
        """Test benchmark with setup and teardown."""
        benchmark = Benchmark(
            warmup_iterations=1,
            min_iterations=3,
            max_iterations=5,
            target_seconds=0.05,
        )

        setup_count = 0
        teardown_count = 0
        run_count = 0

        def setup():
            nonlocal setup_count
            setup_count += 1

        def teardown():
            nonlocal teardown_count
            teardown_count += 1

        def operation():
            nonlocal run_count
            run_count += 1

        result = benchmark.run(
            "test_setup",
            operation,
            setup=setup,
            teardown=teardown,
        )

        # Setup and teardown should be called for each iteration
        total_calls = result.iterations + benchmark.warmup_iterations
        assert setup_count == total_calls
        assert teardown_count == total_calls

    def test_run_with_metadata(self) -> None:
        """Test benchmark with metadata."""
        benchmark = Benchmark(
            warmup_iterations=0,
            min_iterations=1,
            max_iterations=1,
            target_seconds=0.01,
        )

        result = benchmark.run(
            "test_meta",
            lambda: None,
            metadata={"custom": "data"},
        )

        assert result.metadata == {"custom": "data"}

    @pytest.mark.anyio
    async def test_run_async(self) -> None:
        """Test running an async benchmark."""
        import asyncio

        benchmark = Benchmark(
            warmup_iterations=1,
            min_iterations=3,
            max_iterations=5,
            target_seconds=0.05,
        )

        async def async_operation():
            await asyncio.sleep(0.001)

        result = await benchmark.run_async("test_async", async_operation)

        assert result.name == "test_async"
        assert result.iterations >= 3
        assert result.mean_seconds > 0

    def test_statistics_calculation(self) -> None:
        """Test that statistics are calculated correctly."""
        benchmark = Benchmark(
            warmup_iterations=0,
            min_iterations=10,
            max_iterations=10,
            target_seconds=10.0,  # High to ensure 10 iterations
        )

        def operation():
            time.sleep(0.001)

        result = benchmark.run("test_stats", operation)

        assert result.iterations == 10
        assert result.min_seconds <= result.mean_seconds <= result.max_seconds
        assert result.std_seconds >= 0


class TestRunStandardBenchmarks:
    """Tests for run_standard_benchmarks."""

    def test_run_standard_benchmarks(self) -> None:
        """Test running standard benchmarks."""
        # This is a smoke test - just verify it runs without error
        suite = run_standard_benchmarks()

        assert suite is not None
        assert suite.name == "ATP Standard Benchmarks"
        assert len(suite.results) > 0
        assert suite.duration_seconds > 0

        # Check some expected benchmarks
        names = [r.name for r in suite.results]
        assert "test_suite_load_uncached" in names
        assert "test_suite_load_cached" in names
        assert "score_aggregation" in names
        assert "statistics_calculation" in names
