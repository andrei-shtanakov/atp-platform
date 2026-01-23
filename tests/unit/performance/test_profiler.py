"""Tests for the profiler module."""

import asyncio
import time

import pytest

from atp.performance.profiler import (
    Profiler,
    ProfileResult,
    ProfileStats,
    disable_profiling,
    enable_profiling,
    get_profiler,
    profile,
    profile_async,
    profiled,
    set_profiler,
)


class TestProfileResult:
    """Tests for ProfileResult."""

    def test_duration_ms(self) -> None:
        """Test duration conversion to milliseconds."""
        result = ProfileResult(
            operation="test",
            duration_seconds=0.5,
            start_time=0.0,
            end_time=0.5,
        )
        assert result.duration_ms == 500.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = ProfileResult(
            operation="test_op",
            duration_seconds=1.0,
            start_time=100.0,
            end_time=101.0,
            metadata={"key": "value"},
        )
        d = result.to_dict()

        assert d["operation"] == "test_op"
        assert d["duration_seconds"] == 1.0
        assert d["duration_ms"] == 1000.0
        assert d["metadata"] == {"key": "value"}


class TestProfileStats:
    """Tests for ProfileStats."""

    def test_total_ms(self) -> None:
        """Test total time in milliseconds."""
        stats = ProfileStats(
            operation="test",
            count=5,
            total_seconds=2.5,
            min_seconds=0.3,
            max_seconds=0.7,
            mean_seconds=0.5,
        )
        assert stats.total_ms == 2500.0

    def test_mean_ms(self) -> None:
        """Test mean time in milliseconds."""
        stats = ProfileStats(
            operation="test",
            count=5,
            total_seconds=2.5,
            min_seconds=0.3,
            max_seconds=0.7,
            mean_seconds=0.5,
        )
        assert stats.mean_ms == 500.0


class TestProfiler:
    """Tests for Profiler class."""

    def test_init_enabled(self) -> None:
        """Test profiler initialization with enabled=True."""
        profiler = Profiler(enabled=True)
        assert profiler.enabled is True

    def test_init_disabled(self) -> None:
        """Test profiler initialization with enabled=False."""
        profiler = Profiler(enabled=False)
        assert profiler.enabled is False

    def test_profile_sync(self) -> None:
        """Test synchronous profiling."""
        profiler = Profiler(enabled=True)

        with profiler.profile("test_op", {"key": "value"}) as result:
            time.sleep(0.01)

        assert result is not None
        assert result.operation == "test_op"
        assert result.duration_seconds > 0.01
        assert result.metadata == {"key": "value"}

    def test_profile_disabled(self) -> None:
        """Test profiling when disabled."""
        profiler = Profiler(enabled=False)

        with profiler.profile("test_op") as result:
            pass

        assert result is None

    def test_nested_profiling(self) -> None:
        """Test nested profile contexts."""
        profiler = Profiler(enabled=True)

        with profiler.profile("outer") as outer:
            with profiler.profile("inner") as inner:
                time.sleep(0.01)

        assert outer is not None
        assert inner is not None
        assert len(outer.children) == 1
        assert outer.children[0] is inner
        assert outer.duration_seconds >= inner.duration_seconds

    @pytest.mark.anyio
    async def test_profile_async(self) -> None:
        """Test async profiling."""
        profiler = Profiler(enabled=True)

        async with profiler.profile_async("async_op") as result:
            await asyncio.sleep(0.01)

        assert result is not None
        assert result.operation == "async_op"
        assert result.duration_seconds > 0.01

    def test_get_results(self) -> None:
        """Test getting profiling results."""
        profiler = Profiler(enabled=True)

        with profiler.profile("op1"):
            pass
        with profiler.profile("op2"):
            pass

        results = profiler.get_results()
        assert len(results) == 2
        assert results[0].operation == "op1"
        assert results[1].operation == "op2"

    def test_get_stats(self) -> None:
        """Test getting statistics."""
        profiler = Profiler(enabled=True)

        for _ in range(5):
            with profiler.profile("repeated_op"):
                time.sleep(0.001)

        stats = profiler.get_stats()
        assert "repeated_op" in stats
        assert stats["repeated_op"].count == 5
        assert stats["repeated_op"].total_seconds > 0.005

    def test_get_stats_specific_operation(self) -> None:
        """Test getting stats for specific operation."""
        profiler = Profiler(enabled=True)

        with profiler.profile("op1"):
            pass
        with profiler.profile("op2"):
            pass

        stats = profiler.get_stats("op1")
        assert "op1" in stats
        assert "op2" not in stats

    def test_reset(self) -> None:
        """Test resetting profiler data."""
        profiler = Profiler(enabled=True)

        with profiler.profile("test"):
            pass

        assert len(profiler.get_results()) == 1

        profiler.reset()

        assert len(profiler.get_results()) == 0
        assert len(profiler.get_stats()) == 0

    def test_get_summary(self) -> None:
        """Test getting summary."""
        profiler = Profiler(enabled=True)

        with profiler.profile("op"):
            time.sleep(0.01)

        summary = profiler.get_summary()

        assert "total_duration_seconds" in summary
        assert "operation_count" in summary
        assert "stats" in summary
        assert "results" in summary
        assert summary["operation_count"] == 1

    def test_format_report(self) -> None:
        """Test formatting report."""
        profiler = Profiler(enabled=True)

        with profiler.profile("test_operation"):
            time.sleep(0.001)

        report = profiler.format_report()

        assert "ATP Profiler Report" in report
        assert "test_operation" in report


class TestGlobalProfiler:
    """Tests for global profiler functions."""

    def test_get_profiler(self) -> None:
        """Test getting global profiler."""
        profiler = get_profiler()
        assert profiler is not None
        assert isinstance(profiler, Profiler)

    def test_set_profiler(self) -> None:
        """Test setting global profiler."""
        custom = Profiler(enabled=True)
        set_profiler(custom)

        assert get_profiler() is custom

    def test_enable_profiling(self) -> None:
        """Test enabling global profiling."""
        profiler = enable_profiling()
        assert profiler.enabled is True

    def test_disable_profiling(self) -> None:
        """Test disabling global profiling."""
        enable_profiling()
        disable_profiling()
        assert get_profiler().enabled is False

    def test_profile_context_manager(self) -> None:
        """Test global profile context manager."""
        enable_profiling()

        with profile("test_global"):
            pass

        results = get_profiler().get_results()
        assert len(results) >= 1

    @pytest.mark.anyio
    async def test_profile_async_context_manager(self) -> None:
        """Test global async profile context manager."""
        enable_profiling()

        async with profile_async("test_async_global"):
            await asyncio.sleep(0.001)

        results = get_profiler().get_results()
        assert any(r.operation == "test_async_global" for r in results)


class TestProfiledDecorator:
    """Tests for profiled decorator."""

    def test_profiled_sync_function(self) -> None:
        """Test profiled decorator on sync function."""
        enable_profiling()
        get_profiler().reset()

        @profiled("decorated_sync")
        def my_func():
            return 42

        result = my_func()

        assert result == 42
        stats = get_profiler().get_stats()
        assert "decorated_sync" in stats

    def test_profiled_default_name(self) -> None:
        """Test profiled decorator with default name."""
        enable_profiling()
        get_profiler().reset()

        @profiled()
        def another_func():
            return 123

        another_func()

        stats = get_profiler().get_stats()
        assert "another_func" in stats

    @pytest.mark.anyio
    async def test_profiled_async_function(self) -> None:
        """Test profiled decorator on async function."""
        enable_profiling()
        get_profiler().reset()

        @profiled("decorated_async")
        async def async_func():
            await asyncio.sleep(0.001)
            return "done"

        result = await async_func()

        assert result == "done"
        stats = get_profiler().get_stats()
        assert "decorated_async" in stats
