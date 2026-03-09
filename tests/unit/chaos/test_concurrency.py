"""Tests for concurrency stress testing utilities."""

import pytest

from atp.chaos.concurrency import StressResult, stress_test_async, stress_test_sync


class TestStressResult:
    def test_success_rate(self) -> None:
        r = StressResult(total_tasks=10, completed=8, failed=2)
        assert r.success_rate == 0.8

    def test_empty(self) -> None:
        r = StressResult()
        assert r.success_rate == 1.0


class TestStressTestAsync:
    @pytest.mark.anyio
    async def test_all_succeed(self) -> None:
        counter = {"n": 0}

        async def fn() -> None:
            counter["n"] += 1

        result = await stress_test_async(fn, concurrency=5, iterations=20)
        assert result.completed == 20
        assert result.failed == 0
        assert result.success_rate == 1.0

    @pytest.mark.anyio
    async def test_with_failures(self) -> None:
        call_count = {"n": 0}

        async def fn() -> None:
            call_count["n"] += 1
            if call_count["n"] % 3 == 0:
                raise ValueError("boom")

        result = await stress_test_async(fn, concurrency=3, iterations=9)
        assert result.completed == 6
        assert result.failed == 3
        assert len(result.errors) == 3

    @pytest.mark.anyio
    async def test_with_args(self) -> None:
        results: list[int] = []

        async def fn(x: int) -> None:
            results.append(x)

        await stress_test_async(fn, concurrency=2, iterations=5, args=(42,))
        assert len(results) == 5
        assert all(v == 42 for v in results)


class TestStressTestSync:
    @pytest.mark.anyio
    async def test_sync_function(self) -> None:
        import threading

        threads: set[int] = set()

        def fn() -> None:
            threads.add(threading.current_thread().ident or 0)

        result = await stress_test_sync(fn, concurrency=3, iterations=10)
        assert result.completed == 10
        assert result.failed == 0
