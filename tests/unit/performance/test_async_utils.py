"""Tests for the async_utils module."""

import asyncio

import pytest

from atp.performance.async_utils import (
    AsyncBatcher,
    AsyncPool,
    ConcurrencyConfig,
    ParallelExecutor,
    RateLimiter,
    chunked_gather,
    gather_with_limit,
    retry_async,
    timeout_wrapper,
)


class TestConcurrencyConfig:
    """Tests for ConcurrencyConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ConcurrencyConfig()
        assert config.max_parallel == 5
        assert config.batch_size == 10
        assert config.adaptive is False

    def test_auto_config(self) -> None:
        """Test automatic configuration."""
        config = ConcurrencyConfig.auto()
        assert config.max_parallel > 0
        assert config.batch_size > 0
        assert config.adaptive is True


class TestAsyncBatcher:
    """Tests for AsyncBatcher."""

    @pytest.mark.anyio
    async def test_process_empty(self) -> None:
        """Test processing empty list."""

        async def processor(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        batcher = AsyncBatcher(processor, batch_size=3)
        results = await batcher.process([])

        assert results == []

    @pytest.mark.anyio
    async def test_process_single_batch(self) -> None:
        """Test processing items that fit in one batch."""

        async def processor(items: list[int]) -> list[int]:
            return [i * 2 for i in items]

        batcher = AsyncBatcher(processor, batch_size=10)
        results = await batcher.process([1, 2, 3])

        assert results == [2, 4, 6]

    @pytest.mark.anyio
    async def test_process_multiple_batches(self) -> None:
        """Test processing items across multiple batches."""
        batches_processed = []

        async def processor(items: list[int]) -> list[int]:
            batches_processed.append(len(items))
            return [i * 2 for i in items]

        batcher = AsyncBatcher(processor, batch_size=3)
        results = await batcher.process([1, 2, 3, 4, 5, 6, 7])

        assert results == [2, 4, 6, 8, 10, 12, 14]
        assert len(batches_processed) == 3  # 3 + 3 + 1


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.anyio
    async def test_burst(self) -> None:
        """Test burst capacity."""
        limiter = RateLimiter(rate=10, burst=3)

        # Should allow burst without waiting
        start = asyncio.get_event_loop().time()
        for _ in range(3):
            result = await limiter.acquire()
            assert result is True
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.anyio
    async def test_rate_limiting(self) -> None:
        """Test that rate limiting works."""
        limiter = RateLimiter(rate=100, burst=1)  # 100/s = 10ms between

        # First acquire should be instant
        await limiter.acquire()

        # Second should require waiting
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.005  # Should have waited

    @pytest.mark.anyio
    async def test_limit_context_manager(self) -> None:
        """Test limit context manager."""
        limiter = RateLimiter(rate=100, burst=1)

        async with limiter.limit():
            pass  # Should not raise


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    @pytest.mark.anyio
    async def test_run_all(self) -> None:
        """Test running all tasks."""
        executor = ParallelExecutor(max_parallel=2)

        async def task(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2

        tasks = [task(i) for i in range(5)]
        results = await executor.run_all(tasks)

        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.anyio
    async def test_run_all_with_exceptions(self) -> None:
        """Test handling exceptions in tasks."""
        executor = ParallelExecutor(max_parallel=2, return_exceptions=True)

        async def task(n: int) -> int:
            if n == 2:
                raise ValueError("Test error")
            return n

        tasks = [task(i) for i in range(5)]
        results = await executor.run_all(tasks)

        assert results[0] == 0
        assert results[1] == 1
        assert isinstance(results[2], ValueError)
        assert results[3] == 3
        assert results[4] == 4

    @pytest.mark.anyio
    async def test_map(self) -> None:
        """Test mapping over items."""
        executor = ParallelExecutor(max_parallel=3)

        async def double(n: int) -> int:
            await asyncio.sleep(0.001)
            return n * 2

        results = await executor.map(double, [1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10]


class TestGatherWithLimit:
    """Tests for gather_with_limit."""

    @pytest.mark.anyio
    async def test_gather_with_limit(self) -> None:
        """Test gathering with limit."""
        concurrent_count = 0
        max_concurrent = 0

        async def task(n: int) -> int:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return n

        tasks = [task(i) for i in range(10)]
        results = await gather_with_limit(tasks, limit=3)

        assert results == list(range(10))
        assert max_concurrent <= 3

    @pytest.mark.anyio
    async def test_gather_with_exceptions(self) -> None:
        """Test gathering with exceptions."""

        async def task(n: int) -> int:
            if n == 5:
                raise RuntimeError("Test error")
            return n

        tasks = [task(i) for i in range(10)]
        results = await gather_with_limit(tasks, limit=3, return_exceptions=True)

        assert results[5].__class__.__name__ == "RuntimeError"


class TestChunkedGather:
    """Tests for chunked_gather."""

    @pytest.mark.anyio
    async def test_chunked_gather(self) -> None:
        """Test chunked gathering."""

        async def process(n: int) -> int:
            return n * 2

        results = await chunked_gather(list(range(10)), process, chunk_size=3)

        assert results == [i * 2 for i in range(10)]

    @pytest.mark.anyio
    async def test_chunked_gather_with_delay(self) -> None:
        """Test chunked gathering with delay between chunks."""
        chunk_starts = []

        async def process(n: int) -> int:
            chunk_starts.append(asyncio.get_event_loop().time())
            return n

        await chunked_gather(
            list(range(6)),
            process,
            chunk_size=2,
            delay_between_chunks=0.05,
        )

        # Should have delays between chunks
        # Items 0,1 (chunk 1), 2,3 (chunk 2), 4,5 (chunk 3)
        if len(chunk_starts) >= 4:
            # Check delay between first and third chunk
            assert chunk_starts[2] - chunk_starts[0] >= 0.04


class TestTimeoutWrapper:
    """Tests for timeout_wrapper."""

    @pytest.mark.anyio
    async def test_success(self) -> None:
        """Test successful completion."""

        async def fast():
            return "success"

        result = await timeout_wrapper(fast(), timeout=1.0)
        assert result == "success"

    @pytest.mark.anyio
    async def test_timeout(self) -> None:
        """Test timeout with default."""

        async def slow():
            await asyncio.sleep(1.0)
            return "done"

        result = await timeout_wrapper(slow(), timeout=0.01, default="timeout")
        assert result == "timeout"

    @pytest.mark.anyio
    async def test_timeout_none_default(self) -> None:
        """Test timeout with None default."""

        async def slow():
            await asyncio.sleep(1.0)

        result = await timeout_wrapper(slow(), timeout=0.01)
        assert result is None


class TestRetryAsync:
    """Tests for retry_async."""

    @pytest.mark.anyio
    async def test_success_first_try(self) -> None:
        """Test success on first try."""
        call_count = 0

        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_async(succeeds, max_retries=3)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.anyio
    async def test_success_after_retry(self) -> None:
        """Test success after retries."""
        call_count = 0

        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary error")
            return "success"

        result = await retry_async(
            fails_twice,
            max_retries=3,
            delay=0.01,
            exceptions=(RuntimeError,),
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.anyio
    async def test_all_retries_fail(self) -> None:
        """Test when all retries fail."""

        async def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await retry_async(
                always_fails,
                max_retries=2,
                delay=0.01,
                exceptions=(ValueError,),
            )


class TestAsyncPool:
    """Tests for AsyncPool."""

    @pytest.mark.anyio
    async def test_acquire_and_release(self) -> None:
        """Test acquiring and releasing resources."""
        create_count = 0

        async def factory():
            nonlocal create_count
            create_count += 1
            return f"resource_{create_count}"

        pool = AsyncPool(factory, max_size=2)

        # Acquire first resource
        r1 = await pool.acquire()
        assert r1 == "resource_1"

        # Release and re-acquire
        await pool.release(r1)
        r2 = await pool.acquire()
        assert r2 == "resource_1"  # Should get the same one

        await pool.release(r2)
        await pool.close()

    @pytest.mark.anyio
    async def test_resource_context_manager(self) -> None:
        """Test resource context manager."""
        create_count = 0

        async def factory():
            nonlocal create_count
            create_count += 1
            return {"id": create_count}

        pool = AsyncPool(factory, max_size=2)

        async with pool.resource() as r:
            assert r["id"] == 1

        # Resource should be back in pool
        async with pool.resource() as r:
            assert r["id"] == 1  # Same resource

        await pool.close()

    @pytest.mark.anyio
    async def test_cleanup_on_close(self) -> None:
        """Test cleanup function is called on close."""
        cleaned = []

        async def factory():
            return {"cleaned": False}

        async def cleanup(resource):
            resource["cleaned"] = True
            cleaned.append(resource)

        pool = AsyncPool(factory, cleanup=cleanup, max_size=2)

        r1 = await pool.acquire()
        await pool.release(r1)

        await pool.close()

        assert len(cleaned) == 1
        assert cleaned[0]["cleaned"] is True
