"""Async optimization utilities for ATP.

Provides utilities for:
- Batching async operations
- Connection pooling configuration
- Semaphore-based rate limiting
- Efficient parallel execution patterns
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent execution."""

    max_parallel: int = 5
    batch_size: int = 10
    semaphore_timeout: float | None = None
    adaptive: bool = False  # Adjust based on system load

    @classmethod
    def auto(cls) -> "ConcurrencyConfig":
        """
        Create config with automatic settings based on system.

        Uses CPU count to determine parallelism.
        """
        cpu_count = os.cpu_count() or 4
        return cls(
            max_parallel=min(cpu_count * 2, 20),
            batch_size=cpu_count * 5,
            adaptive=True,
        )


class AsyncBatcher[T, R]:
    """
    Batch async operations for improved throughput.

    Groups items into batches and processes them with controlled concurrency.
    """

    def __init__(
        self,
        processor: Callable[[list[T]], Awaitable[list[R]]],
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
    ) -> None:
        """
        Initialize batcher.

        Args:
            processor: Async function that processes a batch of items.
            batch_size: Maximum items per batch.
            max_concurrent_batches: Maximum batches to process concurrently.
        """
        self.processor = processor
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process(self, items: list[T]) -> list[R]:
        """
        Process items in batches.

        Args:
            items: Items to process.

        Returns:
            Results in the same order as inputs.
        """
        if not items:
            return []

        # Split into batches
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        # Process batches concurrently
        async def process_batch(batch: list[T]) -> list[R]:
            async with self._semaphore:
                return await self.processor(batch)

        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        results: list[R] = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results


class RateLimiter:
    """
    Token bucket rate limiter for async operations.

    Limits the rate of operations to prevent overloading resources.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            rate: Maximum operations per second.
            burst: Maximum burst size (tokens).
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float | None = None) -> bool:
        """
        Acquire a token, waiting if necessary.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if token acquired, False if timeout.
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_update
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True

            # Need to wait
            wait_time = (1 - self._tokens) / self._rate

            if timeout is not None and wait_time > timeout:
                return False

            await asyncio.sleep(wait_time)
            self._tokens = 0
            return True

    @asynccontextmanager
    async def limit(self):
        """Context manager for rate-limited operations."""
        await self.acquire()
        yield


class ParallelExecutor:
    """
    Execute async operations in parallel with controlled concurrency.

    Provides efficient patterns for running multiple async tasks.
    """

    def __init__(
        self,
        max_parallel: int = 5,
        return_exceptions: bool = False,
    ) -> None:
        """
        Initialize executor.

        Args:
            max_parallel: Maximum concurrent operations.
            return_exceptions: Whether to return exceptions in results.
        """
        self.max_parallel = max_parallel
        self.return_exceptions = return_exceptions
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def run_all(
        self,
        tasks: list[Awaitable[T]],
    ) -> list[T | BaseException]:
        """
        Run all tasks with concurrency limiting.

        Args:
            tasks: Async tasks to run.

        Returns:
            Results in the same order as tasks.
        """

        async def limited_task(task: Awaitable[T]) -> T:
            async with self._semaphore:
                return await task

        limited = [limited_task(t) for t in tasks]
        return await asyncio.gather(*limited, return_exceptions=self.return_exceptions)

    async def map(
        self,
        func: Callable[[T], Awaitable[R]],
        items: Iterable[T],
    ) -> list[R | BaseException]:
        """
        Map an async function over items with concurrency limiting.

        Args:
            func: Async function to apply.
            items: Items to process.

        Returns:
            Results in the same order as items.
        """

        async def process(item: T) -> R:
            async with self._semaphore:
                return await func(item)

        tasks = [process(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=self.return_exceptions)

    async def first_completed(
        self,
        tasks: list[Awaitable[T]],
    ) -> tuple[T, list[asyncio.Task[T]]]:
        """
        Return the first completed task result and remaining tasks.

        Args:
            tasks: Async tasks to run.

        Returns:
            Tuple of (first result, remaining tasks).
        """
        wrapped = [asyncio.ensure_future(t) for t in tasks]
        done, pending = await asyncio.wait(wrapped, return_when=asyncio.FIRST_COMPLETED)

        first = done.pop()
        return first.result(), list(pending)


async def gather_with_limit[T](
    tasks: list[Awaitable[T]],
    limit: int = 5,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """
    Run tasks with a concurrency limit.

    Convenience function for common pattern of limited parallelism.

    Args:
        tasks: Async tasks to run.
        limit: Maximum concurrent tasks.
        return_exceptions: Whether to return exceptions.

    Returns:
        Results in task order.
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *[limited(t) for t in tasks],
        return_exceptions=return_exceptions,
    )


async def chunked_gather(
    items: list[T],
    func: Callable[[T], Awaitable[R]],
    chunk_size: int = 10,
    delay_between_chunks: float = 0.0,
) -> list[R]:
    """
    Process items in chunks with optional delay between chunks.

    Useful for APIs with rate limits or to prevent overwhelming resources.

    Args:
        items: Items to process.
        func: Async function to apply to each item.
        chunk_size: Size of each chunk.
        delay_between_chunks: Delay in seconds between chunks.

    Returns:
        Results in item order.
    """
    results: list[R] = []

    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        chunk_results = await asyncio.gather(*[func(item) for item in chunk])
        results.extend(chunk_results)

        if delay_between_chunks > 0 and i + chunk_size < len(items):
            await asyncio.sleep(delay_between_chunks)

    return results


async def timeout_wrapper(
    coro: Awaitable[T],
    timeout: float,
    default: T | None = None,
) -> T | None:
    """
    Wrap a coroutine with timeout, returning default on timeout.

    Args:
        coro: Coroutine to run.
        timeout: Timeout in seconds.
        default: Value to return on timeout.

    Returns:
        Coroutine result or default on timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        return default


async def retry_async[T](
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to call.
        max_retries: Maximum retry attempts.
        delay: Initial delay between retries.
        backoff: Multiplier for delay on each retry.
        exceptions: Exception types to catch and retry.

    Returns:
        Function result.

    Raises:
        Last exception if all retries fail.
    """
    last_exception: Exception | None = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.debug("Retry %d/%d after error: %s", attempt + 1, max_retries, e)
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                raise

    # Should never reach here, but satisfy type checker
    raise last_exception  # type: ignore[misc]


class AsyncPool[T]:
    """
    Pool of reusable async resources with automatic cleanup.

    Manages a pool of resources (like connections) with automatic
    creation and cleanup.
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        cleanup: Callable[[T], Awaitable[None]] | None = None,
        max_size: int = 10,
        min_size: int = 1,
    ) -> None:
        """
        Initialize pool.

        Args:
            factory: Async function to create new resources.
            cleanup: Async function to cleanup resources.
            max_size: Maximum pool size.
            min_size: Minimum pool size to maintain.
        """
        self._factory = factory
        self._cleanup = cleanup
        self._max_size = max_size
        self._min_size = min_size
        self._pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def acquire(self) -> T:
        """
        Acquire a resource from the pool.

        Returns:
            Resource from pool or newly created.
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            async with self._lock:
                if self._created < self._max_size:
                    self._created += 1
                    return await self._factory()

            # Pool is full, wait for one
            return await self._pool.get()

    async def release(self, resource: T) -> None:
        """
        Return a resource to the pool.

        Args:
            resource: Resource to return.
        """
        if self._closed:
            if self._cleanup:
                await self._cleanup(resource)
            return

        try:
            self._pool.put_nowait(resource)
        except asyncio.QueueFull:
            if self._cleanup:
                await self._cleanup(resource)
            async with self._lock:
                self._created -= 1

    @asynccontextmanager
    async def resource(self):
        """Context manager for acquiring and releasing a resource."""
        resource = await self.acquire()
        try:
            yield resource
        finally:
            await self.release(resource)

    async def close(self) -> None:
        """Close the pool and cleanup all resources."""
        self._closed = True

        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                if self._cleanup:
                    await self._cleanup(resource)
            except asyncio.QueueEmpty:
                break

        self._created = 0


async def stream_with_timeout(
    stream: AsyncIterator[T],
    timeout: float,
    on_timeout: Callable[[], T] | None = None,
) -> AsyncIterator[T]:
    """
    Add timeout to async iterator items.

    Args:
        stream: Async iterator to wrap.
        timeout: Timeout per item in seconds.
        on_timeout: Optional callback to generate timeout item.

    Yields:
        Items from stream or timeout items.
    """

    async def get_next():
        return await stream.__anext__()

    while True:
        try:
            item = await asyncio.wait_for(get_next(), timeout=timeout)
            yield item
        except TimeoutError:
            if on_timeout:
                yield on_timeout()
            else:
                raise
        except StopAsyncIteration:
            break
