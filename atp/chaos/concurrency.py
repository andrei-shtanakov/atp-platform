"""Concurrency stress testing for ATP platform internals.

Utilities for testing the platform's own concurrency safety:
sandbox creation, event buffer writes, cost tracker, etc.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Result of a concurrency stress test."""

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 1.0
        return self.completed / self.total_tasks


async def stress_test_async(
    fn: Callable[..., Coroutine[Any, Any, Any]],
    concurrency: int = 10,
    iterations: int = 100,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> StressResult:
    """Run an async function under concurrent load.

    Args:
        fn: Async function to stress test.
        concurrency: Number of concurrent tasks.
        iterations: Total number of calls.
        args: Positional arguments for fn.
        kwargs: Keyword arguments for fn.

    Returns:
        StressResult with success/failure counts.
    """
    kwargs = kwargs or {}
    result = StressResult(total_tasks=iterations)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(idx: int) -> None:
        async with semaphore:
            try:
                await fn(*args, **kwargs)
                result.completed += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append((idx, str(exc)))

    tasks = [asyncio.create_task(worker(i)) for i in range(iterations)]
    await asyncio.gather(*tasks)

    return result


async def stress_test_sync(
    fn: Callable[..., Any],
    concurrency: int = 10,
    iterations: int = 100,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> StressResult:
    """Run a sync function under concurrent load via threads.

    Args:
        fn: Sync function to stress test.
        concurrency: Number of concurrent threads.
        iterations: Total number of calls.
        args: Positional arguments for fn.
        kwargs: Keyword arguments for fn.

    Returns:
        StressResult with success/failure counts.
    """
    kwargs = kwargs or {}
    result = StressResult(total_tasks=iterations)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(idx: int) -> None:
        async with semaphore:
            try:
                await asyncio.to_thread(fn, *args, **kwargs)
                result.completed += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append((idx, str(exc)))

    tasks = [asyncio.create_task(worker(i)) for i in range(iterations)]
    await asyncio.gather(*tasks)

    return result
