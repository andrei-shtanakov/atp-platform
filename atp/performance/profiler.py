"""Profiling infrastructure for ATP test execution.

Provides timing and performance measurement for:
- Test suite loading
- Test execution (individual tests and runs)
- Adapter operations
- Evaluator execution
- Score aggregation
"""

import asyncio
import functools
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ProfileResult:
    """Result of a profiled operation."""

    operation: str
    duration_seconds: float
    start_time: float
    end_time: float
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["ProfileResult"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Return duration in milliseconds."""
        return self.duration_seconds * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation": self.operation,
            "duration_seconds": self.duration_seconds,
            "duration_ms": self.duration_ms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ProfileStats:
    """Aggregated statistics for profiled operations."""

    operation: str
    count: int
    total_seconds: float
    min_seconds: float
    max_seconds: float
    mean_seconds: float

    @property
    def total_ms(self) -> float:
        """Return total time in milliseconds."""
        return self.total_seconds * 1000

    @property
    def mean_ms(self) -> float:
        """Return mean time in milliseconds."""
        return self.mean_seconds * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation": self.operation,
            "count": self.count,
            "total_seconds": self.total_seconds,
            "total_ms": self.total_ms,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "mean_seconds": self.mean_seconds,
            "mean_ms": self.mean_ms,
        }


class Profiler:
    """
    Profiler for measuring ATP operations.

    Tracks timing for operations organized hierarchically:
    - Suite execution
        - Test execution
            - Run execution
                - Adapter call
                - Evaluator execution
                - Score aggregation

    Thread-safe and supports async operations.
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialize the profiler.

        Args:
            enabled: Whether profiling is enabled. When disabled,
                    operations have minimal overhead.
        """
        self.enabled = enabled
        self._results: list[ProfileResult] = []
        self._stack: list[ProfileResult] = []
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def reset(self) -> None:
        """Reset all collected profiling data."""
        self._results.clear()
        self._stack.clear()
        self._durations.clear()

    @contextmanager
    def profile(
        self,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for profiling synchronous operations.

        Args:
            operation: Name of the operation being profiled.
            metadata: Optional metadata to attach to the result.

        Yields:
            ProfileResult that will be populated on exit.

        Example:
            with profiler.profile("load_suite", {"file": "test.yaml"}):
                suite = loader.load_file("test.yaml")
        """
        if not self.enabled:
            yield None
            return

        start_time = time.perf_counter()
        result = ProfileResult(
            operation=operation,
            duration_seconds=0,
            start_time=start_time,
            end_time=0,
            metadata=metadata or {},
        )

        # Add to current parent's children if in a nested profile
        if self._stack:
            self._stack[-1].children.append(result)
        else:
            self._results.append(result)

        self._stack.append(result)

        try:
            yield result
        finally:
            end_time = time.perf_counter()
            result.end_time = end_time
            result.duration_seconds = end_time - start_time
            self._durations[operation].append(result.duration_seconds)
            self._stack.pop()

    @asynccontextmanager
    async def profile_async(
        self,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Async context manager for profiling async operations.

        Args:
            operation: Name of the operation being profiled.
            metadata: Optional metadata to attach to the result.

        Yields:
            ProfileResult that will be populated on exit.

        Example:
            async with profiler.profile_async("execute_test"):
                result = await adapter.execute(request)
        """
        if not self.enabled:
            yield None
            return

        async with self._lock:
            start_time = time.perf_counter()
            result = ProfileResult(
                operation=operation,
                duration_seconds=0,
                start_time=start_time,
                end_time=0,
                metadata=metadata or {},
            )

            if self._stack:
                self._stack[-1].children.append(result)
            else:
                self._results.append(result)

            self._stack.append(result)

        try:
            yield result
        finally:
            async with self._lock:
                end_time = time.perf_counter()
                result.end_time = end_time
                result.duration_seconds = end_time - start_time
                self._durations[operation].append(result.duration_seconds)
                if self._stack and self._stack[-1] is result:
                    self._stack.pop()

    def get_results(self) -> list[ProfileResult]:
        """Get all top-level profiling results."""
        return list(self._results)

    def get_stats(self, operation: str | None = None) -> dict[str, ProfileStats]:
        """
        Get aggregated statistics for profiled operations.

        Args:
            operation: Optional specific operation to get stats for.
                      If None, returns stats for all operations.

        Returns:
            Dictionary mapping operation names to ProfileStats.
        """
        if operation:
            operations = {operation: self._durations.get(operation, [])}
        else:
            operations = dict(self._durations)

        stats = {}
        for op, durations in operations.items():
            if not durations:
                continue

            stats[op] = ProfileStats(
                operation=op,
                count=len(durations),
                total_seconds=sum(durations),
                min_seconds=min(durations),
                max_seconds=max(durations),
                mean_seconds=sum(durations) / len(durations),
            )

        return stats

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all profiling data.

        Returns:
            Dictionary with:
            - total_duration: Total profiled time in seconds
            - operation_count: Total number of profiled operations
            - stats: Statistics by operation
            - results: Hierarchical results (if any top-level)
        """
        stats = self.get_stats()

        total_duration = sum(s.total_seconds for s in stats.values())
        operation_count = sum(s.count for s in stats.values())

        return {
            "total_duration_seconds": total_duration,
            "total_duration_ms": total_duration * 1000,
            "operation_count": operation_count,
            "stats": {k: v.to_dict() for k, v in stats.items()},
            "results": [r.to_dict() for r in self._results],
        }

    def format_report(self, include_details: bool = False) -> str:
        """
        Format a human-readable profiling report.

        Args:
            include_details: Whether to include detailed hierarchical results.

        Returns:
            Formatted string report.
        """
        lines = ["ATP Profiler Report", "=" * 50, ""]

        stats = self.get_stats()
        if not stats:
            lines.append("No profiling data collected.")
            return "\n".join(lines)

        # Sort by total time descending
        sorted_stats = sorted(
            stats.values(), key=lambda s: s.total_seconds, reverse=True
        )

        lines.append("Operation Statistics:")
        lines.append("-" * 50)
        lines.append(
            f"{'Operation':<30} {'Count':>6} {'Total':>10} {'Mean':>10} {'Min':>8}"
        )
        lines.append("-" * 50)

        for s in sorted_stats:
            lines.append(
                f"{s.operation:<30} {s.count:>6} "
                f"{s.total_ms:>8.1f}ms {s.mean_ms:>8.2f}ms "
                f"{s.min_seconds * 1000:>6.2f}ms"
            )

        total = sum(s.total_seconds for s in sorted_stats)
        lines.append("-" * 50)
        lines.append(
            f"{'Total':<30} {sum(s.count for s in sorted_stats):>6} "
            f"{total * 1000:>8.1f}ms"
        )
        lines.append("")

        if include_details and self._results:
            lines.append("Detailed Results:")
            lines.append("-" * 50)
            for result in self._results:
                self._format_result(result, lines, indent=0)

        return "\n".join(lines)

    def _format_result(
        self,
        result: ProfileResult,
        lines: list[str],
        indent: int = 0,
    ) -> None:
        """Format a single result with its children."""
        prefix = "  " * indent
        meta = ", ".join(f"{k}={v}" for k, v in result.metadata.items())
        meta_str = f" ({meta})" if meta else ""

        lines.append(
            f"{prefix}- {result.operation}: {result.duration_ms:.2f}ms{meta_str}"
        )

        for child in result.children:
            self._format_result(child, lines, indent + 1)


# Global profiler instance
_global_profiler: Profiler | None = None


def get_profiler() -> Profiler:
    """
    Get the global profiler instance.

    Creates a new disabled profiler if none exists.

    Returns:
        Global Profiler instance.
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler(enabled=False)
    return _global_profiler


def set_profiler(profiler: Profiler) -> None:
    """
    Set the global profiler instance.

    Args:
        profiler: Profiler instance to use globally.
    """
    global _global_profiler
    _global_profiler = profiler


def enable_profiling() -> Profiler:
    """
    Enable global profiling and return the profiler.

    Returns:
        Enabled global Profiler instance.
    """
    global _global_profiler
    _global_profiler = Profiler(enabled=True)
    return _global_profiler


def disable_profiling() -> None:
    """Disable global profiling."""
    profiler = get_profiler()
    profiler.enabled = False


@contextmanager
def profile(operation: str, metadata: dict[str, Any] | None = None):
    """
    Profile a synchronous operation using the global profiler.

    Args:
        operation: Name of the operation.
        metadata: Optional metadata.

    Yields:
        ProfileResult or None if profiling is disabled.
    """
    with get_profiler().profile(operation, metadata) as result:
        yield result


@asynccontextmanager
async def profile_async(operation: str, metadata: dict[str, Any] | None = None):
    """
    Profile an async operation using the global profiler.

    Args:
        operation: Name of the operation.
        metadata: Optional metadata.

    Yields:
        ProfileResult or None if profiling is disabled.
    """
    async with get_profiler().profile_async(operation, metadata) as result:
        yield result


def profiled(operation: str | None = None) -> Callable[[F], F]:
    """
    Decorator for profiling functions.

    Args:
        operation: Operation name. Defaults to function name.

    Returns:
        Decorated function.

    Example:
        @profiled("load_suite")
        def load_suite(path):
            ...

        @profiled()
        async def execute_test(test):
            ...
    """

    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with profile_async(op_name):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with profile(op_name):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator
