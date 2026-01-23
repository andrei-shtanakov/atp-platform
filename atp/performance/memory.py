"""Memory usage tracking and auditing for ATP.

Provides tools for:
- Tracking memory usage during test execution
- Identifying memory-intensive operations
- Detecting potential memory leaks
- Generating memory usage reports
"""

import gc
import logging
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""

    timestamp: float
    label: str
    current_bytes: int
    peak_bytes: int
    traced_blocks: int
    gc_counts: tuple[int, int, int]  # (gen0, gen1, gen2)

    @property
    def current_mb(self) -> float:
        """Current memory in megabytes."""
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "current_bytes": self.current_bytes,
            "current_mb": self.current_mb,
            "peak_bytes": self.peak_bytes,
            "peak_mb": self.peak_mb,
            "traced_blocks": self.traced_blocks,
            "gc_counts": self.gc_counts,
        }


@dataclass
class MemoryDiff:
    """Difference between two memory snapshots."""

    label: str
    duration_seconds: float
    bytes_allocated: int
    bytes_freed: int
    net_change_bytes: int
    peak_during: int
    gc_collections: tuple[int, int, int]

    @property
    def net_change_mb(self) -> float:
        """Net change in megabytes."""
        return self.net_change_bytes / (1024 * 1024)

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in megabytes."""
        return self.bytes_allocated / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "duration_seconds": self.duration_seconds,
            "bytes_allocated": self.bytes_allocated,
            "bytes_freed": self.bytes_freed,
            "net_change_bytes": self.net_change_bytes,
            "net_change_mb": self.net_change_mb,
            "allocated_mb": self.allocated_mb,
            "peak_during": self.peak_during,
            "gc_collections": self.gc_collections,
        }


@dataclass
class MemoryReport:
    """Complete memory usage report."""

    start_time: float
    end_time: float
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    diffs: list[MemoryDiff] = field(default_factory=list)
    top_allocators: list[tuple[str, int]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.end_time - self.start_time

    @property
    def total_allocated_bytes(self) -> int:
        """Total bytes allocated during tracking."""
        return sum(d.bytes_allocated for d in self.diffs)

    @property
    def total_freed_bytes(self) -> int:
        """Total bytes freed during tracking."""
        return sum(d.bytes_freed for d in self.diffs)

    @property
    def net_memory_change_bytes(self) -> int:
        """Net memory change."""
        if not self.snapshots:
            return 0
        return self.snapshots[-1].current_bytes - self.snapshots[0].current_bytes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "total_allocated_bytes": self.total_allocated_bytes,
            "total_freed_bytes": self.total_freed_bytes,
            "net_memory_change_bytes": self.net_memory_change_bytes,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "diffs": [d.to_dict() for d in self.diffs],
            "top_allocators": self.top_allocators,
        }

    def format_report(self) -> str:
        """Format a human-readable report."""
        lines = [
            "Memory Usage Report",
            "=" * 50,
            "",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Net memory change: {self.net_memory_change_bytes / (1024 * 1024):.2f} MB",
            f"Total allocated: {self.total_allocated_bytes / (1024 * 1024):.2f} MB",
            f"Total freed: {self.total_freed_bytes / (1024 * 1024):.2f} MB",
            "",
        ]

        if self.snapshots:
            lines.append("Memory Timeline:")
            lines.append("-" * 50)
            for snap in self.snapshots:
                lines.append(
                    f"  {snap.label}: {snap.current_mb:.2f} MB "
                    f"(peak: {snap.peak_mb:.2f} MB)"
                )
            lines.append("")

        if self.diffs:
            lines.append("Memory Changes:")
            lines.append("-" * 50)
            for diff in self.diffs:
                sign = "+" if diff.net_change_bytes >= 0 else ""
                lines.append(
                    f"  {diff.label}: {sign}{diff.net_change_mb:.2f} MB "
                    f"({diff.duration_seconds:.3f}s)"
                )
            lines.append("")

        if self.top_allocators:
            lines.append("Top Memory Allocators:")
            lines.append("-" * 50)
            for trace, size in self.top_allocators[:10]:
                lines.append(f"  {size / 1024:.1f} KB: {trace}")
            lines.append("")

        return "\n".join(lines)


class MemoryTracker:
    """
    Tracks memory usage during ATP operations.

    Uses tracemalloc for detailed memory tracking and provides
    snapshots, diffs, and reports.
    """

    def __init__(self, enabled: bool = True, nframes: int = 5) -> None:
        """
        Initialize memory tracker.

        Args:
            enabled: Whether tracking is enabled.
            nframes: Number of stack frames to capture in traces.
        """
        self.enabled = enabled
        self._nframes = nframes
        self._snapshots: list[MemorySnapshot] = []
        self._diffs: list[MemoryDiff] = []
        self._tracking = False
        self._start_snapshot: tracemalloc.Snapshot | None = None
        self._start_time: float = 0
        self._last_snapshot: MemorySnapshot | None = None

    def start(self) -> None:
        """Start memory tracking."""
        if not self.enabled:
            return

        if self._tracking:
            logger.warning("Memory tracking already started")
            return

        # Start tracemalloc if not already running
        if not tracemalloc.is_tracing():
            tracemalloc.start(self._nframes)

        self._tracking = True
        self._start_time = time.time()
        self._start_snapshot = tracemalloc.take_snapshot()
        self._snapshots.clear()
        self._diffs.clear()

        # Take initial snapshot
        self.snapshot("start")
        logger.debug("Memory tracking started")

    def stop(self) -> MemoryReport:
        """
        Stop tracking and return a report.

        Returns:
            MemoryReport with all collected data.
        """
        if not self.enabled or not self._tracking:
            return MemoryReport(
                start_time=self._start_time,
                end_time=time.time(),
            )

        # Take final snapshot
        self.snapshot("end")

        # Get top allocators
        current = tracemalloc.take_snapshot()
        top_stats = current.statistics("lineno")
        top_allocators = [(str(stat.traceback), stat.size) for stat in top_stats[:20]]

        self._tracking = False
        end_time = time.time()

        report = MemoryReport(
            start_time=self._start_time,
            end_time=end_time,
            snapshots=list(self._snapshots),
            diffs=list(self._diffs),
            top_allocators=top_allocators,
        )

        logger.debug("Memory tracking stopped, report generated")
        return report

    def snapshot(self, label: str) -> MemorySnapshot | None:
        """
        Take a memory snapshot.

        Args:
            label: Label for this snapshot point.

        Returns:
            MemorySnapshot if tracking is enabled, None otherwise.
        """
        if not self.enabled or not self._tracking:
            return None

        current, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            current_bytes=current,
            peak_bytes=peak,
            traced_blocks=len(snap.traces),
            gc_counts=gc.get_count(),
        )

        # Calculate diff from last snapshot
        if self._last_snapshot is not None:
            diff = MemoryDiff(
                label=f"{self._last_snapshot.label} -> {label}",
                duration_seconds=snapshot.timestamp - self._last_snapshot.timestamp,
                bytes_allocated=max(0, current - self._last_snapshot.current_bytes),
                bytes_freed=max(0, self._last_snapshot.current_bytes - current),
                net_change_bytes=current - self._last_snapshot.current_bytes,
                peak_during=max(peak, self._last_snapshot.peak_bytes),
                gc_collections=tuple(
                    a - b
                    for a, b in zip(snapshot.gc_counts, self._last_snapshot.gc_counts)
                ),  # type: ignore[assignment]
            )
            self._diffs.append(diff)

        self._snapshots.append(snapshot)
        self._last_snapshot = snapshot

        return snapshot

    @contextmanager
    def track(self, label: str):
        """
        Context manager for tracking a specific operation.

        Args:
            label: Label for this operation.

        Yields:
            MemoryDiff result (populated after context exit).

        Example:
            with tracker.track("load_suite"):
                suite = loader.load_file("tests.yaml")
        """
        if not self.enabled:
            yield None
            return

        was_tracking = self._tracking
        if not was_tracking:
            self.start()

        self.snapshot(f"{label}_start")

        try:
            yield
        finally:
            self.snapshot(f"{label}_end")

            if not was_tracking:
                self.stop()

    def get_current_usage(self) -> tuple[int, int]:
        """
        Get current memory usage.

        Returns:
            Tuple of (current_bytes, peak_bytes).
        """
        if tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()
        return (0, 0)

    def force_gc(self) -> tuple[int, int, int]:
        """
        Force garbage collection.

        Returns:
            Tuple of objects collected per generation.
        """
        return (gc.collect(0), gc.collect(1), gc.collect(2))

    def get_object_counts(self) -> dict[str, int]:
        """
        Get counts of tracked Python objects by type.

        Returns:
            Dictionary mapping type names to counts.
        """
        gc.collect()

        counts: dict[str, int] = {}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            counts[type_name] = counts.get(type_name, 0) + 1

        # Sort by count descending
        sorted_counts = dict(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)[:50]
        )
        return sorted_counts


# Global tracker instance
_global_tracker: MemoryTracker | None = None


def get_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker (disabled by default)."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker(enabled=False)
    return _global_tracker


def enable_memory_tracking(nframes: int = 5) -> MemoryTracker:
    """Enable global memory tracking."""
    global _global_tracker
    _global_tracker = MemoryTracker(enabled=True, nframes=nframes)
    return _global_tracker


def disable_memory_tracking() -> None:
    """Disable global memory tracking."""
    tracker = get_memory_tracker()
    tracker.enabled = False


@contextmanager
def track_memory(label: str = "operation"):
    """
    Convenience context manager for memory tracking.

    Args:
        label: Label for the operation.

    Example:
        with track_memory("test_execution"):
            result = await orchestrator.run_suite(suite, agent_name)
    """
    with get_memory_tracker().track(label):
        yield


def get_memory_size(obj: Any, seen: set | None = None) -> int:
    """
    Recursively calculate the memory size of an object.

    Args:
        obj: Object to measure.
        seen: Set of already seen object ids (for cycle detection).

    Returns:
        Approximate size in bytes.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(
            get_memory_size(k, seen) + get_memory_size(v, seen) for k, v in obj.items()
        )
    elif hasattr(obj, "__dict__"):
        size += get_memory_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum(get_memory_size(i, seen) for i in obj)
        except TypeError:
            pass

    return size
