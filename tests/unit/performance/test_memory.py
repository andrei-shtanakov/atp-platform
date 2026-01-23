"""Tests for the memory module."""

from atp.performance.memory import (
    MemoryDiff,
    MemoryReport,
    MemorySnapshot,
    MemoryTracker,
    enable_memory_tracking,
    get_memory_size,
    get_memory_tracker,
    track_memory,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot."""

    def test_current_mb(self) -> None:
        """Test conversion to megabytes."""
        snapshot = MemorySnapshot(
            timestamp=0.0,
            label="test",
            current_bytes=1024 * 1024,  # 1 MB
            peak_bytes=2048 * 1024,
            traced_blocks=100,
            gc_counts=(10, 5, 2),
        )
        assert snapshot.current_mb == 1.0

    def test_peak_mb(self) -> None:
        """Test peak memory in megabytes."""
        snapshot = MemorySnapshot(
            timestamp=0.0,
            label="test",
            current_bytes=1024 * 1024,
            peak_bytes=2 * 1024 * 1024,  # 2 MB
            traced_blocks=100,
            gc_counts=(10, 5, 2),
        )
        assert snapshot.peak_mb == 2.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        snapshot = MemorySnapshot(
            timestamp=123.456,
            label="test_label",
            current_bytes=1000,
            peak_bytes=2000,
            traced_blocks=50,
            gc_counts=(1, 2, 3),
        )
        d = snapshot.to_dict()

        assert d["label"] == "test_label"
        assert d["current_bytes"] == 1000
        assert d["peak_bytes"] == 2000
        assert d["gc_counts"] == (1, 2, 3)


class TestMemoryDiff:
    """Tests for MemoryDiff."""

    def test_net_change_mb(self) -> None:
        """Test net change in megabytes."""
        diff = MemoryDiff(
            label="test",
            duration_seconds=1.0,
            bytes_allocated=2 * 1024 * 1024,
            bytes_freed=1024 * 1024,
            net_change_bytes=1024 * 1024,
            peak_during=3 * 1024 * 1024,
            gc_collections=(1, 0, 0),
        )
        assert diff.net_change_mb == 1.0

    def test_allocated_mb(self) -> None:
        """Test allocated memory in megabytes."""
        diff = MemoryDiff(
            label="test",
            duration_seconds=1.0,
            bytes_allocated=5 * 1024 * 1024,
            bytes_freed=0,
            net_change_bytes=5 * 1024 * 1024,
            peak_during=5 * 1024 * 1024,
            gc_collections=(0, 0, 0),
        )
        assert diff.allocated_mb == 5.0


class TestMemoryReport:
    """Tests for MemoryReport."""

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        report = MemoryReport(start_time=100.0, end_time=105.0)
        assert report.duration_seconds == 5.0

    def test_total_allocated_bytes(self) -> None:
        """Test total allocated bytes."""
        report = MemoryReport(
            start_time=0.0,
            end_time=1.0,
            diffs=[
                MemoryDiff(
                    label="op1",
                    duration_seconds=0.5,
                    bytes_allocated=1000,
                    bytes_freed=0,
                    net_change_bytes=1000,
                    peak_during=1000,
                    gc_collections=(0, 0, 0),
                ),
                MemoryDiff(
                    label="op2",
                    duration_seconds=0.5,
                    bytes_allocated=2000,
                    bytes_freed=500,
                    net_change_bytes=1500,
                    peak_during=2000,
                    gc_collections=(0, 0, 0),
                ),
            ],
        )
        assert report.total_allocated_bytes == 3000

    def test_net_memory_change_bytes(self) -> None:
        """Test net memory change calculation."""
        report = MemoryReport(
            start_time=0.0,
            end_time=1.0,
            snapshots=[
                MemorySnapshot(
                    timestamp=0.0,
                    label="start",
                    current_bytes=1000,
                    peak_bytes=1000,
                    traced_blocks=10,
                    gc_counts=(0, 0, 0),
                ),
                MemorySnapshot(
                    timestamp=1.0,
                    label="end",
                    current_bytes=3000,
                    peak_bytes=4000,
                    traced_blocks=30,
                    gc_counts=(1, 0, 0),
                ),
            ],
        )
        assert report.net_memory_change_bytes == 2000

    def test_format_report(self) -> None:
        """Test report formatting."""
        report = MemoryReport(
            start_time=0.0,
            end_time=1.0,
            snapshots=[
                MemorySnapshot(
                    timestamp=0.0,
                    label="start",
                    current_bytes=1024 * 1024,
                    peak_bytes=1024 * 1024,
                    traced_blocks=10,
                    gc_counts=(0, 0, 0),
                ),
            ],
        )
        formatted = report.format_report()

        assert "Memory Usage Report" in formatted
        assert "Duration:" in formatted


class TestMemoryTracker:
    """Tests for MemoryTracker."""

    def test_init_enabled(self) -> None:
        """Test tracker initialization with enabled=True."""
        tracker = MemoryTracker(enabled=True)
        assert tracker.enabled is True

    def test_init_disabled(self) -> None:
        """Test tracker initialization with enabled=False."""
        tracker = MemoryTracker(enabled=False)
        assert tracker.enabled is False

    def test_start_and_stop(self) -> None:
        """Test starting and stopping tracking."""
        tracker = MemoryTracker(enabled=True)

        tracker.start()
        # Allocate some memory
        data = [i for i in range(10000)]
        report = tracker.stop()

        assert report is not None
        assert report.duration_seconds > 0
        assert len(report.snapshots) >= 2  # start and end

        # Clean up
        del data

    def test_snapshot(self) -> None:
        """Test taking snapshots."""
        tracker = MemoryTracker(enabled=True)
        tracker.start()

        snap1 = tracker.snapshot("checkpoint1")
        snap2 = tracker.snapshot("checkpoint2")

        tracker.stop()

        assert snap1 is not None
        assert snap2 is not None
        assert snap1.label == "checkpoint1"
        assert snap2.label == "checkpoint2"

    def test_track_context_manager(self) -> None:
        """Test track context manager."""
        tracker = MemoryTracker(enabled=True)
        tracker.start()

        with tracker.track("test_operation"):
            data = [i for i in range(1000)]

        report = tracker.stop()

        # Should have snapshots for the tracked operation
        labels = [s.label for s in report.snapshots]
        assert "test_operation_start" in labels
        assert "test_operation_end" in labels

        del data

    def test_disabled_tracker(self) -> None:
        """Test disabled tracker returns minimal report."""
        tracker = MemoryTracker(enabled=False)

        tracker.start()
        report = tracker.stop()

        assert report is not None
        assert len(report.snapshots) == 0

    def test_force_gc(self) -> None:
        """Test forcing garbage collection."""
        tracker = MemoryTracker(enabled=True)
        collected = tracker.force_gc()

        assert isinstance(collected, tuple)
        assert len(collected) == 3

    def test_get_current_usage(self) -> None:
        """Test getting current memory usage."""
        tracker = MemoryTracker(enabled=True)
        tracker.start()

        current, peak = tracker.get_current_usage()

        tracker.stop()

        assert current >= 0
        assert peak >= 0


class TestGlobalTrackerFunctions:
    """Tests for global tracker functions."""

    def test_get_memory_tracker(self) -> None:
        """Test getting global tracker."""
        tracker = get_memory_tracker()
        assert tracker is not None
        assert isinstance(tracker, MemoryTracker)

    def test_enable_memory_tracking(self) -> None:
        """Test enabling global tracking."""
        tracker = enable_memory_tracking()
        assert tracker.enabled is True

    def test_track_memory_context_manager(self) -> None:
        """Test global track_memory context manager."""
        enable_memory_tracking()
        tracker = get_memory_tracker()
        tracker.start()

        with track_memory("global_test"):
            data = list(range(100))

        tracker.stop()

        # Should not raise
        del data


class TestGetMemorySize:
    """Tests for get_memory_size function."""

    def test_simple_types(self) -> None:
        """Test memory size for simple types."""
        size_int = get_memory_size(42)
        size_str = get_memory_size("hello")
        size_float = get_memory_size(3.14)

        assert size_int > 0
        assert size_str > 0
        assert size_float > 0

    def test_list(self) -> None:
        """Test memory size for list."""
        small_list = [1, 2, 3]
        large_list = list(range(1000))

        size_small = get_memory_size(small_list)
        size_large = get_memory_size(large_list)

        assert size_large > size_small

    def test_dict(self) -> None:
        """Test memory size for dict."""
        small_dict = {"a": 1}
        large_dict = {f"key_{i}": i for i in range(100)}

        size_small = get_memory_size(small_dict)
        size_large = get_memory_size(large_dict)

        assert size_large > size_small

    def test_nested_structure(self) -> None:
        """Test memory size for nested structures."""
        nested = {
            "level1": {
                "level2": {
                    "data": [1, 2, 3, 4, 5],
                },
            },
        }
        size = get_memory_size(nested)
        assert size > 0

    def test_circular_reference(self) -> None:
        """Test memory size with circular reference."""
        a: dict = {"self": None}
        a["self"] = a  # Circular reference

        # Should not hang or raise
        size = get_memory_size(a)
        assert size > 0
