"""Tests for progress tracking module."""

import io
from datetime import datetime

import pytest

from atp.runner.models import ProgressEvent, ProgressEventType
from atp.runner.progress import (
    ParallelProgressTracker,
    ProgressStatus,
    SingleTestProgress,
    create_progress_callback,
)


class TestSingleTestProgress:
    """Tests for SingleTestProgress dataclass."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        progress = SingleTestProgress(test_id="test-001", test_name="Test 1")
        assert progress.status == ProgressStatus.PENDING
        assert progress.current_run == 0
        assert progress.total_runs == 1
        assert progress.start_time is None
        assert progress.end_time is None
        assert progress.error is None

    def test_duration_seconds_none_when_not_started(self) -> None:
        """Test duration is None when test not started."""
        progress = SingleTestProgress(test_id="test-001", test_name="Test 1")
        assert progress.duration_seconds is None

    def test_duration_seconds_calculated(self) -> None:
        """Test duration calculation."""
        progress = SingleTestProgress(
            test_id="test-001",
            test_name="Test 1",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 0, 5),
        )
        assert progress.duration_seconds == 5.0

    def test_status_symbols(self) -> None:
        """Test status symbols for each status."""
        assert (
            SingleTestProgress(
                test_id="test", test_name="Test", status=ProgressStatus.PENDING
            ).status_symbol
            == "○"
        )
        assert (
            SingleTestProgress(
                test_id="test", test_name="Test", status=ProgressStatus.RUNNING
            ).status_symbol
            == "●"
        )
        assert (
            SingleTestProgress(
                test_id="test", test_name="Test", status=ProgressStatus.PASSED
            ).status_symbol
            == "✓"
        )
        assert (
            SingleTestProgress(
                test_id="test", test_name="Test", status=ProgressStatus.FAILED
            ).status_symbol
            == "✗"
        )
        assert (
            SingleTestProgress(
                test_id="test", test_name="Test", status=ProgressStatus.TIMEOUT
            ).status_symbol
            == "⏱"
        )


class TestParallelProgressTracker:
    """Tests for ParallelProgressTracker."""

    @pytest.fixture
    def output(self) -> io.StringIO:
        """Create output buffer."""
        return io.StringIO()

    @pytest.fixture
    def tracker(self, output: io.StringIO) -> ParallelProgressTracker:
        """Create tracker with output buffer."""
        return ParallelProgressTracker(
            max_parallel=3,
            verbose=False,
            use_colors=False,
            output=output,
        )

    def test_init(self, tracker: ParallelProgressTracker) -> None:
        """Test tracker initialization."""
        assert tracker.max_parallel == 3
        assert tracker.verbose is False
        assert tracker.use_colors is False

    def test_handle_suite_started(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test suite started event handling."""
        event = ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            suite_name="test-suite",
            total_tests=5,
            details={"agent_name": "test-agent", "runs_per_test": 3},
        )
        tracker.on_progress(event)

        result = output.getvalue()
        assert "test-suite" in result
        assert "test-agent" in result
        assert "5" in result
        assert "parallel" in result
        assert "max 3" in result

    def test_handle_suite_completed_success(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test suite completed event for successful run."""
        # First start the suite
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_STARTED,
                suite_name="test-suite",
                total_tests=2,
                details={"agent_name": "test-agent", "runs_per_test": 1},
            )
        )
        output.truncate(0)
        output.seek(0)

        # Then complete it
        event = ProgressEvent(
            event_type=ProgressEventType.SUITE_COMPLETED,
            suite_name="test-suite",
            success=True,
            details={
                "passed_tests": 2,
                "failed_tests": 0,
                "success_rate": 1.0,
                "duration_seconds": 5.5,
            },
        )
        tracker.on_progress(event)

        result = output.getvalue()
        assert "PASSED" in result
        assert "2/2" in result
        assert "100.0%" in result

    def test_handle_suite_completed_failure(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test suite completed event for failed run."""
        tracker._total_tests = 3  # Set manually for test
        event = ProgressEvent(
            event_type=ProgressEventType.SUITE_COMPLETED,
            suite_name="test-suite",
            success=False,
            details={
                "passed_tests": 1,
                "failed_tests": 2,
                "success_rate": 0.333,
                "duration_seconds": 10.0,
            },
        )
        tracker.on_progress(event)

        result = output.getvalue()
        assert "FAILED" in result
        assert "1/3" in result

    def test_handle_test_started(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test test started event handling."""
        tracker._total_tests = 5  # Set manually for test
        event = ProgressEvent(
            event_type=ProgressEventType.TEST_STARTED,
            test_id="test-001",
            test_name="First Test",
            total_runs=3,
        )
        tracker.on_progress(event)

        result = output.getvalue()
        assert "First Test" in result
        assert "started" in result
        assert tracker._tests["test-001"].status == ProgressStatus.RUNNING

    def test_handle_test_completed_passed(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test test completed event for passed test."""
        # Start the test first
        tracker._total_tests = 1
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_STARTED,
                test_id="test-001",
                test_name="First Test",
                total_runs=1,
            )
        )
        output.truncate(0)
        output.seek(0)

        # Complete it
        event = ProgressEvent(
            event_type=ProgressEventType.TEST_COMPLETED,
            test_id="test-001",
            test_name="First Test",
            success=True,
        )
        tracker.on_progress(event)

        assert tracker._tests["test-001"].status == ProgressStatus.PASSED

    def test_handle_test_completed_failed(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test test completed event for failed test."""
        tracker._total_tests = 1
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_STARTED,
                test_id="test-001",
                test_name="First Test",
                total_runs=1,
            )
        )

        event = ProgressEvent(
            event_type=ProgressEventType.TEST_FAILED,
            test_id="test-001",
            test_name="First Test",
            success=False,
            error="Test assertion failed",
        )
        tracker.on_progress(event)

        assert tracker._tests["test-001"].status == ProgressStatus.FAILED
        assert "assertion failed" in output.getvalue()

    def test_handle_test_timeout(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test test timeout event handling."""
        tracker._total_tests = 1
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_STARTED,
                test_id="test-001",
                test_name="First Test",
                total_runs=1,
            )
        )

        event = ProgressEvent(
            event_type=ProgressEventType.TEST_TIMEOUT,
            test_id="test-001",
            test_name="First Test",
            success=False,
            error="Timeout after 30s",
        )
        tracker.on_progress(event)

        assert tracker._tests["test-001"].status == ProgressStatus.TIMEOUT

    def test_handle_run_started(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test run started event handling."""
        # Start test first
        tracker._total_tests = 1
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_STARTED,
                test_id="test-001",
                test_name="First Test",
                total_runs=3,
            )
        )

        event = ProgressEvent(
            event_type=ProgressEventType.RUN_STARTED,
            test_id="test-001",
            run_number=2,
            total_runs=3,
        )
        tracker.on_progress(event)

        assert tracker._tests["test-001"].current_run == 2

    def test_summary(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test summary property."""
        tracker._total_tests = 4

        # Add tests with different statuses
        tracker._tests["test-001"] = SingleTestProgress(
            test_id="test-001", test_name="Test 1", status=ProgressStatus.PASSED
        )
        tracker._tests["test-002"] = SingleTestProgress(
            test_id="test-002", test_name="Test 2", status=ProgressStatus.FAILED
        )
        tracker._tests["test-003"] = SingleTestProgress(
            test_id="test-003", test_name="Test 3", status=ProgressStatus.TIMEOUT
        )
        tracker._tests["test-004"] = SingleTestProgress(
            test_id="test-004", test_name="Test 4", status=ProgressStatus.RUNNING
        )

        summary = tracker.summary
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["timeout"] == 1
        assert summary["running"] == 1
        assert summary["total"] == 4

    def test_verbose_mode_run_events(self, output: io.StringIO) -> None:
        """Test verbose mode shows run events."""
        tracker = ParallelProgressTracker(
            max_parallel=1,
            verbose=True,
            use_colors=False,
            output=output,
        )
        tracker._total_tests = 1

        # Start test
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_STARTED,
                test_id="test-001",
                test_name="Test",
                total_runs=2,
            )
        )

        # Start run
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.RUN_STARTED,
                test_id="test-001",
                run_number=1,
                total_runs=2,
            )
        )

        result = output.getvalue()
        assert "Run 1/2 started" in result

    def test_format_duration(
        self, tracker: ParallelProgressTracker, output: io.StringIO
    ) -> None:
        """Test duration formatting."""
        assert tracker._format_duration(None) == "..."
        assert tracker._format_duration(5.5) == "5.5s"
        assert tracker._format_duration(65.0) == "1m5.0s"
        assert tracker._format_duration(125.5) == "2m5.5s"


class TestCreateProgressCallback:
    """Tests for create_progress_callback factory."""

    def test_create_default(self) -> None:
        """Test creating default callback."""
        callback = create_progress_callback()
        assert callable(callback)

    def test_create_with_options(self) -> None:
        """Test creating callback with options."""
        output = io.StringIO()
        callback = create_progress_callback(
            max_parallel=5,
            verbose=True,
            use_colors=False,
            output=output,
        )
        assert callable(callback)

        # Test callback works
        event = ProgressEvent(
            event_type=ProgressEventType.SUITE_STARTED,
            suite_name="test-suite",
            total_tests=2,
            details={"agent_name": "agent", "runs_per_test": 1},
        )
        callback(event)

        assert "test-suite" in output.getvalue()


class TestColorHandling:
    """Tests for color handling."""

    def test_colors_disabled_on_non_tty(self) -> None:
        """Test colors are disabled for non-tty output."""
        output = io.StringIO()
        tracker = ParallelProgressTracker(
            use_colors=True,  # Request colors
            output=output,  # StringIO is not a TTY
        )
        # Colors should be disabled because StringIO is not a TTY
        assert tracker.use_colors is False

    def test_color_method_no_color(self) -> None:
        """Test _color method when colors disabled."""
        output = io.StringIO()
        tracker = ParallelProgressTracker(use_colors=False, output=output)
        result = tracker._color("test", "\033[32m")
        assert result == "test"  # No color codes

    def test_color_method_with_color(self) -> None:
        """Test _color method when colors enabled."""
        output = io.StringIO()
        tracker = ParallelProgressTracker(use_colors=False, output=output)
        # Force colors on for test
        tracker.use_colors = True
        result = tracker._color("test", "\033[32m")
        assert "\033[32m" in result
        assert "\033[0m" in result


class TestParallelTestScenarios:
    """Integration tests for parallel test scenarios."""

    @pytest.fixture
    def output(self) -> io.StringIO:
        """Create output buffer."""
        return io.StringIO()

    def test_full_parallel_suite_execution(self, output: io.StringIO) -> None:
        """Test tracking a full parallel suite execution."""
        tracker = ParallelProgressTracker(
            max_parallel=3,
            verbose=False,
            use_colors=False,
            output=output,
        )

        # Suite started
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_STARTED,
                suite_name="parallel-suite",
                total_tests=3,
                details={"agent_name": "test-agent", "runs_per_test": 2},
            )
        )

        # Start all tests (parallel)
        for i in range(1, 4):
            tracker.on_progress(
                ProgressEvent(
                    event_type=ProgressEventType.TEST_STARTED,
                    test_id=f"test-{i:03d}",
                    test_name=f"Test {i}",
                    total_runs=2,
                )
            )

        # Complete tests in different order (simulating parallel)
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_COMPLETED,
                test_id="test-002",
                test_name="Test 2",
                success=True,
            )
        )
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_FAILED,
                test_id="test-001",
                test_name="Test 1",
                success=False,
                error="Assertion error",
            )
        )
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.TEST_COMPLETED,
                test_id="test-003",
                test_name="Test 3",
                success=True,
            )
        )

        # Suite completed
        tracker.on_progress(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_COMPLETED,
                suite_name="parallel-suite",
                success=False,
                details={
                    "passed_tests": 2,
                    "failed_tests": 1,
                    "success_rate": 0.667,
                    "duration_seconds": 8.5,
                },
            )
        )

        result = output.getvalue()

        # Verify output contains expected elements
        assert "parallel-suite" in result
        assert "Test 1" in result
        assert "Test 2" in result
        assert "Test 3" in result
        assert "PASSED" in result
        assert "FAILED" in result
        assert "Assertion error" in result

        # Verify summary
        summary = tracker.summary
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["total"] == 3
