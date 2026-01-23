"""Progress tracking for parallel test execution."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TextIO

from atp.runner.models import ProgressEvent, ProgressEventType


class ProgressStatus(str, Enum):
    """Status of a test in the progress tracker."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SingleTestProgress:
    """Track progress of a single test."""

    test_id: str
    test_name: str
    status: ProgressStatus = ProgressStatus.PENDING
    current_run: int = 0
    total_runs: int = 1
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def status_symbol(self) -> str:
        """Get status symbol for display."""
        return {
            ProgressStatus.PENDING: "○",
            ProgressStatus.RUNNING: "●",
            ProgressStatus.PASSED: "✓",
            ProgressStatus.FAILED: "✗",
            ProgressStatus.TIMEOUT: "⏱",
        }[self.status]

    @property
    def status_color(self) -> str:
        """Get ANSI color code for status."""
        return {
            ProgressStatus.PENDING: "\033[90m",  # Gray
            ProgressStatus.RUNNING: "\033[33m",  # Yellow
            ProgressStatus.PASSED: "\033[32m",  # Green
            ProgressStatus.FAILED: "\033[31m",  # Red
            ProgressStatus.TIMEOUT: "\033[35m",  # Magenta
        }[self.status]


@dataclass
class ParallelProgressTracker:
    """
    Track progress of multiple tests running in parallel.

    Provides real-time progress updates for concurrent test execution.
    """

    max_parallel: int = 1
    verbose: bool = False
    use_colors: bool = True
    output: TextIO = field(default_factory=lambda: sys.stdout)

    _tests: dict[str, SingleTestProgress] = field(default_factory=dict)
    _test_order: list[str] = field(default_factory=list)
    _suite_name: str | None = None
    _suite_start: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _completed_count: int = 0
    _total_tests: int = 0

    def __post_init__(self) -> None:
        """Initialize progress tracker."""
        # Check if output supports colors
        if self.use_colors and hasattr(self.output, "isatty"):
            self.use_colors = self.output.isatty()

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}\033[0m"
        return text

    def _write(self, text: str) -> None:
        """Write to output (caller must hold lock)."""
        self.output.write(text)
        self.output.flush()

    def _format_duration(self, seconds: float | None) -> str:
        """Format duration for display."""
        if seconds is None:
            return "..."
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"

    def on_progress(self, event: ProgressEvent) -> None:
        """
        Handle a progress event.

        This method is designed to be used as a progress callback.

        Args:
            event: Progress event from the orchestrator.
        """
        with self._lock:
            if event.event_type == ProgressEventType.SUITE_STARTED:
                self._handle_suite_started(event)
            elif event.event_type == ProgressEventType.SUITE_COMPLETED:
                self._handle_suite_completed(event)
            elif event.event_type == ProgressEventType.TEST_STARTED:
                self._handle_test_started(event)
            elif event.event_type in (
                ProgressEventType.TEST_COMPLETED,
                ProgressEventType.TEST_FAILED,
                ProgressEventType.TEST_TIMEOUT,
            ):
                self._handle_test_completed(event)
            elif event.event_type == ProgressEventType.RUN_STARTED:
                self._handle_run_started(event)
            elif event.event_type == ProgressEventType.RUN_COMPLETED:
                self._handle_run_completed(event)
            elif event.event_type == ProgressEventType.AGENT_EVENT:
                if self.verbose:
                    self._handle_agent_event(event)

    def _handle_suite_started(self, event: ProgressEvent) -> None:
        """Handle suite started event."""
        self._suite_name = event.suite_name
        self._suite_start = datetime.now()
        self._total_tests = event.total_tests or 0
        self._completed_count = 0

        agent_name = event.details.get("agent_name", "unknown")
        runs_per_test = event.details.get("runs_per_test", 1)

        mode = "parallel" if self.max_parallel > 1 else "sequential"
        parallel_info = (
            f" (max {self.max_parallel} concurrent)" if self.max_parallel > 1 else ""
        )
        self._write(
            f"\n{'═' * 60}\n"
            f"Suite: {self._suite_name}\n"
            f"Agent: {agent_name}\n"
            f"Tests: {self._total_tests}, Runs per test: {runs_per_test}\n"
            f"Mode: {mode}{parallel_info}\n"
            f"{'═' * 60}\n\n"
        )

    def _handle_suite_completed(self, event: ProgressEvent) -> None:
        """Handle suite completed event."""
        passed = event.details.get("passed_tests", 0)
        success_rate = event.details.get("success_rate", 0.0)
        duration = event.details.get("duration_seconds", 0.0)

        if event.success:
            status_text = self._color("PASSED", "\033[32m")
        else:
            status_text = self._color("FAILED", "\033[31m")

        self._write(
            f"\n{'─' * 60}\n"
            f"Result: {status_text}\n"
            f"Passed: {passed}/{self._total_tests} ({success_rate * 100:.1f}%)\n"
            f"Duration: {self._format_duration(duration)}\n"
            f"{'═' * 60}\n\n"
        )

    def _handle_test_started(self, event: ProgressEvent) -> None:
        """Handle test started event."""
        if event.test_id is None:
            return

        test_progress = SingleTestProgress(
            test_id=event.test_id,
            test_name=event.test_name or event.test_id,
            status=ProgressStatus.RUNNING,
            total_runs=event.total_runs or 1,
            start_time=datetime.now(),
        )
        self._tests[event.test_id] = test_progress
        self._test_order.append(event.test_id)

        self._print_test_status(test_progress, "started")

    def _handle_test_completed(self, event: ProgressEvent) -> None:
        """Handle test completed event."""
        if event.test_id is None or event.test_id not in self._tests:
            return

        test_progress = self._tests[event.test_id]
        test_progress.end_time = datetime.now()

        if event.event_type == ProgressEventType.TEST_TIMEOUT:
            test_progress.status = ProgressStatus.TIMEOUT
        elif event.success:
            test_progress.status = ProgressStatus.PASSED
        else:
            test_progress.status = ProgressStatus.FAILED

        test_progress.error = event.error
        self._completed_count += 1

        self._print_test_status(test_progress, "completed")

    def _handle_run_started(self, event: ProgressEvent) -> None:
        """Handle run started event."""
        if event.test_id is None or event.test_id not in self._tests:
            return

        test_progress = self._tests[event.test_id]
        test_progress.current_run = event.run_number or 1

        if self.verbose:
            run_info = f"{test_progress.current_run}/{test_progress.total_runs}"
            self._write(f"    Run {run_info} started\n")

    def _handle_run_completed(self, event: ProgressEvent) -> None:
        """Handle run completed event."""
        if not self.verbose:
            return

        if event.test_id is None or event.test_id not in self._tests:
            return

        test_progress = self._tests[event.test_id]
        status = "✓" if event.success else "✗"
        duration = event.details.get("duration_seconds", 0.0)

        self._write(
            f"    Run {test_progress.current_run}/{test_progress.total_runs} "
            f"{status} ({self._format_duration(duration)})\n"
        )

    def _handle_agent_event(self, event: ProgressEvent) -> None:
        """Handle agent event (verbose mode only)."""
        if event.agent_event is None:
            return

        event_type = event.agent_event.event_type.value
        self._write(f"      [Agent] {event_type}\n")

    def _print_test_status(self, test: SingleTestProgress, action: str) -> None:
        """Print test status line."""
        symbol = self._color(test.status_symbol, test.status_color)
        name = test.test_name[:40].ljust(40)
        duration = self._format_duration(test.duration_seconds)

        if action == "started":
            progress = f"[{self._completed_count}/{self._total_tests}]"
            self._write(f"  {symbol} {name} {progress} {action}...\n")
        else:
            runs_info = ""
            if test.total_runs > 1:
                details = test.error or ""
                if "successful_runs" in details:
                    runs_info = f" ({details})"

            status = test.status.value.upper()
            self._write(f"  {symbol} {name} [{duration}] {status}{runs_info}\n")

            if test.error and test.status != ProgressStatus.PASSED:
                error_lines = test.error.split("\n")[:3]
                for line in error_lines:
                    self._write(f"      {self._color(line, '\033[31m')}\n")

    def get_callback(self) -> Callable[[ProgressEvent], None]:
        """Get the progress callback function."""
        return self.on_progress

    @property
    def summary(self) -> dict[str, int]:
        """Get summary of test results."""
        tests = self._tests.values()
        passed = sum(1 for t in tests if t.status == ProgressStatus.PASSED)
        failed = sum(1 for t in tests if t.status == ProgressStatus.FAILED)
        timeout = sum(1 for t in tests if t.status == ProgressStatus.TIMEOUT)
        pending = sum(1 for t in tests if t.status == ProgressStatus.PENDING)
        running = sum(1 for t in tests if t.status == ProgressStatus.RUNNING)

        return {
            "passed": passed,
            "failed": failed,
            "timeout": timeout,
            "pending": pending,
            "running": running,
            "total": len(self._tests),
        }


def create_progress_callback(
    max_parallel: int = 1,
    verbose: bool = False,
    use_colors: bool = True,
    output: TextIO | None = None,
) -> Callable[[ProgressEvent], None]:
    """
    Create a progress callback for the test orchestrator.

    Args:
        max_parallel: Maximum number of parallel tests.
        verbose: Whether to show detailed progress.
        use_colors: Whether to use ANSI colors.
        output: Output stream (defaults to stdout).

    Returns:
        Progress callback function.
    """
    tracker = ParallelProgressTracker(
        max_parallel=max_parallel,
        verbose=verbose,
        use_colors=use_colors,
        output=output or sys.stdout,
    )
    return tracker.get_callback()
