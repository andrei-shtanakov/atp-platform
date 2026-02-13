"""Rich Live display for streaming test progress."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from atp.runner.models import ProgressEvent, ProgressEventType


class LiveDisplayState:
    """Thread-safe state for the live display.

    Tracks suite and per-test progress, token/cost accumulation,
    and evaluator scores as they stream in.
    """

    def __init__(self) -> None:
        """Initialize the live display state."""
        self._lock = threading.Lock()
        self.suite_name: str | None = None
        self.agent_name: str | None = None
        self.total_tests: int = 0
        self.completed_tests: int = 0
        self.passed_tests: int = 0
        self.failed_tests: int = 0
        self.suite_start: datetime | None = None
        self.tests: dict[str, _TestState] = {}
        self.test_order: list[str] = []
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.finished: bool = False

    def update(self, event: ProgressEvent) -> None:
        """Process a progress event and update state.

        Args:
            event: Progress event from the orchestrator.
        """
        with self._lock:
            self._handle_event(event)

    def _handle_event(self, event: ProgressEvent) -> None:
        """Handle a single event (caller holds lock)."""
        etype = event.event_type

        if etype == ProgressEventType.SUITE_STARTED:
            self.suite_name = event.suite_name
            self.agent_name = event.details.get("agent_name")
            self.total_tests = event.total_tests or 0
            self.suite_start = datetime.now(UTC)

        elif etype == ProgressEventType.TEST_STARTED:
            if event.test_id is not None:
                self.tests[event.test_id] = _TestState(
                    test_id=event.test_id,
                    test_name=event.test_name or event.test_id,
                    total_runs=event.total_runs or 1,
                    start_time=datetime.now(UTC),
                )
                self.test_order.append(event.test_id)

        elif etype in (
            ProgressEventType.TEST_COMPLETED,
            ProgressEventType.TEST_FAILED,
            ProgressEventType.TEST_TIMEOUT,
        ):
            if event.test_id and event.test_id in self.tests:
                ts = self.tests[event.test_id]
                ts.end_time = datetime.now(UTC)
                if etype == ProgressEventType.TEST_TIMEOUT:
                    ts.status = "timeout"
                elif event.success:
                    ts.status = "passed"
                else:
                    ts.status = "failed"
                ts.error = event.error
                self.completed_tests += 1
                if ts.status == "passed":
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1
                # Accumulate tokens/cost from details
                self.total_tokens += event.details.get("tokens", 0)
                self.total_cost += event.details.get("cost_usd", 0.0)

        elif etype == ProgressEventType.RUN_STARTED:
            if event.test_id and event.test_id in self.tests:
                ts = self.tests[event.test_id]
                ts.current_run = event.run_number or 1

        elif etype == ProgressEventType.RUN_COMPLETED:
            if event.test_id and event.test_id in self.tests:
                ts = self.tests[event.test_id]
                if event.success:
                    ts.successful_runs += 1
                run_tokens = event.details.get("tokens", 0)
                run_cost = event.details.get("cost_usd", 0.0)
                self.total_tokens += run_tokens
                self.total_cost += run_cost

        elif etype == ProgressEventType.SUITE_COMPLETED:
            self.finished = True

    def snapshot(self) -> _Snapshot:
        """Take a thread-safe snapshot of the current state.

        Returns:
            Immutable snapshot of the display state.
        """
        with self._lock:
            tests = []
            for tid in self.test_order:
                ts = self.tests[tid]
                tests.append(
                    _TestSnap(
                        test_id=ts.test_id,
                        test_name=ts.test_name,
                        status=ts.status,
                        current_run=ts.current_run,
                        total_runs=ts.total_runs,
                        successful_runs=ts.successful_runs,
                        duration=_duration(ts.start_time, ts.end_time),
                        error=ts.error,
                    )
                )
            return _Snapshot(
                suite_name=self.suite_name or "",
                agent_name=self.agent_name or "",
                total_tests=self.total_tests,
                completed_tests=self.completed_tests,
                passed_tests=self.passed_tests,
                failed_tests=self.failed_tests,
                elapsed=_duration(self.suite_start, None),
                total_tokens=self.total_tokens,
                total_cost=self.total_cost,
                tests=tests,
                finished=self.finished,
            )


class _TestState:
    """Mutable per-test state."""

    __slots__ = (
        "test_id",
        "test_name",
        "status",
        "current_run",
        "total_runs",
        "successful_runs",
        "start_time",
        "end_time",
        "error",
    )

    def __init__(
        self,
        test_id: str,
        test_name: str,
        total_runs: int,
        start_time: datetime,
    ) -> None:
        self.test_id = test_id
        self.test_name = test_name
        self.status = "running"
        self.current_run = 0
        self.total_runs = total_runs
        self.successful_runs = 0
        self.start_time = start_time
        self.end_time: datetime | None = None
        self.error: str | None = None


class _TestSnap:
    """Immutable test snapshot."""

    __slots__ = (
        "test_id",
        "test_name",
        "status",
        "current_run",
        "total_runs",
        "successful_runs",
        "duration",
        "error",
    )

    def __init__(
        self,
        test_id: str,
        test_name: str,
        status: str,
        current_run: int,
        total_runs: int,
        successful_runs: int,
        duration: float,
        error: str | None,
    ) -> None:
        self.test_id = test_id
        self.test_name = test_name
        self.status = status
        self.current_run = current_run
        self.total_runs = total_runs
        self.successful_runs = successful_runs
        self.duration = duration
        self.error = error


class _Snapshot:
    """Immutable snapshot of the full display state."""

    __slots__ = (
        "suite_name",
        "agent_name",
        "total_tests",
        "completed_tests",
        "passed_tests",
        "failed_tests",
        "elapsed",
        "total_tokens",
        "total_cost",
        "tests",
        "finished",
    )

    def __init__(
        self,
        suite_name: str,
        agent_name: str,
        total_tests: int,
        completed_tests: int,
        passed_tests: int,
        failed_tests: int,
        elapsed: float,
        total_tokens: int,
        total_cost: float,
        tests: list[_TestSnap],
        finished: bool,
    ) -> None:
        self.suite_name = suite_name
        self.agent_name = agent_name
        self.total_tests = total_tests
        self.completed_tests = completed_tests
        self.passed_tests = passed_tests
        self.failed_tests = failed_tests
        self.elapsed = elapsed
        self.total_tokens = total_tokens
        self.total_cost = total_cost
        self.tests = tests
        self.finished = finished


def _duration(start: datetime | None, end: datetime | None) -> float:
    """Calculate duration in seconds."""
    if start is None:
        return 0.0
    actual_end = end or datetime.now(UTC)
    return (actual_end - start).total_seconds()


def _format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.1f}s"


_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "running": ("[bold yellow]", "..."),
    "passed": ("[bold green]", "PASS"),
    "failed": ("[bold red]", "FAIL"),
    "timeout": ("[bold magenta]", "TIME"),
}


def build_table(snap: _Snapshot) -> Table:
    """Build a Rich Table from a display snapshot.

    Args:
        snap: Current display state snapshot.

    Returns:
        Rich Table renderable for the live display.
    """
    table = Table(
        title=f"Suite: {snap.suite_name}  |  Agent: {snap.agent_name}",
        title_style="bold",
        expand=True,
        show_lines=False,
    )
    table.add_column("Status", width=6, justify="center")
    table.add_column("Test", ratio=3)
    table.add_column("Run", width=8, justify="center")
    table.add_column("Time", width=8, justify="right")

    for t in snap.tests:
        style, label = _STATUS_STYLE.get(t.status, ("[white]", "?"))
        status_text = Text(label, style=style)

        name = t.test_name
        if len(name) > 45:
            name = name[:42] + "..."

        run_str = f"{t.current_run}/{t.total_runs}"
        time_str = _format_duration(t.duration)

        table.add_row(status_text, name, run_str, time_str)

        if t.error and t.status != "passed":
            err = t.error.split("\n")[0][:70]
            table.add_row("", Text(err, style="red"), "", "")

    # Summary footer
    pct = f"{snap.completed_tests}/{snap.total_tests}"
    summary_parts = [
        f"Progress: {pct}",
        f"Passed: {snap.passed_tests}",
        f"Failed: {snap.failed_tests}",
        f"Elapsed: {_format_duration(snap.elapsed)}",
    ]
    if snap.total_tokens > 0:
        summary_parts.append(f"Tokens: {snap.total_tokens:,}")
    if snap.total_cost > 0:
        summary_parts.append(f"Cost: ${snap.total_cost:.4f}")

    table.add_section()
    table.add_row(
        "",
        Text("  ".join(summary_parts), style="dim"),
        "",
        "",
    )

    return table


def is_terminal_capable() -> bool:
    """Check if the terminal supports Rich live display.

    Returns:
        True if the terminal is interactive and supports
        Rich rendering.
    """
    try:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        return False


class LiveProgressDisplay:
    """Rich Live display that renders streaming test progress.

    Wraps a Rich Live context and a LiveDisplayState, providing
    a progress callback compatible with TestOrchestrator.
    """

    def __init__(
        self,
        console: Console | None = None,
        refresh_per_second: int = 4,
    ) -> None:
        """Initialize the live progress display.

        Args:
            console: Optional Rich Console instance.
            refresh_per_second: Display refresh rate.
        """
        self._console = console or Console()
        self._refresh_per_second = refresh_per_second
        self._state = LiveDisplayState()
        self._live: Live | None = None

    @property
    def state(self) -> LiveDisplayState:
        """Return the underlying display state."""
        return self._state

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            build_table(self._state.snapshot()),
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def on_progress(self, event: ProgressEvent) -> None:
        """Handle a progress event and refresh the display.

        Designed to be used as a progress callback for the
        TestOrchestrator.

        Args:
            event: Progress event from the orchestrator.
        """
        self._state.update(event)
        if self._live is not None:
            self._live.update(build_table(self._state.snapshot()))

    def __enter__(self) -> LiveProgressDisplay:
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context manager."""
        self.stop()


class SimpleFallbackDisplay:
    """Simple line-based fallback for non-interactive terminals.

    Prints one line per event, no cursor movement or ANSI escapes.
    """

    def __init__(self) -> None:
        """Initialize the fallback display."""
        self._completed: int = 0
        self._total: int = 0

    def on_progress(self, event: ProgressEvent) -> None:
        """Handle a progress event with simple line output.

        Args:
            event: Progress event from the orchestrator.
        """
        etype = event.event_type

        if etype == ProgressEventType.SUITE_STARTED:
            self._total = event.total_tests or 0
            agent = event.details.get("agent_name", "?")
            print(f"Suite: {event.suite_name}  Agent: {agent}  Tests: {self._total}")

        elif etype == ProgressEventType.TEST_STARTED:
            name = event.test_name or event.test_id or "?"
            print(f"  -> {name} ...")

        elif etype in (
            ProgressEventType.TEST_COMPLETED,
            ProgressEventType.TEST_FAILED,
            ProgressEventType.TEST_TIMEOUT,
        ):
            self._completed += 1
            name = event.test_name or event.test_id or "?"
            if etype == ProgressEventType.TEST_TIMEOUT:
                tag = "TIMEOUT"
            elif event.success:
                tag = "PASS"
            else:
                tag = "FAIL"
            pct = f"[{self._completed}/{self._total}]"
            print(f"  {tag} {name} {pct}")

        elif etype == ProgressEventType.SUITE_COMPLETED:
            passed = event.details.get("passed_tests", 0)
            rate = event.details.get("success_rate", 0.0)
            dur = event.details.get("duration_seconds", 0.0)
            print(
                f"Done: {passed}/{self._total} passed "
                f"({rate * 100:.1f}%) "
                f"in {_format_duration(dur)}"
            )


def create_live_progress_callback(
    console: Console | None = None,
) -> tuple[
    LiveProgressDisplay | SimpleFallbackDisplay,
    Callable[[ProgressEvent], None],
]:
    """Create a live progress display or fallback.

    Returns a (display, callback) pair. The caller should use
    ``display`` as a context manager (if it is a
    ``LiveProgressDisplay``) and pass ``callback`` to the
    orchestrator.

    Args:
        console: Optional Rich Console.

    Returns:
        Tuple of (display_instance, progress_callback).
    """
    if is_terminal_capable():
        display = LiveProgressDisplay(console=console)
        return display, display.on_progress  # type: ignore[return-value]
    fallback = SimpleFallbackDisplay()
    return fallback, fallback.on_progress  # type: ignore[return-value]
