"""Tests for CLI live display module."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from atp.cli.live_display import (
    LiveDisplayState,
    LiveProgressDisplay,
    SimpleFallbackDisplay,
    build_table,
    is_terminal_capable,
)
from atp.runner.models import ProgressEvent, ProgressEventType


def _make_event(
    event_type: ProgressEventType,
    *,
    suite_name: str | None = None,
    test_id: str | None = None,
    test_name: str | None = None,
    run_number: int | None = None,
    total_runs: int | None = None,
    total_tests: int | None = None,
    success: bool | None = None,
    error: str | None = None,
    details: dict | None = None,
) -> ProgressEvent:
    """Helper to create progress events for tests."""
    return ProgressEvent(
        event_type=event_type,
        suite_name=suite_name,
        test_id=test_id,
        test_name=test_name,
        run_number=run_number,
        total_runs=total_runs,
        total_tests=total_tests,
        success=success,
        error=error,
        details=details or {},
    )


class TestLiveDisplayState:
    """Tests for LiveDisplayState."""

    def test_initial_state(self) -> None:
        """Test that initial state is empty."""
        state = LiveDisplayState()
        assert state.suite_name is None
        assert state.total_tests == 0
        assert state.completed_tests == 0
        assert state.passed_tests == 0
        assert state.failed_tests == 0
        assert state.finished is False
        assert len(state.tests) == 0

    def test_suite_started(self) -> None:
        """Test suite started event updates state."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="my-suite",
                total_tests=5,
                details={"agent_name": "test-agent"},
            )
        )
        assert state.suite_name == "my-suite"
        assert state.agent_name == "test-agent"
        assert state.total_tests == 5

    def test_test_started(self) -> None:
        """Test test started event adds a test entry."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
                total_runs=3,
            )
        )
        assert "t1" in state.tests
        assert state.tests["t1"].test_name == "Test One"
        assert state.tests["t1"].total_runs == 3
        assert state.tests["t1"].status == "running"
        assert state.test_order == ["t1"]

    def test_test_completed_passed(self) -> None:
        """Test test completed event marks pass."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_COMPLETED,
                test_id="t1",
                success=True,
            )
        )
        assert state.tests["t1"].status == "passed"
        assert state.completed_tests == 1
        assert state.passed_tests == 1
        assert state.failed_tests == 0

    def test_test_completed_failed(self) -> None:
        """Test test failed event marks failure."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_FAILED,
                test_id="t1",
                success=False,
                error="Something broke",
            )
        )
        assert state.tests["t1"].status == "failed"
        assert state.tests["t1"].error == "Something broke"
        assert state.completed_tests == 1
        assert state.passed_tests == 0
        assert state.failed_tests == 1

    def test_test_timeout(self) -> None:
        """Test timeout event marks timeout status."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_TIMEOUT,
                test_id="t1",
                success=False,
            )
        )
        assert state.tests["t1"].status == "timeout"
        assert state.failed_tests == 1

    def test_run_started(self) -> None:
        """Test run started updates current run."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
                total_runs=3,
            )
        )
        state.update(
            _make_event(
                ProgressEventType.RUN_STARTED,
                test_id="t1",
                run_number=2,
            )
        )
        assert state.tests["t1"].current_run == 2

    def test_run_completed_accumulates_tokens(self) -> None:
        """Test run completed accumulates token counts."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
            )
        )
        state.update(
            _make_event(
                ProgressEventType.RUN_COMPLETED,
                test_id="t1",
                success=True,
                details={"tokens": 100, "cost_usd": 0.01},
            )
        )
        assert state.total_tokens == 100
        assert state.total_cost == pytest.approx(0.01)

    def test_suite_completed(self) -> None:
        """Test suite completed sets finished flag."""
        state = LiveDisplayState()
        state.update(_make_event(ProgressEventType.SUITE_COMPLETED))
        assert state.finished is True

    def test_snapshot_returns_consistent_data(self) -> None:
        """Test snapshot captures the state correctly."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="s",
                total_tests=2,
                details={"agent_name": "a"},
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test 1",
                total_runs=1,
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_COMPLETED,
                test_id="t1",
                success=True,
            )
        )
        snap = state.snapshot()
        assert snap.suite_name == "s"
        assert snap.agent_name == "a"
        assert snap.total_tests == 2
        assert snap.completed_tests == 1
        assert snap.passed_tests == 1
        assert len(snap.tests) == 1
        assert snap.tests[0].test_name == "Test 1"
        assert snap.tests[0].status == "passed"

    def test_ignores_unknown_test_id(self) -> None:
        """Test that events for unknown test ids are ignored."""
        state = LiveDisplayState()
        # No crash on events for unknown test ids
        state.update(
            _make_event(
                ProgressEventType.TEST_COMPLETED,
                test_id="unknown",
                success=True,
            )
        )
        assert state.completed_tests == 0

    def test_ignores_none_test_id(self) -> None:
        """Test that events with None test_id are ignored."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id=None,
            )
        )
        assert len(state.tests) == 0


class TestBuildTable:
    """Tests for build_table function."""

    def test_empty_snapshot(self) -> None:
        """Test building a table from empty state."""
        state = LiveDisplayState()
        snap = state.snapshot()
        table = build_table(snap)
        assert table is not None
        assert table.row_count >= 1  # At least summary row

    def test_table_with_tests(self) -> None:
        """Test building table with test data."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="my-suite",
                total_tests=2,
                details={"agent_name": "bot"},
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test One",
                total_runs=1,
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_COMPLETED,
                test_id="t1",
                success=True,
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t2",
                test_name="Test Two",
                total_runs=1,
            )
        )
        snap = state.snapshot()
        table = build_table(snap)
        # 2 test rows + 1 summary row
        assert table.row_count >= 3

    def test_table_with_error(self) -> None:
        """Test table shows error for failed test."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="s",
                total_tests=1,
                details={"agent_name": "a"},
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test",
                total_runs=1,
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_FAILED,
                test_id="t1",
                success=False,
                error="Assertion failed",
            )
        )
        snap = state.snapshot()
        table = build_table(snap)
        # 1 test row + 1 error row + 1 summary row
        assert table.row_count >= 3

    def test_table_shows_token_count(self) -> None:
        """Test table includes token count when > 0."""
        state = LiveDisplayState()
        state.update(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="s",
                total_tests=1,
                details={"agent_name": "a"},
            )
        )
        state.update(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="Test",
            )
        )
        state.update(
            _make_event(
                ProgressEventType.RUN_COMPLETED,
                test_id="t1",
                success=True,
                details={"tokens": 500, "cost_usd": 0.05},
            )
        )
        snap = state.snapshot()
        table = build_table(snap)
        # Render the table to verify content
        console = Console(file=StringIO(), force_terminal=True)
        console.print(table)
        output = console.file.getvalue()
        assert "500" in output
        assert "$0.05" in output


class TestLiveProgressDisplay:
    """Tests for LiveProgressDisplay."""

    def test_start_stop(self) -> None:
        """Test start/stop lifecycle."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveProgressDisplay(console=console, refresh_per_second=1)
        display.start()
        assert display._live is not None
        display.stop()
        assert display._live is None

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        console = Console(file=StringIO(), force_terminal=True)
        with LiveProgressDisplay(console=console, refresh_per_second=1) as display:
            assert display._live is not None
        assert display._live is None

    def test_on_progress_updates_state(self) -> None:
        """Test that on_progress feeds the display state."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveProgressDisplay(console=console, refresh_per_second=1)
        display.start()
        try:
            display.on_progress(
                _make_event(
                    ProgressEventType.SUITE_STARTED,
                    suite_name="test-suite",
                    total_tests=3,
                    details={"agent_name": "agent"},
                )
            )
            assert display.state.suite_name == "test-suite"
            assert display.state.total_tests == 3
        finally:
            display.stop()

    def test_on_progress_without_live(self) -> None:
        """Test on_progress works before start (no crash)."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveProgressDisplay(console=console)
        # Should not crash even when _live is None
        display.on_progress(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="s",
                total_tests=1,
                details={"agent_name": "a"},
            )
        )
        assert display.state.suite_name == "s"


class TestSimpleFallbackDisplay:
    """Tests for SimpleFallbackDisplay."""

    def test_suite_started(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test suite started prints header."""
        fb = SimpleFallbackDisplay()
        fb.on_progress(
            _make_event(
                ProgressEventType.SUITE_STARTED,
                suite_name="my-suite",
                total_tests=5,
                details={"agent_name": "bot"},
            )
        )
        out = capsys.readouterr().out
        assert "my-suite" in out
        assert "bot" in out
        assert "5" in out

    def test_test_started(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test test started prints arrow."""
        fb = SimpleFallbackDisplay()
        fb._total = 2
        fb.on_progress(
            _make_event(
                ProgressEventType.TEST_STARTED,
                test_id="t1",
                test_name="First Test",
            )
        )
        out = capsys.readouterr().out
        assert "First Test" in out

    def test_test_passed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test test completed with success prints PASS."""
        fb = SimpleFallbackDisplay()
        fb._total = 2
        fb.on_progress(
            _make_event(
                ProgressEventType.TEST_COMPLETED,
                test_id="t1",
                test_name="Test",
                success=True,
            )
        )
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "[1/2]" in out

    def test_test_failed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test test failed prints FAIL."""
        fb = SimpleFallbackDisplay()
        fb._total = 1
        fb.on_progress(
            _make_event(
                ProgressEventType.TEST_FAILED,
                test_id="t1",
                test_name="Bad Test",
                success=False,
            )
        )
        out = capsys.readouterr().out
        assert "FAIL" in out

    def test_test_timeout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test timeout prints TIMEOUT."""
        fb = SimpleFallbackDisplay()
        fb._total = 1
        fb.on_progress(
            _make_event(
                ProgressEventType.TEST_TIMEOUT,
                test_id="t1",
                test_name="Slow Test",
                success=False,
            )
        )
        out = capsys.readouterr().out
        assert "TIMEOUT" in out

    def test_suite_completed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test suite completed prints summary."""
        fb = SimpleFallbackDisplay()
        fb._total = 3
        fb.on_progress(
            _make_event(
                ProgressEventType.SUITE_COMPLETED,
                details={
                    "passed_tests": 2,
                    "success_rate": 0.667,
                    "duration_seconds": 12.5,
                },
            )
        )
        out = capsys.readouterr().out
        assert "Done" in out
        assert "2/3" in out
        assert "66.7%" in out


class TestIsTerminalCapable:
    """Tests for is_terminal_capable."""

    def test_non_tty(self) -> None:
        """Test returns False for non-tty stdout."""
        with patch("sys.stdout", new=StringIO()):
            assert is_terminal_capable() is False

    def test_tty(self) -> None:
        """Test returns True for tty stdout."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        with patch("sys.stdout", new=mock_stdout):
            assert is_terminal_capable() is True


class TestCLILiveFlagIntegration:
    """Test --live flag is recognized by the CLI commands."""

    def test_test_help_shows_live(self) -> None:
        """Test that --live appears in test command help."""
        from click.testing import CliRunner

        from atp.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["test", "--help"])
        assert result.exit_code == 0
        assert "--live" in result.output

    def test_run_help_shows_live(self) -> None:
        """Test that --live appears in run command help."""
        from click.testing import CliRunner

        from atp.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--live" in result.output
