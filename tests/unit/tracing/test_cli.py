"""Tests for trace CLI commands."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from atp.cli.commands.traces import (
    replay_command,
    traces_command,
)
from atp.protocol.models import EventType
from atp.tracing.models import Trace, TraceMetadata, TraceStep
from atp.tracing.storage import FileTraceStorage


@pytest.fixture()
def traces_dir(tmp_path: Path) -> Path:
    return tmp_path / "traces"


@pytest.fixture()
def storage(traces_dir: Path) -> FileTraceStorage:
    return FileTraceStorage(base_dir=traces_dir)


@pytest.fixture()
def sample_trace(storage: FileTraceStorage) -> Trace:
    now = datetime.now(UTC)
    trace = Trace(
        test_id="test-001",
        test_name="Sample Test",
        status="completed",
        started_at=now,
        completed_at=now,
        total_events=3,
        metadata=TraceMetadata(
            agent_name="test-agent",
            suite_name="smoke",
        ),
        steps=[
            TraceStep(
                sequence=i,
                timestamp=now,
                event_type=EventType.TOOL_CALL,
                task_id="task-1",
                payload={"tool": f"tool-{i}"},
            )
            for i in range(3)
        ],
    )
    storage.save(trace)
    return trace


class TestTracesListCommand:
    """Tests for 'atp traces list'."""

    def test_list_empty(self, traces_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            ["list", "--traces-dir", str(traces_dir)],
        )
        assert result.exit_code == 0
        assert "No traces found" in result.output

    def test_list_with_traces(
        self,
        traces_dir: Path,
        sample_trace: Trace,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            ["list", "--traces-dir", str(traces_dir)],
        )
        assert result.exit_code == 0
        assert "Sample Test" in result.output
        assert "completed" in result.output

    def test_list_filter_test_id(
        self,
        traces_dir: Path,
        sample_trace: Trace,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "list",
                "--traces-dir",
                str(traces_dir),
                "--test-id",
                "nonexistent",
            ],
        )
        assert result.exit_code == 0
        assert "No traces found" in result.output


class TestTracesShowCommand:
    """Tests for 'atp traces show'."""

    def test_show_trace(
        self,
        traces_dir: Path,
        sample_trace: Trace,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "show",
                sample_trace.trace_id,
                "--traces-dir",
                str(traces_dir),
            ],
        )
        assert result.exit_code == 0
        assert sample_trace.trace_id in result.output
        assert "Sample Test" in result.output
        assert "completed" in result.output

    def test_show_not_found(self, traces_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "show",
                "nonexistent",
                "--traces-dir",
                str(traces_dir),
            ],
        )
        assert result.exit_code == 1

    def test_show_json(
        self,
        traces_dir: Path,
        sample_trace: Trace,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "show",
                sample_trace.trace_id,
                "--traces-dir",
                str(traces_dir),
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0
        assert '"trace_id"' in result.output

    def test_show_prefix_match(
        self,
        traces_dir: Path,
        sample_trace: Trace,
    ) -> None:
        prefix = sample_trace.trace_id[:8]
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "show",
                prefix,
                "--traces-dir",
                str(traces_dir),
            ],
        )
        assert result.exit_code == 0
        assert sample_trace.trace_id in result.output


class TestTracesDeleteCommand:
    """Tests for 'atp traces delete'."""

    def test_delete_trace(
        self,
        traces_dir: Path,
        sample_trace: Trace,
        storage: FileTraceStorage,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "delete",
                sample_trace.trace_id,
                "--traces-dir",
                str(traces_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert storage.load(sample_trace.trace_id) is None

    def test_delete_not_found(self, traces_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            traces_command,
            [
                "delete",
                "nonexistent",
                "--traces-dir",
                str(traces_dir),
            ],
        )
        assert result.exit_code == 1


class TestReplayCommand:
    """Tests for 'atp replay'."""

    def test_replay_trace(
        self,
        traces_dir: Path,
        sample_trace: Trace,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            replay_command,
            [
                sample_trace.trace_id,
                "--traces-dir",
                str(traces_dir),
                "--speed",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "Replaying trace" in result.output
        assert "tool_call" in result.output
        assert "Replay complete" in result.output

    def test_replay_not_found(self, traces_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            replay_command,
            [
                "nonexistent",
                "--traces-dir",
                str(traces_dir),
            ],
        )
        assert result.exit_code == 1
