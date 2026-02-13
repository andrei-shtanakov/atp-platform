"""Tests for trace storage backends."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atp.protocol.models import EventType
from atp.tracing.models import (
    Trace,
    TraceMetadata,
    TraceStep,
)
from atp.tracing.storage import FileTraceStorage


@pytest.fixture()
def storage(tmp_path: Path) -> FileTraceStorage:
    """Create a file-based storage with a temp directory."""
    return FileTraceStorage(base_dir=tmp_path)


def _make_trace(
    test_id: str = "test-1",
    test_name: str = "Test One",
    status: str = "completed",
    total_events: int = 2,
    agent_name: str | None = None,
) -> Trace:
    """Create a sample trace."""
    now = datetime.now(UTC)
    return Trace(
        test_id=test_id,
        test_name=test_name,
        status=status,
        total_events=total_events,
        started_at=now,
        completed_at=now,
        metadata=TraceMetadata(agent_name=agent_name),
        steps=[
            TraceStep(
                sequence=i,
                timestamp=now,
                event_type=EventType.TOOL_CALL,
                task_id="task-1",
                payload={"tool": f"tool-{i}"},
            )
            for i in range(total_events)
        ],
    )


class TestFileTraceStorage:
    """Tests for FileTraceStorage."""

    def test_save_and_load(self, storage: FileTraceStorage) -> None:
        trace = _make_trace()
        storage.save(trace)

        loaded = storage.load(trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == trace.trace_id
        assert loaded.test_id == "test-1"
        assert loaded.test_name == "Test One"
        assert loaded.status == "completed"
        assert loaded.total_events == 2
        assert len(loaded.steps) == 2

    def test_load_nonexistent(self, storage: FileTraceStorage) -> None:
        assert storage.load("nonexistent") is None

    def test_list_traces_empty(self, storage: FileTraceStorage) -> None:
        summaries = storage.list_traces()
        assert summaries == []

    def test_list_traces(self, storage: FileTraceStorage) -> None:
        t1 = _make_trace(test_id="test-1")
        t2 = _make_trace(test_id="test-2")
        storage.save(t1)
        storage.save(t2)

        summaries = storage.list_traces()
        assert len(summaries) == 2

    def test_list_traces_filter_test_id(self, storage: FileTraceStorage) -> None:
        t1 = _make_trace(test_id="test-1")
        t2 = _make_trace(test_id="test-2")
        storage.save(t1)
        storage.save(t2)

        summaries = storage.list_traces(test_id="test-1")
        assert len(summaries) == 1
        assert summaries[0].test_id == "test-1"

    def test_list_traces_filter_status(self, storage: FileTraceStorage) -> None:
        t1 = _make_trace(status="completed")
        t2 = _make_trace(status="failed")
        storage.save(t1)
        storage.save(t2)

        summaries = storage.list_traces(status="failed")
        assert len(summaries) == 1
        assert summaries[0].status == "failed"

    def test_list_traces_limit(self, storage: FileTraceStorage) -> None:
        for i in range(5):
            storage.save(_make_trace(test_id=f"test-{i}"))

        summaries = storage.list_traces(limit=3)
        assert len(summaries) == 3

    def test_delete(self, storage: FileTraceStorage) -> None:
        trace = _make_trace()
        storage.save(trace)
        assert storage.load(trace.trace_id) is not None

        deleted = storage.delete(trace.trace_id)
        assert deleted is True
        assert storage.load(trace.trace_id) is None

    def test_delete_nonexistent(self, storage: FileTraceStorage) -> None:
        assert storage.delete("nonexistent") is False

    def test_base_dir_property(self, tmp_path: Path) -> None:
        storage = FileTraceStorage(base_dir=tmp_path)
        assert storage.base_dir == tmp_path

    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "traces"
        FileTraceStorage(base_dir=new_dir)
        assert new_dir.exists()

    def test_handles_corrupt_file(self, storage: FileTraceStorage) -> None:
        """Corrupt files should be skipped during listing."""
        # Write a corrupt file
        corrupt = storage.base_dir / "corrupt.json"
        corrupt.write_text("not valid json{{{")

        summaries = storage.list_traces()
        assert len(summaries) == 0

    def test_overwrite_existing_trace(self, storage: FileTraceStorage) -> None:
        trace = _make_trace(test_name="Original")
        storage.save(trace)

        trace.test_name = "Updated"
        storage.save(trace)

        loaded = storage.load(trace.trace_id)
        assert loaded is not None
        assert loaded.test_name == "Updated"

    def test_json_file_content(self, storage: FileTraceStorage) -> None:
        trace = _make_trace()
        storage.save(trace)

        path = storage.base_dir / f"{trace.trace_id}.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["trace_id"] == trace.trace_id
        assert data["test_id"] == "test-1"
