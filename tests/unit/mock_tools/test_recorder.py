"""Tests for call recorder."""

from atp.mock_tools.recorder import CallRecorder


class TestCallRecorder:
    """Tests for CallRecorder class."""

    def test_record_call(self) -> None:
        """Test recording a call."""
        recorder = CallRecorder()

        record = recorder.record(
            tool="test_tool",
            input_data={"query": "test"},
            output={"result": "value"},
            error=None,
            status="success",
            duration_ms=50.0,
            task_id="task-001",
        )

        assert record.tool == "test_tool"
        assert record.input == {"query": "test"}
        assert record.output == {"result": "value"}
        assert record.status == "success"
        assert record.duration_ms == 50.0
        assert record.task_id == "task-001"

    def test_get_records_all(self) -> None:
        """Test getting all records."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0)
        recorder.record("tool2", None, None, None, "success", 20.0)

        records = recorder.get_records()

        assert len(records) == 2
        assert records[0].tool == "tool1"
        assert records[1].tool == "tool2"

    def test_get_records_filter_by_tool(self) -> None:
        """Test filtering records by tool name."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0)
        recorder.record("tool2", None, None, None, "success", 20.0)
        recorder.record("tool1", None, None, None, "success", 30.0)

        records = recorder.get_records(tool="tool1")

        assert len(records) == 2
        assert all(r.tool == "tool1" for r in records)

    def test_get_records_filter_by_task_id(self) -> None:
        """Test filtering records by task ID."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0, "task-001")
        recorder.record("tool1", None, None, None, "success", 20.0, "task-002")
        recorder.record("tool1", None, None, None, "success", 30.0, "task-001")

        records = recorder.get_records(task_id="task-001")

        assert len(records) == 2
        assert all(r.task_id == "task-001" for r in records)

    def test_get_records_with_limit(self) -> None:
        """Test limiting number of records."""
        recorder = CallRecorder()

        for i in range(5):
            recorder.record(f"tool{i}", None, None, None, "success", 10.0)

        records = recorder.get_records(limit=3)

        assert len(records) == 3
        # Should return last 3 records
        assert records[0].tool == "tool2"
        assert records[1].tool == "tool3"
        assert records[2].tool == "tool4"

    def test_get_records_combined_filters(self) -> None:
        """Test combining multiple filters."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0, "task-001")
        recorder.record("tool1", None, None, None, "success", 20.0, "task-002")
        recorder.record("tool2", None, None, None, "success", 30.0, "task-001")

        records = recorder.get_records(tool="tool1", task_id="task-001")

        assert len(records) == 1
        assert records[0].tool == "tool1"
        assert records[0].task_id == "task-001"

    def test_get_call_count_all(self) -> None:
        """Test getting total call count."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0)
        recorder.record("tool2", None, None, None, "success", 20.0)

        assert recorder.get_call_count() == 2

    def test_get_call_count_by_tool(self) -> None:
        """Test getting call count for specific tool."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0)
        recorder.record("tool2", None, None, None, "success", 20.0)
        recorder.record("tool1", None, None, None, "success", 30.0)

        assert recorder.get_call_count(tool="tool1") == 2
        assert recorder.get_call_count(tool="tool2") == 1
        assert recorder.get_call_count(tool="nonexistent") == 0

    def test_clear(self) -> None:
        """Test clearing all records."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0)
        recorder.record("tool2", None, None, None, "success", 20.0)

        cleared = recorder.clear()

        assert cleared == 2
        assert len(recorder.get_records()) == 0

    def test_clear_empty(self) -> None:
        """Test clearing empty recorder."""
        recorder = CallRecorder()

        cleared = recorder.clear()

        assert cleared == 0

    def test_len(self) -> None:
        """Test __len__ method."""
        recorder = CallRecorder()

        assert len(recorder) == 0

        recorder.record("tool1", None, None, None, "success", 10.0)
        assert len(recorder) == 1

        recorder.record("tool2", None, None, None, "success", 20.0)
        assert len(recorder) == 2

    def test_iter(self) -> None:
        """Test iteration over records."""
        recorder = CallRecorder()

        recorder.record("tool1", None, None, None, "success", 10.0)
        recorder.record("tool2", None, None, None, "success", 20.0)

        tools = [r.tool for r in recorder]

        assert tools == ["tool1", "tool2"]

    def test_thread_safety(self) -> None:
        """Test recorder is thread-safe for basic operations."""
        import threading

        recorder = CallRecorder()
        errors: list[Exception] = []

        def record_calls() -> None:
            try:
                for i in range(100):
                    recorder.record(f"tool_{i}", None, None, None, "success", 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_calls) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert recorder.get_call_count() == 400
