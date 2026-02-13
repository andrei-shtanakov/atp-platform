"""Tests for TraceRecorder."""

from atp.protocol.models import (
    ATPResponse,
    EventType,
    ResponseStatus,
)
from atp.tracing.models import TraceMetadata
from atp.tracing.recorder import TraceRecorder


def _make_event(
    sequence: int = 0,
    event_type: EventType = EventType.TOOL_CALL,
    task_id: str = "task-1",
    payload: dict | None = None,
) -> "ATPEvent":  # noqa: F821
    from atp.protocol.models import ATPEvent

    return ATPEvent(
        task_id=task_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload or {},
    )


class TestTraceRecorder:
    """Tests for TraceRecorder."""

    def test_initial_state(self) -> None:
        recorder = TraceRecorder(test_id="test-1", test_name="My Test")
        assert recorder.is_recording
        assert recorder.trace.test_id == "test-1"
        assert recorder.trace.test_name == "My Test"
        assert recorder.trace.status == "recording"
        assert recorder.trace.total_events == 0

    def test_record_event(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        event = _make_event(sequence=0, payload={"tool": "search"})
        recorder.record_event(event)

        assert recorder.trace.total_events == 1
        assert len(recorder.trace.steps) == 1
        step = recorder.trace.steps[0]
        assert step.sequence == 0
        assert step.event_type == EventType.TOOL_CALL
        assert step.payload == {"tool": "search"}

    def test_record_multiple_events(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        for i in range(5):
            recorder.record_event(_make_event(sequence=i))

        assert recorder.trace.total_events == 5
        assert len(recorder.trace.steps) == 5

    def test_complete(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        recorder.record_event(_make_event())

        response = ATPResponse(
            task_id="task-1",
            status=ResponseStatus.COMPLETED,
        )
        trace = recorder.complete(response)

        assert not recorder.is_recording
        assert trace.status == "completed"
        assert trace.completed_at is not None
        assert trace.error is None

    def test_complete_with_error_response(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        response = ATPResponse(
            task_id="task-1",
            status=ResponseStatus.FAILED,
            error="something broke",
        )
        trace = recorder.complete(response)

        assert trace.status == "failed"
        assert trace.error == "something broke"

    def test_complete_without_response(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        trace = recorder.complete()

        assert trace.status == "completed"
        assert trace.completed_at is not None

    def test_fail(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        recorder.record_event(_make_event())

        trace = recorder.fail("timeout exceeded")

        assert not recorder.is_recording
        assert trace.status == "failed"
        assert trace.error == "timeout exceeded"
        assert trace.completed_at is not None

    def test_record_after_complete_ignored(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        recorder.complete()
        recorder.record_event(_make_event())

        assert recorder.trace.total_events == 0

    def test_metadata_passed_through(self) -> None:
        meta = TraceMetadata(agent_name="agent-x", suite_name="smoke")
        recorder = TraceRecorder(test_id="test-1", metadata=meta)

        assert recorder.trace.metadata.agent_name == "agent-x"
        assert recorder.trace.metadata.suite_name == "smoke"

    def test_trace_id_accessible(self) -> None:
        recorder = TraceRecorder(test_id="test-1")
        assert recorder.trace_id == recorder.trace.trace_id
        assert len(recorder.trace_id) > 0
