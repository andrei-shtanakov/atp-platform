"""Tests for tracing data models."""

from datetime import UTC, datetime, timedelta

from atp.protocol.models import EventType
from atp.tracing.models import (
    Trace,
    TraceMetadata,
    TraceStep,
    TraceSummary,
)


class TestTraceStep:
    """Tests for TraceStep model."""

    def test_create_step(self) -> None:
        now = datetime.now(UTC)
        step = TraceStep(
            sequence=0,
            timestamp=now,
            event_type=EventType.TOOL_CALL,
            task_id="task-1",
            payload={"tool": "search"},
        )
        assert step.sequence == 0
        assert step.event_type == EventType.TOOL_CALL
        assert step.task_id == "task-1"
        assert step.payload == {"tool": "search"}
        assert step.duration_ms is None

    def test_step_with_duration(self) -> None:
        step = TraceStep(
            sequence=1,
            timestamp=datetime.now(UTC),
            event_type=EventType.LLM_REQUEST,
            task_id="task-1",
            payload={"model": "gpt-4"},
            duration_ms=150.5,
        )
        assert step.duration_ms == 150.5


class TestTrace:
    """Tests for Trace model."""

    def test_create_trace_defaults(self) -> None:
        trace = Trace(test_id="test-1")
        assert trace.trace_id  # UUID generated
        assert trace.test_id == "test-1"
        assert trace.test_name == ""
        assert trace.status == "recording"
        assert trace.steps == []
        assert trace.total_events == 0
        assert trace.error is None
        assert trace.completed_at is None

    def test_duration_seconds(self) -> None:
        start = datetime.now(UTC)
        end = start + timedelta(seconds=5.5)
        trace = Trace(
            test_id="test-1",
            started_at=start,
            completed_at=end,
        )
        assert trace.duration_seconds is not None
        assert abs(trace.duration_seconds - 5.5) < 0.01

    def test_duration_none_when_not_completed(self) -> None:
        trace = Trace(test_id="test-1")
        assert trace.duration_seconds is None

    def test_event_type_counts(self) -> None:
        now = datetime.now(UTC)
        trace = Trace(
            test_id="test-1",
            steps=[
                TraceStep(
                    sequence=0,
                    timestamp=now,
                    event_type=EventType.TOOL_CALL,
                    task_id="t1",
                    payload={},
                ),
                TraceStep(
                    sequence=1,
                    timestamp=now,
                    event_type=EventType.TOOL_CALL,
                    task_id="t1",
                    payload={},
                ),
                TraceStep(
                    sequence=2,
                    timestamp=now,
                    event_type=EventType.LLM_REQUEST,
                    task_id="t1",
                    payload={},
                ),
            ],
        )
        counts = trace.event_type_counts
        assert counts == {"tool_call": 2, "llm_request": 1}

    def test_event_type_counts_empty(self) -> None:
        trace = Trace(test_id="test-1")
        assert trace.event_type_counts == {}


class TestTraceMetadata:
    """Tests for TraceMetadata model."""

    def test_defaults(self) -> None:
        meta = TraceMetadata()
        assert meta.agent_name is None
        assert meta.adapter_type is None
        assert meta.suite_name is None
        assert meta.tags == []
        assert meta.extra == {}

    def test_with_values(self) -> None:
        meta = TraceMetadata(
            agent_name="test-agent",
            adapter_type="http",
            suite_name="smoke",
            tags=["fast", "core"],
        )
        assert meta.agent_name == "test-agent"
        assert meta.tags == ["fast", "core"]


class TestTraceSummary:
    """Tests for TraceSummary model."""

    def test_from_trace(self) -> None:
        trace = Trace(
            test_id="test-1",
            test_name="My Test",
            status="completed",
            total_events=5,
            metadata=TraceMetadata(agent_name="agent-1"),
        )
        summary = TraceSummary.from_trace(trace)
        assert summary.trace_id == trace.trace_id
        assert summary.test_id == "test-1"
        assert summary.test_name == "My Test"
        assert summary.status == "completed"
        assert summary.total_events == 5
        assert summary.metadata.agent_name == "agent-1"
