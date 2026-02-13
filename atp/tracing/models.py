"""Data models for agent execution tracing."""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from atp.protocol.models import EventType


class TraceStep(BaseModel):
    """A single step in an execution trace."""

    sequence: int = Field(..., description="Step sequence number", ge=0)
    timestamp: datetime = Field(..., description="When the step occurred")
    event_type: EventType = Field(..., description="Type of event")
    task_id: str = Field(..., description="Task identifier")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Event payload data"
    )
    duration_ms: float | None = Field(
        None, description="Step duration in milliseconds", ge=0
    )


class TraceMetadata(BaseModel):
    """Metadata associated with a trace."""

    agent_name: str | None = Field(None, description="Name of the agent")
    adapter_type: str | None = Field(None, description="Adapter type used")
    suite_name: str | None = Field(None, description="Test suite name")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class Trace(BaseModel):
    """Complete execution trace for a test run."""

    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trace identifier",
    )
    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(default="", description="Human-readable test name")
    status: str = Field(
        default="recording",
        description="Trace status: recording, completed, failed",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the trace started",
    )
    completed_at: datetime | None = Field(None, description="When the trace completed")
    steps: list[TraceStep] = Field(
        default_factory=list, description="Ordered list of trace steps"
    )
    metadata: TraceMetadata = Field(
        default_factory=TraceMetadata,
        description="Trace metadata",
    )
    total_events: int = Field(default=0, description="Total number of events recorded")
    error: str | None = Field(None, description="Error message if trace failed")

    @property
    def duration_seconds(self) -> float | None:
        """Calculate trace duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def event_type_counts(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for step in self.steps:
            key = step.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class TraceSummary(BaseModel):
    """Lightweight summary of a trace for listing."""

    trace_id: str = Field(..., description="Unique trace identifier")
    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(default="", description="Test name")
    status: str = Field(..., description="Trace status")
    started_at: datetime = Field(..., description="Start time")
    completed_at: datetime | None = Field(None, description="End time")
    total_events: int = Field(default=0, description="Event count")
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)

    @classmethod
    def from_trace(cls, trace: Trace) -> "TraceSummary":
        """Create a summary from a full trace."""
        return cls(
            trace_id=trace.trace_id,
            test_id=trace.test_id,
            test_name=trace.test_name,
            status=trace.status,
            started_at=trace.started_at,
            completed_at=trace.completed_at,
            total_events=trace.total_events,
            metadata=trace.metadata,
        )
