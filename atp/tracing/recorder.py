"""Trace recorder for capturing ATP events during test runs."""

import logging
from datetime import UTC, datetime

from atp.protocol.models import ATPEvent, ATPResponse
from atp.tracing.models import Trace, TraceMetadata, TraceStep

logger = logging.getLogger(__name__)


class TraceRecorder:
    """Records ATP events into a Trace object during test execution.

    Usage::

        recorder = TraceRecorder(test_id="test-1", test_name="My Test")
        recorder.record_event(event)
        recorder.complete(response)
        trace = recorder.trace
    """

    def __init__(
        self,
        test_id: str,
        test_name: str = "",
        metadata: TraceMetadata | None = None,
    ) -> None:
        """Initialize the trace recorder.

        Args:
            test_id: Identifier of the test being traced.
            test_name: Human-readable test name.
            metadata: Optional trace metadata.
        """
        self._trace = Trace(
            test_id=test_id,
            test_name=test_name,
            metadata=metadata or TraceMetadata(),
        )
        self._recording = True

    @property
    def trace(self) -> Trace:
        """Return the trace being recorded."""
        return self._trace

    @property
    def trace_id(self) -> str:
        """Return the trace ID."""
        return self._trace.trace_id

    @property
    def is_recording(self) -> bool:
        """Check if recorder is still active."""
        return self._recording

    def record_event(self, event: ATPEvent) -> None:
        """Record a single ATP event as a trace step.

        Args:
            event: The ATP event to record.
        """
        if not self._recording:
            logger.warning(
                "Attempted to record event on completed trace %s",
                self._trace.trace_id,
            )
            return

        step = TraceStep(
            sequence=event.sequence,
            timestamp=event.timestamp,
            event_type=event.event_type,
            task_id=event.task_id,
            payload=event.payload,
        )
        self._trace.steps.append(step)
        self._trace.total_events += 1

    def complete(self, response: ATPResponse | None = None) -> Trace:
        """Mark the trace as completed.

        Args:
            response: Optional final response from the agent.

        Returns:
            The completed trace.
        """
        self._recording = False
        self._trace.completed_at = datetime.now(UTC)

        if response is not None:
            self._trace.status = response.status.value
            if response.error:
                self._trace.error = response.error
        else:
            self._trace.status = "completed"

        return self._trace

    def fail(self, error: str) -> Trace:
        """Mark the trace as failed.

        Args:
            error: Error description.

        Returns:
            The failed trace.
        """
        self._recording = False
        self._trace.completed_at = datetime.now(UTC)
        self._trace.status = "failed"
        self._trace.error = error
        return self._trace
