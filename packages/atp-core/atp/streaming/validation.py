"""Event ordering validation for ATP Protocol streaming."""

from collections.abc import AsyncIterator
from datetime import datetime

from atp.core.exceptions import ATPError
from atp.protocol import ATPEvent, ATPResponse


class EventOrderingError(ATPError):
    """Error raised when event ordering is violated."""

    def __init__(
        self,
        message: str,
        expected_sequence: int | None = None,
        actual_sequence: int | None = None,
        event: ATPEvent | None = None,
    ) -> None:
        """Initialize event ordering error.

        Args:
            message: Error message.
            expected_sequence: Expected sequence number.
            actual_sequence: Actual sequence number received.
            event: The event that violated ordering.
        """
        super().__init__(message)
        self.expected_sequence = expected_sequence
        self.actual_sequence = actual_sequence
        self.event = event


class EventValidator:
    """Validates event ordering and consistency.

    Ensures events arrive in proper sequence order and have valid timestamps.
    """

    def __init__(
        self,
        strict_sequence: bool = True,
        strict_timestamps: bool = False,
    ) -> None:
        """Initialize the event validator.

        Args:
            strict_sequence: Raise error on sequence gaps (default True).
            strict_timestamps: Raise error on timestamp violations (default False).
        """
        self.strict_sequence = strict_sequence
        self.strict_timestamps = strict_timestamps
        self._last_sequence: int | None = None
        self._last_timestamp: datetime | None = None
        self._task_id: str | None = None

    def reset(self) -> None:
        """Reset validator state for a new event stream."""
        self._last_sequence = None
        self._last_timestamp = None
        self._task_id = None

    def validate(self, event: ATPEvent) -> None:
        """Validate a single event.

        Args:
            event: The event to validate.

        Raises:
            EventOrderingError: If event ordering is violated.
        """
        # Validate task_id consistency
        if self._task_id is None:
            self._task_id = event.task_id
        elif event.task_id != self._task_id:
            raise EventOrderingError(
                f"Task ID mismatch: expected '{self._task_id}', got '{event.task_id}'",
                event=event,
            )

        # Validate sequence ordering
        if self._last_sequence is not None:
            expected = self._last_sequence + 1
            if event.sequence < self._last_sequence:
                raise EventOrderingError(
                    f"Out of order event: sequence {event.sequence} "
                    f"received after {self._last_sequence}",
                    expected_sequence=expected,
                    actual_sequence=event.sequence,
                    event=event,
                )
            if self.strict_sequence and event.sequence != expected:
                raise EventOrderingError(
                    f"Sequence gap detected: expected {expected}, got {event.sequence}",
                    expected_sequence=expected,
                    actual_sequence=event.sequence,
                    event=event,
                )

        # Validate timestamp ordering (optional)
        if self.strict_timestamps and self._last_timestamp is not None:
            if event.timestamp < self._last_timestamp:
                raise EventOrderingError(
                    f"Timestamp violation: event at {event.timestamp} "
                    f"is before previous event at {self._last_timestamp}",
                    event=event,
                )

        self._last_sequence = event.sequence
        self._last_timestamp = event.timestamp

    @property
    def event_count(self) -> int:
        """Return the number of validated events."""
        if self._last_sequence is None:
            return 0
        return self._last_sequence + 1


def validate_event_sequence(events: list[ATPEvent]) -> list[EventOrderingError]:
    """Validate a sequence of events and return all errors.

    Unlike EventValidator.validate(), this function collects all errors
    instead of raising on the first one.

    Args:
        events: List of events to validate.

    Returns:
        List of EventOrderingError for any ordering violations found.
    """
    errors: list[EventOrderingError] = []

    if not events:
        return errors

    # Check for sequence gaps and out-of-order events
    sorted_events = sorted(events, key=lambda e: e.sequence)
    task_ids = {e.task_id for e in events}

    # Check task_id consistency
    if len(task_ids) > 1:
        errors.append(
            EventOrderingError(
                f"Multiple task IDs found in event stream: {task_ids}",
            )
        )

    # Check sequence ordering
    for i, event in enumerate(sorted_events):
        expected_seq = i
        if event.sequence != expected_seq:
            errors.append(
                EventOrderingError(
                    f"Sequence gap: expected {expected_seq}, got {event.sequence}",
                    expected_sequence=expected_seq,
                    actual_sequence=event.sequence,
                    event=event,
                )
            )

    # Check for duplicates
    sequences = [e.sequence for e in events]
    seen: set[int] = set()
    for seq in sequences:
        if seq in seen:
            errors.append(
                EventOrderingError(
                    f"Duplicate sequence number: {seq}",
                    actual_sequence=seq,
                )
            )
        seen.add(seq)

    return errors


class ValidatingEventIterator:
    """AsyncIterator wrapper that validates events as they are yielded.

    Wraps another AsyncIterator and validates each event before yielding.
    """

    def __init__(
        self,
        source: AsyncIterator[ATPEvent | ATPResponse],
        validator: EventValidator | None = None,
        on_error: str = "raise",
    ) -> None:
        """Initialize the validating iterator.

        Args:
            source: Source iterator to wrap.
            validator: EventValidator instance (creates new one if None).
            on_error: What to do on validation error:
                - "raise": Raise EventOrderingError (default)
                - "skip": Skip the invalid event
                - "warn": Log warning and continue
        """
        self._source = source
        self._validator = validator or EventValidator()
        self._on_error = on_error

    def __aiter__(self) -> "ValidatingEventIterator":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> ATPEvent | ATPResponse:
        """Get next validated event or response.

        Returns:
            Next event or response from the source.

        Raises:
            StopAsyncIteration: When source is exhausted.
            EventOrderingError: If event validation fails and on_error="raise".
        """
        while True:
            item = await self._source.__anext__()

            # Don't validate the final response
            if isinstance(item, ATPResponse):
                return item

            try:
                self._validator.validate(item)
                return item
            except EventOrderingError:
                if self._on_error == "raise":
                    raise
                elif self._on_error == "skip":
                    continue
                else:  # warn
                    # Just continue, logging can be added if needed
                    return item
