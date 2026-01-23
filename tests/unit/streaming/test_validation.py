"""Unit tests for event validation."""

from datetime import datetime, timedelta

import pytest

from atp.protocol import ATPEvent, ATPResponse, EventType, ResponseStatus
from atp.streaming.validation import (
    EventOrderingError,
    EventValidator,
    ValidatingEventIterator,
    validate_event_sequence,
)


def make_event(
    sequence: int,
    task_id: str = "task-001",
    event_type: EventType = EventType.PROGRESS,
    timestamp: datetime | None = None,
) -> ATPEvent:
    """Create a test event with given parameters."""
    return ATPEvent(
        task_id=task_id,
        sequence=sequence,
        event_type=event_type,
        payload={"message": f"Event {sequence}"},
        timestamp=timestamp or datetime.now(),
    )


class TestEventValidator:
    """Tests for EventValidator."""

    def test_validate_sequential_events(self) -> None:
        """Test validation of correctly sequenced events."""
        validator = EventValidator()

        for i in range(5):
            event = make_event(sequence=i)
            validator.validate(event)

        assert validator.event_count == 5

    def test_validate_detects_out_of_order(self) -> None:
        """Test that out-of-order events raise error."""
        validator = EventValidator()

        validator.validate(make_event(sequence=0))
        validator.validate(make_event(sequence=1))
        validator.validate(make_event(sequence=2))

        with pytest.raises(EventOrderingError) as exc_info:
            validator.validate(make_event(sequence=1))

        assert exc_info.value.actual_sequence == 1
        assert "Out of order" in str(exc_info.value)

    def test_validate_detects_sequence_gap_strict(self) -> None:
        """Test that sequence gaps are detected in strict mode."""
        validator = EventValidator(strict_sequence=True)

        validator.validate(make_event(sequence=0))

        with pytest.raises(EventOrderingError) as exc_info:
            validator.validate(make_event(sequence=2))

        assert exc_info.value.expected_sequence == 1
        assert exc_info.value.actual_sequence == 2
        assert "Sequence gap" in str(exc_info.value)

    def test_validate_allows_sequence_gap_non_strict(self) -> None:
        """Test that sequence gaps are allowed in non-strict mode."""
        validator = EventValidator(strict_sequence=False)

        validator.validate(make_event(sequence=0))
        # This should not raise
        validator.validate(make_event(sequence=5))

        assert validator.event_count == 6

    def test_validate_task_id_mismatch(self) -> None:
        """Test that task ID mismatch raises error."""
        validator = EventValidator()

        validator.validate(make_event(sequence=0, task_id="task-001"))

        with pytest.raises(EventOrderingError) as exc_info:
            validator.validate(make_event(sequence=1, task_id="task-002"))

        assert "Task ID mismatch" in str(exc_info.value)

    def test_validate_timestamp_strict_mode(self) -> None:
        """Test timestamp validation in strict mode."""
        validator = EventValidator(strict_timestamps=True)

        now = datetime.now()
        validator.validate(make_event(sequence=0, timestamp=now))

        with pytest.raises(EventOrderingError) as exc_info:
            validator.validate(
                make_event(sequence=1, timestamp=now - timedelta(seconds=10))
            )

        assert "Timestamp violation" in str(exc_info.value)

    def test_validate_timestamp_non_strict_allows_any_order(self) -> None:
        """Test that timestamp order is ignored in non-strict mode."""
        validator = EventValidator(strict_timestamps=False)

        now = datetime.now()
        validator.validate(make_event(sequence=0, timestamp=now))
        # Should not raise even with earlier timestamp
        validator.validate(
            make_event(sequence=1, timestamp=now - timedelta(seconds=10))
        )

    def test_reset_clears_state(self) -> None:
        """Test that reset clears validator state."""
        validator = EventValidator()

        validator.validate(make_event(sequence=0, task_id="task-001"))
        validator.validate(make_event(sequence=1, task_id="task-001"))

        validator.reset()

        # Should be able to start fresh with different task_id
        validator.validate(make_event(sequence=0, task_id="task-002"))

    def test_event_count_empty(self) -> None:
        """Test event count with no events."""
        validator = EventValidator()
        assert validator.event_count == 0


class TestValidateEventSequence:
    """Tests for validate_event_sequence function."""

    def test_valid_sequence(self) -> None:
        """Test validation of correct sequence."""
        events = [make_event(sequence=i) for i in range(5)]
        errors = validate_event_sequence(events)
        assert errors == []

    def test_empty_sequence(self) -> None:
        """Test validation of empty sequence."""
        errors = validate_event_sequence([])
        assert errors == []

    def test_detects_gaps(self) -> None:
        """Test detection of sequence gaps."""
        events = [
            make_event(sequence=0),
            make_event(sequence=1),
            make_event(sequence=3),  # Gap: 2 is missing
        ]
        errors = validate_event_sequence(events)
        assert len(errors) == 1
        assert "Sequence gap" in str(errors[0])

    def test_detects_duplicates(self) -> None:
        """Test detection of duplicate sequences."""
        events = [
            make_event(sequence=0),
            make_event(sequence=1),
            make_event(sequence=1),  # Duplicate
        ]
        errors = validate_event_sequence(events)
        # Should have one error for duplicate and one for gap
        assert len(errors) >= 1
        has_duplicate_error = any("Duplicate" in str(e) for e in errors)
        assert has_duplicate_error

    def test_detects_multiple_task_ids(self) -> None:
        """Test detection of multiple task IDs."""
        events = [
            make_event(sequence=0, task_id="task-001"),
            make_event(sequence=1, task_id="task-002"),
        ]
        errors = validate_event_sequence(events)
        assert len(errors) >= 1
        # Check for task IDs error (message contains "task IDs" or "task_id")
        has_task_id_error = any(
            "task ids" in str(e).lower() or "task_ids" in str(e).lower() for e in errors
        )
        assert has_task_id_error


class TestValidatingEventIterator:
    """Tests for ValidatingEventIterator."""

    @pytest.mark.anyio
    async def test_validates_and_yields_events(self) -> None:
        """Test that iterator validates and yields events."""

        async def event_source():
            for i in range(3):
                yield make_event(sequence=i)
            yield ATPResponse(
                task_id="task-001",
                status=ResponseStatus.COMPLETED,
            )

        iterator = ValidatingEventIterator(event_source())
        results = [item async for item in iterator]

        assert len(results) == 4
        assert isinstance(results[-1], ATPResponse)

    @pytest.mark.anyio
    async def test_raises_on_invalid_sequence(self) -> None:
        """Test that iterator raises on invalid sequence."""

        async def bad_event_source():
            yield make_event(sequence=0)
            yield make_event(sequence=2)  # Gap

        iterator = ValidatingEventIterator(
            bad_event_source(),
            on_error="raise",
        )

        with pytest.raises(EventOrderingError):
            async for _ in iterator:
                pass

    @pytest.mark.anyio
    async def test_skips_invalid_events_on_skip_mode(self) -> None:
        """Test that iterator skips invalid events in skip mode."""

        async def bad_event_source():
            yield make_event(sequence=0)
            yield make_event(sequence=2)  # Gap - will be skipped
            yield make_event(sequence=3)
            yield ATPResponse(
                task_id="task-001",
                status=ResponseStatus.COMPLETED,
            )

        validator = EventValidator(strict_sequence=True)
        iterator = ValidatingEventIterator(
            bad_event_source(),
            validator=validator,
            on_error="skip",
        )

        results = [item async for item in iterator]

        # Should have first event and response, gap event skipped
        assert len(results) == 2
        assert isinstance(results[0], ATPEvent)
        assert results[0].sequence == 0
        assert isinstance(results[-1], ATPResponse)

    @pytest.mark.anyio
    async def test_warns_on_invalid_events_warn_mode(self) -> None:
        """Test that iterator continues on warn mode."""

        async def bad_event_source():
            yield make_event(sequence=0)
            yield make_event(sequence=2)  # Gap - will warn but continue
            yield ATPResponse(
                task_id="task-001",
                status=ResponseStatus.COMPLETED,
            )

        iterator = ValidatingEventIterator(
            bad_event_source(),
            on_error="warn",
        )

        results = [item async for item in iterator]

        # Should have all events
        assert len(results) == 3


class TestEventOrderingError:
    """Tests for EventOrderingError."""

    def test_error_with_all_fields(self) -> None:
        """Test creating error with all fields."""
        event = make_event(sequence=5)
        error = EventOrderingError(
            "Test error",
            expected_sequence=4,
            actual_sequence=5,
            event=event,
        )

        assert error.expected_sequence == 4
        assert error.actual_sequence == 5
        assert error.event is event
        assert "Test error" in str(error)

    def test_error_with_minimal_fields(self) -> None:
        """Test creating error with minimal fields."""
        error = EventOrderingError("Simple error")

        assert error.expected_sequence is None
        assert error.actual_sequence is None
        assert error.event is None
        assert "Simple error" in str(error)
