"""Unit tests for event buffer."""

from datetime import datetime, timedelta

import pytest

from atp.protocol import ATPEvent, ATPResponse, EventType, ResponseStatus
from atp.streaming.buffer import (
    BufferingEventIterator,
    EventBuffer,
    EventReplayIterator,
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


class TestEventBuffer:
    """Tests for EventBuffer."""

    def test_add_and_retrieve_events(self) -> None:
        """Test adding and retrieving events."""
        buffer = EventBuffer()

        events = [make_event(sequence=i) for i in range(3)]
        for event in events:
            buffer.add(event)

        assert buffer.event_count == 3
        assert len(buffer.events) == 3

    def test_events_returns_copy(self) -> None:
        """Test that events property returns a copy."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))

        events1 = buffer.events
        events2 = buffer.events

        assert events1 is not events2
        assert events1 == events2

    def test_max_size_limit(self) -> None:
        """Test that max_size limits buffer."""
        buffer = EventBuffer(max_size=3)

        for i in range(5):
            buffer.add(make_event(sequence=i))

        assert buffer.event_count == 3
        # Should have the last 3 events
        sequences = [e.sequence for e in buffer.events]
        assert sequences == [2, 3, 4]

    def test_set_and_get_response(self) -> None:
        """Test setting and getting response."""
        buffer = EventBuffer()

        assert buffer.response is None

        response = ATPResponse(
            task_id="task-001",
            status=ResponseStatus.COMPLETED,
        )
        buffer.set_response(response)

        assert buffer.response is response

    def test_clear(self) -> None:
        """Test clearing buffer."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))
        buffer.set_response(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        buffer.clear()

        assert buffer.event_count == 0
        assert buffer.response is None

    def test_get_by_sequence_found(self) -> None:
        """Test getting event by sequence when exists."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))
        buffer.add(make_event(sequence=1))
        buffer.add(make_event(sequence=2))

        event = buffer.get_by_sequence(1)

        assert event is not None
        assert event.sequence == 1

    def test_get_by_sequence_not_found(self) -> None:
        """Test getting event by sequence when not exists."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))

        event = buffer.get_by_sequence(99)

        assert event is None

    def test_get_by_type(self) -> None:
        """Test filtering events by type."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0, event_type=EventType.PROGRESS))
        buffer.add(make_event(sequence=1, event_type=EventType.TOOL_CALL))
        buffer.add(make_event(sequence=2, event_type=EventType.PROGRESS))

        progress_events = buffer.get_by_type(EventType.PROGRESS)

        assert len(progress_events) == 2
        assert all(e.event_type == EventType.PROGRESS for e in progress_events)

    def test_get_by_type_empty(self) -> None:
        """Test filtering by type with no matches."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0, event_type=EventType.PROGRESS))

        error_events = buffer.get_by_type(EventType.ERROR)

        assert error_events == []

    def test_get_in_range(self) -> None:
        """Test getting events in sequence range."""
        buffer = EventBuffer()
        for i in range(10):
            buffer.add(make_event(sequence=i))

        events = buffer.get_in_range(start_sequence=3, end_sequence=6)

        assert len(events) == 4
        sequences = [e.sequence for e in events]
        assert sequences == [3, 4, 5, 6]

    def test_get_in_range_no_start(self) -> None:
        """Test range with no start bound."""
        buffer = EventBuffer()
        for i in range(10):
            buffer.add(make_event(sequence=i))

        events = buffer.get_in_range(end_sequence=2)

        assert len(events) == 3
        sequences = [e.sequence for e in events]
        assert sequences == [0, 1, 2]

    def test_get_in_range_no_end(self) -> None:
        """Test range with no end bound."""
        buffer = EventBuffer()
        for i in range(10):
            buffer.add(make_event(sequence=i))

        events = buffer.get_in_range(start_sequence=7)

        assert len(events) == 3
        sequences = [e.sequence for e in events]
        assert sequences == [7, 8, 9]

    def test_get_by_time_range(self) -> None:
        """Test filtering by time range."""
        buffer = EventBuffer()
        base_time = datetime.now()

        for i in range(5):
            buffer.add(
                make_event(
                    sequence=i,
                    timestamp=base_time + timedelta(minutes=i),
                )
            )

        events = buffer.get_by_time_range(
            start_time=base_time + timedelta(minutes=1),
            end_time=base_time + timedelta(minutes=3),
        )

        assert len(events) == 3
        sequences = [e.sequence for e in events]
        assert sequences == [1, 2, 3]

    def test_replay_returns_sorted_events(self) -> None:
        """Test that replay returns events in sequence order."""
        buffer = EventBuffer()
        # Add out of order
        buffer.add(make_event(sequence=2))
        buffer.add(make_event(sequence=0))
        buffer.add(make_event(sequence=1))

        replayed = list(buffer.replay())

        sequences = [e.sequence for e in replayed]
        assert sequences == [0, 1, 2]

    def test_replay_with_response(self) -> None:
        """Test replay includes response at end."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))
        buffer.add(make_event(sequence=1))
        buffer.set_response(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        items = list(buffer.replay_with_response())

        assert len(items) == 3
        assert isinstance(items[0], ATPEvent)
        assert isinstance(items[1], ATPEvent)
        assert isinstance(items[2], ATPResponse)

    def test_replay_with_response_no_response(self) -> None:
        """Test replay without response."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))

        items = list(buffer.replay_with_response())

        assert len(items) == 1
        assert isinstance(items[0], ATPEvent)

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))
        buffer.set_response(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        data = buffer.to_dict()

        assert "events" in data
        assert "response" in data
        assert len(data["events"]) == 1
        assert data["response"]["status"] == "completed"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        original = EventBuffer()
        original.add(make_event(sequence=0))
        original.add(make_event(sequence=1))
        original.set_response(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        data = original.to_dict()
        restored = EventBuffer.from_dict(data)

        assert restored.event_count == 2
        assert restored.response is not None
        assert restored.response.status == ResponseStatus.COMPLETED

    def test_from_dict_empty(self) -> None:
        """Test deserialization from empty dict."""
        restored = EventBuffer.from_dict({})

        assert restored.event_count == 0
        assert restored.response is None


class TestEventReplayIterator:
    """Tests for EventReplayIterator."""

    @pytest.mark.anyio
    async def test_replay_events(self) -> None:
        """Test replaying events from buffer."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))
        buffer.add(make_event(sequence=1))
        buffer.set_response(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        iterator = EventReplayIterator(buffer)
        items = [item async for item in iterator]

        assert len(items) == 3
        assert isinstance(items[0], ATPEvent)
        assert isinstance(items[1], ATPEvent)
        assert isinstance(items[2], ATPResponse)

    @pytest.mark.anyio
    async def test_replay_without_response(self) -> None:
        """Test replaying without including response."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))
        buffer.set_response(
            ATPResponse(task_id="task-001", status=ResponseStatus.COMPLETED)
        )

        iterator = EventReplayIterator(buffer, include_response=False)
        items = [item async for item in iterator]

        assert len(items) == 1
        assert isinstance(items[0], ATPEvent)

    @pytest.mark.anyio
    async def test_replay_empty_buffer(self) -> None:
        """Test replaying from empty buffer."""
        buffer = EventBuffer()

        iterator = EventReplayIterator(buffer)
        items = [item async for item in iterator]

        assert items == []

    @pytest.mark.anyio
    async def test_replay_multiple_times(self) -> None:
        """Test that replay can be done multiple times."""
        buffer = EventBuffer()
        buffer.add(make_event(sequence=0))

        iterator = EventReplayIterator(buffer, include_response=False)

        # First iteration
        items1 = [item async for item in iterator]
        # Second iteration (re-iterate)
        items2 = [item async for item in iterator]

        assert len(items1) == 1
        assert len(items2) == 1


class TestBufferingEventIterator:
    """Tests for BufferingEventIterator."""

    @pytest.mark.anyio
    async def test_buffers_events_while_iterating(self) -> None:
        """Test that events are buffered during iteration."""

        async def event_source():
            yield make_event(sequence=0)
            yield make_event(sequence=1)
            yield ATPResponse(
                task_id="task-001",
                status=ResponseStatus.COMPLETED,
            )

        iterator = BufferingEventIterator(event_source())
        items = [item async for item in iterator]

        assert len(items) == 3
        assert iterator.buffer.event_count == 2
        assert iterator.buffer.response is not None

    @pytest.mark.anyio
    async def test_uses_provided_buffer(self) -> None:
        """Test that iterator uses provided buffer."""
        buffer = EventBuffer()

        async def event_source():
            yield make_event(sequence=0)

        iterator = BufferingEventIterator(event_source(), buffer=buffer)
        [item async for item in iterator]

        assert buffer.event_count == 1

    @pytest.mark.anyio
    async def test_buffer_accessible_during_iteration(self) -> None:
        """Test that buffer can be accessed during iteration."""

        async def event_source():
            yield make_event(sequence=0)
            yield make_event(sequence=1)
            yield make_event(sequence=2)

        iterator = BufferingEventIterator(event_source())
        counts = []

        async for _ in iterator:
            counts.append(iterator.buffer.event_count)

        assert counts == [1, 2, 3]
