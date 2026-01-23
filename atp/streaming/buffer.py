"""Event buffering and replay for ATP Protocol streaming."""

from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any

from atp.protocol import ATPEvent, ATPResponse, EventType


class EventBuffer:
    """Buffer for storing and replaying ATP events.

    Stores events during streaming for later replay or analysis.
    Supports filtering by event type, time range, and sequence.
    """

    def __init__(self, max_size: int | None = None) -> None:
        """Initialize the event buffer.

        Args:
            max_size: Maximum number of events to store (None for unlimited).
        """
        self._events: list[ATPEvent] = []
        self._response: ATPResponse | None = None
        self._max_size = max_size

    def add(self, event: ATPEvent) -> None:
        """Add an event to the buffer.

        Args:
            event: Event to add.
        """
        if self._max_size is not None and len(self._events) >= self._max_size:
            # Remove oldest event when buffer is full
            self._events.pop(0)
        self._events.append(event)

    def set_response(self, response: ATPResponse) -> None:
        """Set the final response.

        Args:
            response: The final response from the agent.
        """
        self._response = response

    @property
    def events(self) -> list[ATPEvent]:
        """Return all buffered events."""
        return list(self._events)

    @property
    def response(self) -> ATPResponse | None:
        """Return the final response if available."""
        return self._response

    @property
    def event_count(self) -> int:
        """Return the number of buffered events."""
        return len(self._events)

    def clear(self) -> None:
        """Clear all buffered events and response."""
        self._events.clear()
        self._response = None

    def get_by_sequence(self, sequence: int) -> ATPEvent | None:
        """Get event by sequence number.

        Args:
            sequence: The sequence number to look up.

        Returns:
            Event with matching sequence or None if not found.
        """
        for event in self._events:
            if event.sequence == sequence:
                return event
        return None

    def get_by_type(self, event_type: EventType) -> list[ATPEvent]:
        """Get all events of a specific type.

        Args:
            event_type: The event type to filter by.

        Returns:
            List of events matching the type.
        """
        return [e for e in self._events if e.event_type == event_type]

    def get_in_range(
        self,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
    ) -> list[ATPEvent]:
        """Get events within a sequence range.

        Args:
            start_sequence: Start of range (inclusive), None for no lower bound.
            end_sequence: End of range (inclusive), None for no upper bound.

        Returns:
            List of events within the range.
        """
        result = []
        for event in self._events:
            if start_sequence is not None and event.sequence < start_sequence:
                continue
            if end_sequence is not None and event.sequence > end_sequence:
                continue
            result.append(event)
        return result

    def get_by_time_range(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ATPEvent]:
        """Get events within a time range.

        Args:
            start_time: Start of range (inclusive), None for no lower bound.
            end_time: End of range (inclusive), None for no upper bound.

        Returns:
            List of events within the time range.
        """
        result = []
        for event in self._events:
            if start_time is not None and event.timestamp < start_time:
                continue
            if end_time is not None and event.timestamp > end_time:
                continue
            result.append(event)
        return result

    def replay(self) -> Iterator[ATPEvent]:
        """Replay all buffered events in sequence order.

        Yields:
            Events in sequence order.
        """
        sorted_events = sorted(self._events, key=lambda e: e.sequence)
        yield from sorted_events

    def replay_with_response(self) -> Iterator[ATPEvent | ATPResponse]:
        """Replay all events followed by the response.

        Yields:
            Events in sequence order, then the response.
        """
        yield from self.replay()
        if self._response is not None:
            yield self._response

    def to_dict(self) -> dict[str, Any]:
        """Serialize buffer to dictionary.

        Returns:
            Dictionary representation of buffer.
        """
        return {
            "events": [e.model_dump(mode="json") for e in self._events],
            "response": (
                self._response.model_dump(mode="json")
                if self._response is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventBuffer":
        """Deserialize buffer from dictionary.

        Args:
            data: Dictionary representation of buffer.

        Returns:
            Reconstructed EventBuffer.
        """
        buffer = cls()
        for event_data in data.get("events", []):
            buffer.add(ATPEvent.model_validate(event_data))
        if data.get("response") is not None:
            buffer.set_response(ATPResponse.model_validate(data["response"]))
        return buffer


class EventReplayIterator:
    """AsyncIterator that replays events from a buffer.

    Provides an async interface for replaying buffered events.
    """

    def __init__(
        self,
        buffer: EventBuffer,
        include_response: bool = True,
        delay_ms: float = 0,
    ) -> None:
        """Initialize the replay iterator.

        Args:
            buffer: EventBuffer to replay from.
            include_response: Whether to yield the final response.
            delay_ms: Delay between yielded events in milliseconds.
        """
        self._buffer = buffer
        self._include_response = include_response
        self._delay_ms = delay_ms
        self._events_iter: Iterator[ATPEvent] | None = None
        self._response_yielded = False

    def __aiter__(self) -> "EventReplayIterator":
        """Return the async iterator."""
        self._events_iter = self._buffer.replay()
        self._response_yielded = False
        return self

    async def __anext__(self) -> ATPEvent | ATPResponse:
        """Get next event or response.

        Returns:
            Next event from buffer or final response.

        Raises:
            StopAsyncIteration: When all events and response have been yielded.
        """
        import asyncio

        if self._events_iter is None:
            self._events_iter = self._buffer.replay()

        # Try to get next event
        try:
            event = next(self._events_iter)
            if self._delay_ms > 0:
                await asyncio.sleep(self._delay_ms / 1000)
            return event
        except StopIteration:
            pass

        # Yield response if available and not yet yielded
        if (
            self._include_response
            and not self._response_yielded
            and self._buffer.response is not None
        ):
            self._response_yielded = True
            return self._buffer.response

        raise StopAsyncIteration


class BufferingEventIterator:
    """AsyncIterator wrapper that buffers events as they pass through.

    Wraps another AsyncIterator, buffering events while yielding them.
    The buffer can be accessed after iteration for replay.
    """

    def __init__(
        self,
        source: AsyncIterator[ATPEvent | ATPResponse],
        buffer: EventBuffer | None = None,
    ) -> None:
        """Initialize the buffering iterator.

        Args:
            source: Source iterator to wrap.
            buffer: EventBuffer to store events (creates new one if None).
        """
        self._source = source
        self._buffer = buffer or EventBuffer()

    @property
    def buffer(self) -> EventBuffer:
        """Return the event buffer."""
        return self._buffer

    def __aiter__(self) -> "BufferingEventIterator":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> ATPEvent | ATPResponse:
        """Get next event or response, buffering as we go.

        Returns:
            Next event or response from the source.

        Raises:
            StopAsyncIteration: When source is exhausted.
        """
        item = await self._source.__anext__()

        if isinstance(item, ATPEvent):
            self._buffer.add(item)
        elif isinstance(item, ATPResponse):
            self._buffer.set_response(item)

        return item
