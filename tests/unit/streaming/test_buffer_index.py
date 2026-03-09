"""Tests for EventBuffer sequence index (O(1) lookup)."""

from datetime import UTC, datetime

from atp.protocol import ATPEvent, EventType
from atp.streaming.buffer import EventBuffer


def _make_event(seq: int) -> ATPEvent:
    return ATPEvent(
        task_id="t1",
        event_type=EventType.PROGRESS,
        payload={"step": seq},
        sequence=seq,
        timestamp=datetime.now(tz=UTC),
    )


class TestSequenceIndex:
    def test_o1_lookup(self) -> None:
        buf = EventBuffer(max_size=100)
        for i in range(10):
            buf.add(_make_event(i))
        assert buf.get_by_sequence(5) is not None
        assert buf.get_by_sequence(5).sequence == 5
        assert buf.get_by_sequence(99) is None

    def test_eviction_cleans_index(self) -> None:
        buf = EventBuffer(max_size=3)
        buf.add(_make_event(1))
        buf.add(_make_event(2))
        buf.add(_make_event(3))
        # Adding 4th evicts seq=1
        buf.add(_make_event(4))
        assert buf.get_by_sequence(1) is None
        assert buf.get_by_sequence(2) is not None
        assert buf.get_by_sequence(4) is not None

    def test_clear_resets_index(self) -> None:
        buf = EventBuffer(max_size=100)
        buf.add(_make_event(1))
        buf.clear()
        assert buf.get_by_sequence(1) is None
