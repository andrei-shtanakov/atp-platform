"""Tests for TournamentEventBus and TournamentEvent."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest


def test_tournament_event_dataclass_holds_required_fields() -> None:
    from atp.dashboard.tournament.events import TournamentEvent

    event = TournamentEvent(
        event_type="round_started",
        tournament_id=7,
        round_number=1,
        data={"foo": "bar"},
        timestamp=datetime(2026, 4, 10, 12, 0, 0, tzinfo=UTC),
    )
    assert event.event_type == "round_started"
    assert event.tournament_id == 7
    assert event.round_number == 1
    assert event.data == {"foo": "bar"}
    assert event.timestamp.year == 2026


@pytest.mark.anyio
async def test_publish_delivers_to_single_subscriber() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    event = TournamentEvent(
        event_type="round_started",
        tournament_id=1,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )

    async with bus.subscribe(tournament_id=1) as queue:
        await bus.publish(event)
        received = await queue.get()
        assert received is event


@pytest.mark.anyio
async def test_publish_to_other_tournament_does_not_reach_subscriber() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    other_event = TournamentEvent(
        event_type="round_started",
        tournament_id=2,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )

    async with bus.subscribe(tournament_id=1) as queue:
        await bus.publish(other_event)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)


@pytest.mark.anyio
async def test_publish_to_no_subscribers_is_noop() -> None:
    from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus

    bus = TournamentEventBus()
    event = TournamentEvent(
        event_type="round_started",
        tournament_id=99,
        round_number=1,
        data={},
        timestamp=datetime.now(tz=UTC),
    )
    await bus.publish(event)


@pytest.mark.anyio
async def test_unsubscribe_on_context_exit_removes_queue() -> None:
    from atp.dashboard.tournament.events import TournamentEventBus

    bus = TournamentEventBus()
    async with bus.subscribe(tournament_id=1):
        pass
    assert 1 not in bus._subscribers
