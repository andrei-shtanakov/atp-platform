"""Tournament event bus and event dataclass.

Protocol-agnostic in-process pub/sub. Used by TournamentService to
broadcast state-change events; consumed by the MCP notification layer
to push notifications/message to connected clients.

Events are ephemeral: not persisted, not replayed on subscriber
reconnect (per AD-7 in the design spec).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger("atp.dashboard.tournament.events")

_QUEUE_MAXSIZE = 100

EventType = Literal[
    "round_started",
    "round_ended",
    "tournament_completed",
    "tournament_cancelled",
]


@dataclass(frozen=True)
class TournamentEvent:
    """One state-change event in a tournament's lifecycle.

    The ``data`` field is a generic dict; per-player personalization
    happens at the subscriber level (the MCP notification layer calls
    TournamentService.get_state_for to format a player-private view).
    """

    event_type: EventType
    tournament_id: int
    round_number: int | None
    data: dict[str, Any]
    timestamp: datetime


class TournamentEventBus:
    """In-process pub/sub for tournament events.

    Subscribers register per-tournament_id and receive only events for
    that tournament. Each subscriber gets its own asyncio.Queue
    (maxsize=100) so a slow consumer cannot block others.

    Events are ephemeral. Missed events on disconnect are NOT replayed.
    See AD-4 and AD-7 in the design spec.
    """

    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue[TournamentEvent]]] = {}

    async def publish(self, event: TournamentEvent) -> None:
        """Fan out an event to all subscribers of event.tournament_id.

        Best-effort: if a subscriber's queue is full, the event is
        dropped for that subscriber and a warning is logged. Never
        raises — publish is fire-and-forget.
        """
        queues = self._subscribers.get(event.tournament_id, set())
        for queue in list(queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "subscriber queue full for tournament %d, dropping event %s",
                    event.tournament_id,
                    event.event_type,
                )

    @asynccontextmanager
    async def subscribe(
        self, tournament_id: int
    ) -> AsyncIterator[asyncio.Queue[TournamentEvent]]:
        """Subscribe to events for one tournament.

        Usage::

            async with bus.subscribe(tournament_id=7) as queue:
                while True:
                    event = await queue.get()
                    ...

        On context exit (including via task cancellation), the queue
        is removed from the subscribers set.
        """
        queue: asyncio.Queue[TournamentEvent] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.setdefault(tournament_id, set()).add(queue)
        try:
            yield queue
        finally:
            subs = self._subscribers.get(tournament_id)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    del self._subscribers[tournament_id]
