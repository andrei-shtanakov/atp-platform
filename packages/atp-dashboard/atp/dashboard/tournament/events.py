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

from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason

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
        self._subscribers: dict[int, set[asyncio.Queue[AnyTournamentEvent]]] = {}

    async def publish(self, event: AnyTournamentEvent) -> None:
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
                    getattr(event, "event_type", type(event).__name__),
                )

    @asynccontextmanager
    async def subscribe(
        self, tournament_id: int
    ) -> AsyncIterator[asyncio.Queue[AnyTournamentEvent]]:
        """Subscribe to events for one tournament.

        Usage::

            async with bus.subscribe(tournament_id=7) as queue:
                while True:
                    event = await queue.get()
                    ...

        On context exit (including via task cancellation), the queue
        is removed from the subscribers set.
        """
        queue: asyncio.Queue[AnyTournamentEvent] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.setdefault(tournament_id, set()).add(queue)
        try:
            yield queue
        finally:
            subs = self._subscribers.get(tournament_id)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    del self._subscribers[tournament_id]


_SYSTEM_CANCEL_REASONS: frozenset[CancelReason] = frozenset(
    {
        CancelReason.PENDING_TIMEOUT,
        CancelReason.ABANDONED,
    }
)


@dataclass(frozen=True)
class TournamentCancelEvent:
    """Payload for `tournament_cancelled` bus event.

    Field invariant (enforced three ways — defense in depth):

    1. DB CHECK constraint `ck_tournament_cancel_consistency` on the
       tournaments table.
    2. `__post_init__` validator on this dataclass.
    3. Construction call site in `TournamentService._cancel_impl` —
       always builds from consistent inputs by construction.

    Invariant:
        cancelled_by IS NULL  <->  cancelled_reason in {PENDING_TIMEOUT, ABANDONED}
        cancelled_by NOT NULL <->  cancelled_reason == ADMIN_ACTION
    """

    tournament_id: int
    cancelled_at: datetime
    cancelled_by: int | None
    cancelled_reason: CancelReason
    cancelled_reason_detail: str | None
    final_rounds_played: int
    final_status: TournamentStatus

    def __post_init__(self) -> None:
        is_system = self.cancelled_reason in _SYSTEM_CANCEL_REASONS
        has_actor = self.cancelled_by is not None

        if is_system and has_actor:
            raise ValueError(
                f"system cancel (reason={self.cancelled_reason.value}) "
                f"must have cancelled_by=None, got {self.cancelled_by}"
            )
        if not is_system and not has_actor:
            raise ValueError(
                f"user-initiated cancel (reason={self.cancelled_reason.value}) "
                f"must have cancelled_by set, got None"
            )
        if self.final_status != TournamentStatus.CANCELLED:
            raise ValueError(
                f"TournamentCancelEvent.final_status must be CANCELLED, "
                f"got {self.final_status.value}"
            )


# Union of all event types the bus can carry.  Defined after both
# dataclasses so that `AnyTournamentEvent` is resolvable at runtime
# (with `from __future__ import annotations` all class-body annotations
# are lazy strings, so the forward reference in TournamentEventBus is safe).
AnyTournamentEvent = TournamentEvent | TournamentCancelEvent
