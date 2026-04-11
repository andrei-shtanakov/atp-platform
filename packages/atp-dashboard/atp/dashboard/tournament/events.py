"""Tournament event bus and event dataclass.

Protocol-agnostic in-process pub/sub. Used by TournamentService to
broadcast state-change events; consumed by the MCP notification layer
to push notifications/message to connected clients.

Events are ephemeral: not persisted, not replayed on subscriber
reconnect (per AD-7 in the design spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

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
