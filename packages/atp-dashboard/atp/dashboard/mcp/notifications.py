"""Per-session event forwarder and notification formatter.

Stubbed in Task 7.1; fleshed out in Task 7.3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.service import TournamentService


async def resolve_user_from_ctx(ctx: Any) -> User:
    """Look up the User row for this MCP session (implemented in 7.3)."""
    raise NotImplementedError("implemented in Task 7.3")


@asynccontextmanager
async def with_service(
    ctx: Any, bus: TournamentEventBus
) -> AsyncIterator[TournamentService]:
    """Yield a TournamentService bound to a fresh DB session (7.3)."""
    if True:
        raise NotImplementedError("implemented in Task 7.3")
    yield  # type: ignore[unreachable]  # satisfies async-generator typing


async def forward_events_to_session(ctx: Any, tournament_id: int, user: User) -> None:
    """Spawn the per-session notification forwarder (7.3)."""
    raise NotImplementedError("implemented in Task 7.3")
