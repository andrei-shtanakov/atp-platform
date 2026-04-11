"""Per-session event forwarder and notification formatter.

When a player joins a tournament via the ``join_tournament`` tool,
this module spawns a background ``asyncio.Task`` that subscribes to
the ``TournamentEventBus`` for that tournament and forwards each
event as an MCP ``notifications/message`` to the session.

Per-player personalization happens HERE, not in the service layer
(see Service-layer §Invariants in the design spec).

Phase 0 verification (``docs/notes/phase0-fastmcp-findings.md``)
provides the exact ``ctx`` attribute paths used in this module.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from atp.dashboard.database import get_database
from atp.dashboard.models import User
from atp.dashboard.tournament.events import (
    AnyTournamentEvent,
    TournamentCancelEvent,
    TournamentEventBus,
)
from atp.dashboard.tournament.service import TournamentService

logger = logging.getLogger("atp.dashboard.mcp.notifications")

# session_id → tournament_id → asyncio.Task
_session_tasks: dict[Any, dict[int, asyncio.Task[None]]] = {}


# ---------------------------------------------------------------------------
# ctx helpers — attribute paths come from Phase 0 findings
# ---------------------------------------------------------------------------


async def resolve_user_from_ctx(ctx: Any) -> User:
    """Look up the User row corresponding to ctx's session.

    Reads ``request.state.user_id`` from the underlying Starlette
    request via FastMCP's public ``get_http_request()`` helper — the
    verified pattern from Phase 0.2. Do NOT use the internal
    ``ctx.request_context.request`` path (version-fragile).
    """
    from fastmcp.server.dependencies import get_http_request

    request = get_http_request()
    user_id: int | None = getattr(request.state, "user_id", None)
    if user_id is None:
        raise RuntimeError("MCP session has no authenticated user_id")

    db = get_database()
    async with db.session() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise RuntimeError(f"user {user_id} not found in database")
        return user


@asynccontextmanager
async def with_service(
    ctx: Any, bus: TournamentEventBus
) -> AsyncIterator[TournamentService]:
    """Yield a TournamentService bound to a fresh DB session.

    The session is committed and closed on context exit via
    ``Database.session()``'s own lifecycle.
    """
    db = get_database()
    async with db.session() as session:
        yield TournamentService(session, bus)


# ---------------------------------------------------------------------------
# Notification formatting and forwarding
# ---------------------------------------------------------------------------


async def _format_notification_for_user(
    event: AnyTournamentEvent,
    user: User,
    service: TournamentService,
) -> dict[str, Any] | None:
    """Convert a TournamentEvent into a per-player MCP notifications/message.

    For ``round_started``, calls ``service.get_state_for`` to build the
    player-private RoundState. For ``tournament_completed`` the final
    scoreboard is global so no per-player call is needed.

    Returns the structured payload that the forwarder passes to
    ``session.send_log_message(..., data=payload)``. The wire-level
    envelope is built by the MCP SDK; the outer ``method`` /
    ``params`` keys are retained here so unit tests can assert on the
    same shape the clients ultimately see.
    """
    if isinstance(event, TournamentCancelEvent):
        payload = _format_for_user(event, user)
        if payload is None:
            return None
        return {
            "method": "notifications/message",
            "params": {
                "level": "info",
                "logger": "atp.tournament",
                "data": payload,
            },
        }

    # From here on, event is narrowed to TournamentEvent.
    if event.event_type == "round_started":
        state = await service.get_state_for(event.tournament_id, user)
        return {
            "method": "notifications/message",
            "params": {
                "level": "info",
                "logger": "atp.tournament",
                "data": {
                    "event": "round_started",
                    "tournament_id": event.tournament_id,
                    "round_number": event.round_number,
                    "state": state.to_dict(),
                },
            },
        }
    if event.event_type == "tournament_completed":
        return {
            "method": "notifications/message",
            "params": {
                "level": "info",
                "logger": "atp.tournament",
                "data": {
                    "event": "tournament_completed",
                    "tournament_id": event.tournament_id,
                    "final_scores": event.data.get("final_scores", {}),
                },
            },
        }
    return None


async def forward_events_to_session(ctx: Any, tournament_id: int, user: User) -> None:
    """Spawn a background task that forwards bus events for one
    tournament to one MCP session. The task is cancelled when the
    session disconnects or leaves the tournament.
    """
    from atp.dashboard.mcp import tournament_event_bus

    session_id = getattr(ctx, "session_id", None) or id(ctx)

    async def _forward() -> None:
        try:
            async with tournament_event_bus.subscribe(tournament_id) as queue:
                while True:
                    event = await queue.get()
                    db = get_database()
                    async with db.session() as session:
                        service = TournamentService(session, tournament_event_bus)
                        notification = await _format_notification_for_user(
                            event, user, service
                        )
                    if notification is None:
                        continue
                    params = notification["params"]
                    session = getattr(ctx, "session", None)
                    if session is None:
                        continue
                    # ``send_log_message`` builds the proper pydantic
                    # notification model internally; we pass the raw
                    # data payload our formatter produced. The SDK
                    # wraps it into a ``notifications/message`` frame
                    # on the wire.
                    await session.send_log_message(
                        level=params["level"],
                        data=params["data"],
                        logger=params.get("logger"),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "notification forwarder crashed for session=%s tournament=%d",
                session_id,
                tournament_id,
            )

    task = asyncio.create_task(_forward())
    _session_tasks.setdefault(session_id, {})[tournament_id] = task
    # Let the forwarder's ``subscribe()`` context register on the bus
    # before returning — otherwise the caller may publish events that
    # this fresh subscriber misses.
    await asyncio.sleep(0)


async def _cancel_session_task(ctx: Any, tournament_id: int) -> None:
    """Cancel the per-session forwarder task for one tournament."""
    session_id = getattr(ctx, "session_id", None) or id(ctx)
    tasks = _session_tasks.get(session_id)
    if tasks is None:
        return
    task = tasks.pop(tournament_id, None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    if not tasks:
        _session_tasks.pop(session_id, None)


def _format_for_user(event: Any, user: Any) -> dict[str, Any] | None:
    """Dispatch-based notification formatter.

    Takes a bus event and a recipient User, returns the personalized
    ``data`` dict to send via session.send_notification, or None if the
    event is not deliverable to this user.

    Responsibilities:
    1. Add ``event`` discriminator.
    2. Apply game-specific reveal semantics (Plan 2a is PD-only, near-identity).
    3. Privacy filter admin-only fields (cancelled_by for non-admins).
    """
    if isinstance(event, TournamentCancelEvent):
        cancelled_by = event.cancelled_by if user.is_admin else None
        return {
            "event": "tournament_cancelled",
            "tournament_id": event.tournament_id,
            "reason": event.cancelled_reason.value,
            "reason_detail": event.cancelled_reason_detail,
            "cancelled_by": cancelled_by,
            "final_rounds_played": event.final_rounds_played,
            "cancelled_at": event.cancelled_at.isoformat(),
        }

    # round_started, round_ended, tournament_completed events are
    # existing dict-shaped payloads from the vertical slice bus.
    # For Plan 2a they pass through with an added `event` discriminator
    # only — no per-recipient privacy filtering is needed because PD
    # reveals all actions after each round (source spec §Notification
    # personalization). Future hidden-information games override this
    # branch per-game.
    if isinstance(event, dict) and "event_type" in event:
        event_type = event["event_type"]
        payload = {k: v for k, v in event.items() if k != "event_type"}
        payload["event"] = event_type
        return payload

    return None
