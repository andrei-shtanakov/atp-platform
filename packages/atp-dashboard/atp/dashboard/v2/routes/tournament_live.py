"""Live tournament dashboard endpoints.

Two routes, both anonymous-friendly when the tournament is public
(``join_token IS NULL``):

  GET /api/v1/tournaments/{tournament_id}/dashboard
      One-shot JSON snapshot of the El Farol DashboardPayload built from
      live ORM rows. Same shape as the post-completion endpoint
      ``/api/v1/games/{match_id}/dashboard``; only resolved rounds are
      included in ``DATA``.

  GET /api/v1/tournaments/{tournament_id}/dashboard/stream
      Server-Sent Events stream. Sends the initial snapshot, then a
      fresh snapshot on each ``round_ended`` and ``tournament_completed``
      event. ``event: completed`` is sent at the very end with the
      replay's ``match_id`` (when the dual-write succeeded). Heartbeat
      comments every 15 s keep idle proxies from closing the connection.

Visibility rule is shared with ``ui.py`` via ``is_tournament_visible_to``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from atp.dashboard.database import get_database
from atp.dashboard.models import GameResult
from atp.dashboard.tournament.events import (
    TournamentCancelEvent,
    TournamentEventBus,
)
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.visibility import is_tournament_visible_to
from atp.dashboard.v2.dependencies import CurrentUser, DBSession
from atp.dashboard.v2.routes.el_farol_dashboard import DashboardPayload
from atp.dashboard.v2.routes.el_farol_from_tournament import (
    _reshape_from_tournament,
)

router = APIRouter(prefix="/v1/tournaments", tags=["tournaments", "dashboard"])

_HEARTBEAT_SECONDS = 15.0

# `_complete_tournament` and `_cancel_impl` publish their terminal events
# from inside the still-uncommitted service-layer transaction (the
# request handler / MCP session owns the commit boundary). A fresh
# session opened here can race ahead of the commit and miss the
# just-finalized round payoffs and the GameResult.match_id. Poll the
# committed Tournament.status before emitting the final snapshot and
# `completed` event so spectators always see the canonical end state.
_COMPLETION_WAIT_TIMEOUT_S = 30.0
_COMPLETION_POLL_INTERVAL_S = 0.1
_TERMINAL_STATUSES = (TournamentStatus.COMPLETED, TournamentStatus.CANCELLED)


async def _wait_for_committed_terminal_status(
    request: Request, tournament_id: int
) -> bool:
    """Poll until Tournament.status is COMPLETED/CANCELLED in a fresh
    session, the client disconnects, or ``_COMPLETION_WAIT_TIMEOUT_S``
    elapses. Returns True if the terminal status was observed."""
    db = get_database()
    deadline = asyncio.get_event_loop().time() + _COMPLETION_WAIT_TIMEOUT_S
    while True:
        async with db.session_factory() as session:
            row_status = await session.scalar(
                select(Tournament.status).where(Tournament.id == tournament_id)
            )
            if row_status in _TERMINAL_STATUSES:
                return True
        if await request.is_disconnected():
            return False
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(_COMPLETION_POLL_INTERVAL_S)


def _bus_from_request(request: Request) -> TournamentEventBus:
    """Resolve the shared TournamentEventBus.

    Prefers ``app.state.tournament_event_bus`` (set in factory wiring or
    overridden in tests) and falls back to the ``mcp`` module singleton
    that ``TournamentService`` and the MCP notification forwarder share.
    """
    from atp.dashboard.mcp import tournament_event_bus

    return getattr(request.app.state, "tournament_event_bus", tournament_event_bus)


async def _gate_tournament_access(session, tournament_id: int, user) -> Tournament:
    tournament = await session.get(Tournament, tournament_id)
    if tournament is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"tournament {tournament_id} not found",
        )
    if not await is_tournament_visible_to(session, tournament, user):
        # Hide existence: same 404 for "does not exist" and "not visible"
        # so the endpoint can't be used as an enumeration oracle.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"tournament {tournament_id} not found",
        )
    return tournament


@router.get(
    "/{tournament_id}/dashboard",
    response_model=DashboardPayload,
    summary="Live El Farol dashboard payload (round-level snapshot)",
    responses={
        404: {"description": "tournament not found or not visible"},
    },
)
async def get_tournament_dashboard(
    tournament_id: int,
    session: DBSession,
    user: CurrentUser,
) -> DashboardPayload:
    """Return the El Farol DashboardPayload built from live ORM rows.

    Only rounds with status COMPLETED are included; in-progress rounds
    are omitted to match the existing post-resolution visibility rule
    (no fog-of-war, no leakage of pending opponent moves).
    """
    await _gate_tournament_access(session, tournament_id, user)
    return await _reshape_from_tournament(tournament_id, session)


def _format_sse(event_type: str, payload: dict) -> bytes:
    return (
        f"event: {event_type}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode()


async def _resolved_match_id(tournament_id: int) -> str | None:
    """Look up the GameResult.match_id for a completed tournament.

    Opens a short-lived session so the read sees the post-commit state
    written by ``_complete_tournament._write_game_result_for_tournament``.
    Returns None if the dual-write hasn't landed yet (race) or is absent.
    """
    db = get_database()
    async with db.session_factory() as session:
        row = (
            await session.execute(
                select(GameResult).where(GameResult.tournament_id == tournament_id)
            )
        ).scalar_one_or_none()
        return row.match_id if row is not None else None


async def _project_snapshot_fresh(tournament_id: int) -> DashboardPayload:
    """Project a snapshot using a short-lived DB session.

    Each snapshot emission opens its own session so a long-running SSE
    stream doesn't pin a connection from the request-scoped session for
    minutes — and so the read always sees the latest committed state.
    """
    db = get_database()
    async with db.session_factory() as session:
        return await _reshape_from_tournament(tournament_id, session)


async def _sse_event_generator(
    request: Request, tournament_id: int
) -> AsyncGenerator[bytes, None]:
    """Yield SSE chunks until the client disconnects or tournament ends."""
    bus = _bus_from_request(request)

    initial = await _project_snapshot_fresh(tournament_id)
    yield _format_sse("snapshot", initial.model_dump(mode="json"))

    async with bus.subscribe(tournament_id) as queue:
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
            except TimeoutError:
                yield b": ping\n\n"
                continue

            if isinstance(event, TournamentCancelEvent):
                # Tournament cancelled — push a final snapshot of whatever
                # rounds resolved before cancellation, then close the
                # stream with a `completed` event (no match_id, since
                # cancelled tournaments still get a GameResult dual-write).
                # The cancel event is published mid-transaction by
                # _cancel_impl's caller; wait for the commit so the
                # final read sees the canonical state.
                await _wait_for_committed_terminal_status(request, tournament_id)
                snapshot = await _project_snapshot_fresh(tournament_id)
                yield _format_sse("snapshot", snapshot.model_dump(mode="json"))
                match_id = await _resolved_match_id(tournament_id)
                yield _format_sse(
                    "completed",
                    {"match_id": match_id, "cancelled": True},
                )
                return

            if event.event_type == "round_ended":
                # round_ended is published post-commit (see service.py),
                # so a fresh read already reflects the resolved round.
                snapshot = await _project_snapshot_fresh(tournament_id)
                yield _format_sse("snapshot", snapshot.model_dump(mode="json"))

            if event.event_type == "tournament_completed":
                # Published mid-transaction by _complete_tournament. Wait
                # for the commit so the snapshot includes the final
                # round's payoffs and _resolved_match_id() can see the
                # GameResult dual-write.
                await _wait_for_committed_terminal_status(request, tournament_id)
                snapshot = await _project_snapshot_fresh(tournament_id)
                yield _format_sse("snapshot", snapshot.model_dump(mode="json"))
                match_id = await _resolved_match_id(tournament_id)
                yield _format_sse("completed", {"match_id": match_id})
                return


@router.get(
    "/{tournament_id}/dashboard/stream",
    summary="Live tournament dashboard SSE stream",
    responses={
        200: {"content": {"text/event-stream": {}}},
        404: {"description": "tournament not found or not visible"},
    },
)
async def stream_tournament_dashboard(
    tournament_id: int,
    request: Request,
    session: DBSession,
    user: CurrentUser,
) -> StreamingResponse:
    """Open a Server-Sent Events stream for a live tournament.

    Visibility is checked once at request time using a normal session;
    the streaming generator then opens its own short-lived sessions so
    the request session is released back to the pool after the gate.
    """
    await _gate_tournament_access(session, tournament_id, user)

    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _sse_event_generator(request, tournament_id),
        media_type="text/event-stream",
        headers=headers,
    )


__all__ = ["router"]


# ---------------------------------------------------------------------------
# Test hooks (kept module-level so tests can monkeypatch them)
# ---------------------------------------------------------------------------

# ``_project_snapshot_fresh`` and ``_resolved_match_id`` are intentionally
# module-level (not nested) so tests can swap them without touching the
# database. The SSE generator references them via the module binding so
# monkeypatch-style replacement works cleanly.
