"""MCP tool handlers for tournament gameplay.

This module is imported for side effects (registering tools on the
``mcp_server`` FastMCP instance). The actual logic is in private
``_*_impl`` helpers that take user and service as arguments, which
makes them trivially unit-testable with mocks.

Phase 0 verification (docs/notes/phase0-fastmcp-findings.md) is the
source of truth for the exact ctx attribute paths used in the thin
wrappers below.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Context

from atp.dashboard.mcp import mcp_server, tournament_event_bus
from atp.dashboard.tournament.service import TournamentService

logger = logging.getLogger("atp.dashboard.mcp.tools")


# ---------------------------------------------------------------------------
# Pure-impl helpers (unit-tested directly with mocks)
# ---------------------------------------------------------------------------


async def _join_tournament_impl(
    *,
    tournament_id: int,
    agent_name: str,
    user: Any,
    service: TournamentService,
) -> dict[str, Any]:
    participant = await service.join(
        tournament_id=tournament_id, user=user, agent_name=agent_name
    )
    return {
        "tournament_id": tournament_id,
        "participant_id": participant.id,
        "agent_name": agent_name,
        "status": "joined",
    }


# ---------------------------------------------------------------------------
# FastMCP-registered tool entry points
# ---------------------------------------------------------------------------


@mcp_server.tool()
async def join_tournament(
    ctx: Context,
    tournament_id: int,
    agent_name: str,
) -> dict:
    """Join an open tournament.

    Starts an event subscription for this MCP session — ``round_started``
    and ``tournament_completed`` notifications are pushed automatically
    until the session disconnects.
    """
    from atp.dashboard.mcp.notifications import (
        forward_events_to_session,
        resolve_user_from_ctx,
        with_service,
    )

    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        result = await _join_tournament_impl(
            tournament_id=tournament_id,
            agent_name=agent_name,
            user=user,
            service=service,
        )

    await forward_events_to_session(ctx, tournament_id, user)
    return result
