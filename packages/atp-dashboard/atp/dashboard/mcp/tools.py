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
    participant, is_new = await service.join(
        tournament_id=tournament_id, user=user, agent_name=agent_name
    )
    return {
        "tournament_id": tournament_id,
        "participant_id": participant.id,
        "agent_name": agent_name,
        "status": "joined",
        "is_new": is_new,
    }


async def join_tournament(
    ctx: Any,
    service: TournamentService,
    user: Any,
    tournament_id: int,
    agent_name: str,
    join_token: str | None = None,
) -> dict[str, Any]:
    """MCP tool handler: idempotent join with session_sync on every call.

    Called both from the FastMCP tool decorator shim and from integration
    tests. Extracting the body from the decorator makes it testable
    without a live MCP session.

    session_sync is emitted on BOTH new-join (is_new=True) AND reconnect
    (is_new=False) so that the MCP client can catch up on state it missed
    while disconnected.
    """
    participant, is_new = await service.join(
        tournament_id=tournament_id,
        user=user,
        agent_name=agent_name,
        join_token=join_token,
    )
    await service._session.commit()

    try:
        state = await service.get_state_for(tournament_id=tournament_id, user=user)
        state_payload: dict[str, Any] = (
            state.to_dict() if hasattr(state, "to_dict") else state  # type: ignore[assignment]
        )
    except Exception:
        # Tournament may be PENDING with fewer than num_players joined;
        # state is not yet available. Return a minimal pending snapshot.
        state_payload = {"status": "pending", "tournament_id": tournament_id}

    session_sync_payload = {
        "event": "session_sync",
        "tournament_id": tournament_id,
        "state": state_payload,
    }
    await ctx.session.send_notification(session_sync_payload)

    return {
        "joined": True,
        "participant_id": participant.id,
        "is_new": is_new,
    }


async def _get_current_state_impl(
    *,
    tournament_id: int,
    user: Any,
    service: TournamentService,
) -> dict[str, Any]:
    state = await service.get_state_for(tournament_id, user)
    return state.to_dict()


async def _make_move_impl(
    *,
    tournament_id: int,
    action: dict[str, Any],
    user: Any,
    service: TournamentService,
) -> dict[str, Any]:
    return await service.submit_action(tournament_id, user, action=action)


# ---------------------------------------------------------------------------
# FastMCP-registered tool entry points
# ---------------------------------------------------------------------------


@mcp_server.tool()
async def _join_tournament_mcp(
    ctx: Context,
    tournament_id: int,
    agent_name: str,
    join_token: str | None = None,
) -> dict:
    """FastMCP shim: join an open tournament.

    Starts an event subscription for this MCP session BEFORE calling
    the service — otherwise the ``round_started`` event that fires
    when our join fills the last slot races us and is published to a
    bus channel that nobody is listening on yet. With the subscriber
    attached first, that first event is guaranteed to reach us.

    Delegates to the testable ``join_tournament`` function after
    setting up the event subscription.
    """
    from atp.dashboard.mcp.notifications import (
        forward_events_to_session,
        resolve_user_from_ctx,
        with_service,
    )

    user = await resolve_user_from_ctx(ctx)

    await forward_events_to_session(ctx, tournament_id, user)
    try:
        async with with_service(ctx, tournament_event_bus) as service:
            result = await join_tournament(
                ctx=ctx,
                service=service,
                user=user,
                tournament_id=tournament_id,
                agent_name=agent_name,
                join_token=join_token,
            )
    except Exception:
        # Join failed — cancel the orphan forwarder task so we don't
        # leak a subscription for a tournament the caller never
        # actually joined.
        from atp.dashboard.mcp.notifications import _cancel_session_task

        await _cancel_session_task(ctx, tournament_id)
        raise

    return result


@mcp_server.tool()
async def get_current_state(
    ctx: Context,
    tournament_id: int,
) -> dict:
    """Return a player-private RoundState for the current round."""
    from atp.dashboard.mcp.notifications import (
        resolve_user_from_ctx,
        with_service,
    )

    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await _get_current_state_impl(
            tournament_id=tournament_id, user=user, service=service
        )


@mcp_server.tool()
async def make_move(
    ctx: Context,
    tournament_id: int,
    action: dict,
) -> dict:
    """Submit an action for the current round."""
    from atp.dashboard.mcp.notifications import (
        resolve_user_from_ctx,
        with_service,
    )

    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await _make_move_impl(
            tournament_id=tournament_id,
            action=action,
            user=user,
            service=service,
        )
