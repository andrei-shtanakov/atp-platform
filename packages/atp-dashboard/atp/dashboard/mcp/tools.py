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
from datetime import UTC, datetime
from typing import Any

from fastmcp import Context

from atp.dashboard.mcp import mcp_server, tournament_event_bus
from atp.dashboard.mcp.observability import emit_tool_call
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
    agent_id: int | None = None,
) -> dict[str, Any]:
    participant, is_new = await service.join(
        tournament_id=tournament_id,
        user=user,
        agent_name=agent_name,
        agent_id=agent_id,
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
    agent_id: int | None = None,
) -> dict[str, Any]:
    """MCP tool handler: idempotent join with session_sync on every call.

    Called both from the FastMCP tool decorator shim and from integration
    tests. Extracting the body from the decorator makes it testable
    without a live MCP session.

    LABS-TSA PR-6 (post-review): ``agent_id`` comes from the agent-scoped
    token on the MCP transport. When provided, service.join() uses it
    directly and skips the by-name auto-provision path — critical for
    users running multiple agents, where name-only lookup cannot tell
    the agents apart safely.

    session_sync is emitted on BOTH new-join (is_new=True) AND reconnect
    (is_new=False) so that the MCP client can catch up on state it missed
    while disconnected.
    """
    participant, is_new = await service.join(
        tournament_id=tournament_id,
        user=user,
        agent_name=agent_name,
        join_token=join_token,
        agent_id=agent_id,
    )
    await service._session.commit()

    try:
        state = await service.get_state_for(
            tournament_id=tournament_id, user=user, agent_id=agent_id
        )
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
    # Use send_log_message so the payload is wrapped in a proper
    # notifications/message JSON-RPC frame — send_notification requires a
    # typed Pydantic notification model, not a plain dict.
    await ctx.session.send_log_message(
        level="info",
        data=session_sync_payload,
        logger="atp.tournament",
    )

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
    agent_id: int | None = None,
) -> dict[str, Any]:
    state = await service.get_state_for(tournament_id, user, agent_id=agent_id)
    return state.to_dict()


async def _make_move_impl(
    *,
    tournament_id: int,
    action: dict[str, Any],
    user: Any,
    service: TournamentService,
    agent_id: int | None = None,
) -> dict[str, Any]:
    return await service.submit_action(
        tournament_id, user, action=action, agent_id=agent_id
    )


async def _ping_impl() -> dict[str, Any]:
    """Pure-impl helper for the ``ping`` tool.

    No DB access, no service dependency — the response is built from
    in-process state only. Designed as a connectivity warm-up SDK
    clients can call before issuing real tournament operations.

    Diagnostic interpretation of failure modes:

    - ``ping`` missing from the advertised tool list (or any
      transport-level error returned by the SSE handshake itself):
      tool registration / handshake has not completed yet — the
      client should retry the SSE connection before attempting real
      operations.
    - ``401 Unauthorized``: an authentication problem (missing,
      invalid, or expired token / wrong token purpose). This is NOT
      a handshake-retry signal; the client should fix or refresh
      credentials rather than reconnect in a loop.

    See docs/superpowers/plans/2026-04-27-mcp-server-reliability.md
    Task 3 for the broader context.
    """
    from atp.dashboard.v2.config import get_config

    return {
        "ok": True,
        "server_version": get_config().version,
        "ts": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# FastMCP-registered tool entry points
# ---------------------------------------------------------------------------


@mcp_server.tool(name="join_tournament")
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
        resolve_agent_id_from_ctx,
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="join_tournament")
    user = await resolve_user_from_ctx(ctx)
    agent_id = resolve_agent_id_from_ctx(ctx)

    await forward_events_to_session(ctx, tournament_id, user, agent_id=agent_id)
    try:
        async with with_service(ctx, tournament_event_bus) as service:
            result = await join_tournament(
                ctx=ctx,
                service=service,
                user=user,
                tournament_id=tournament_id,
                agent_name=agent_name,
                join_token=join_token,
                agent_id=agent_id,
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
        resolve_agent_id_from_ctx,
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="get_current_state")
    user = await resolve_user_from_ctx(ctx)
    agent_id = resolve_agent_id_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await _get_current_state_impl(
            tournament_id=tournament_id,
            user=user,
            service=service,
            agent_id=agent_id,
        )


@mcp_server.tool()
async def ping(ctx: Context) -> dict:
    """Cheap connectivity probe — no DB access, no auth lookup beyond
    what ``MCPAuthMiddleware`` already enforced.

    Use this as a warm-up before issuing real tournament operations.
    If ``ping`` is missing from the discovered tool list or the SSE
    handshake itself surfaces a transport-level error, tool
    registration hasn't finished — retry the connection rather than
    calling ``join_tournament`` / ``make_move`` and racing the
    registration.

    A 401 here means the credentials are bad (missing, invalid, or
    wrong purpose), not that the handshake is incomplete; clients
    should refresh credentials, not reconnect in a loop.

    Returns:
        ``{"ok": True, "server_version": "<dashboard version>", "ts": "<iso8601 UTC>"}``
    """
    emit_tool_call(ctx, tool="ping")
    return await _ping_impl()


@mcp_server.tool()
async def make_move(
    ctx: Context,
    tournament_id: int,
    action: dict,
) -> dict:
    """Submit an action for the current round.

    Args:
        tournament_id: The tournament to submit to.
        action: Dict whose required fields depend on the tournament's
            game_type.

            - prisoners_dilemma: ``{"choice": "cooperate" | "defect"}``
            - stag_hunt: ``{"choice": "stag" | "hare"}``
            - battle_of_sexes: ``{"choice": "A" | "B"}``
            - el_farol: ``{"slots": list[int], values in [0, num_slots-1],
              unique, max 8 entries}``. El Farol players may attend at most
              8 of 16 slots per day.
            - public_goods: ``{"contribution": float in [0, endowment]}``.
              Endowment defaults to 20; your contribution is pooled,
              multiplied by 1.6, and split equally among all N players.

            Optional for every game type:

            - ``reasoning``: short free-form rationale for this move
              (max 8000 chars; empty/whitespace-only treated as absent).
              Example:
              ``{"choice": "defect", "reasoning": "Opponent defected in
              round 2; retaliating to discourage exploitation."}``
              Revealed publicly only after tournament completion; visible
              during live play only to the tournament owner, admins, and
              the submitting agent itself.

    ``game_type`` is optional on the wire; the server reads it from the
    tournament record. If you send it and it mismatches the tournament's
    game_type, the call is rejected (422).
    """
    from atp.dashboard.mcp.notifications import (
        resolve_agent_id_from_ctx,
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="make_move")
    user = await resolve_user_from_ctx(ctx)
    agent_id = resolve_agent_id_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await _make_move_impl(
            tournament_id=tournament_id,
            action=action,
            user=user,
            service=service,
            agent_id=agent_id,
        )


# ---------------------------------------------------------------------------
# Plan 2a: five additional MCP tool functions (unit-testable, no decorator)
# ---------------------------------------------------------------------------


async def leave_tournament(
    ctx: Any,
    service: Any,
    user: Any,
    tournament_id: int,
    agent_id: int | None = None,
) -> dict[str, Any]:
    """MCP tool: leave a tournament.

    LABS-TSA PR-6 (post-review): ``agent_id`` routes the leave to the
    caller's specific Participant when the user has multiple agents in
    the tournament. MCP agent-scoped callers always pass it.

    Idempotent at the DB level — a retry after a successful-but-unacknowledged
    first call will return NotFoundError; SDK callers MUST treat that as
    success.
    """
    await service.leave(tournament_id=tournament_id, user=user, agent_id=agent_id)
    return {"left": True, "tournament_id": tournament_id}


async def get_history(
    ctx: Any,
    service: Any,
    user: Any,
    tournament_id: int,
    last_n: int | None = None,
) -> dict[str, Any]:
    """MCP tool: retrieve round history for a tournament."""
    rounds = await service.get_history(
        tournament_id=tournament_id, user=user, last_n=last_n
    )
    return {
        "tournament_id": tournament_id,
        "rounds": [
            {
                "round_number": r.round_number,
                "status": getattr(r, "status", None),
            }
            for r in rounds
        ],
    }


async def list_tournaments(
    ctx: Any,
    service: Any,
    user: Any,
    status: str | None = None,
    game_type: str | None = None,
) -> dict[str, Any]:
    """MCP tool: list tournaments, optionally filtered by status and game_type.

    Args:
        status: optional filter; one of the TournamentStatus values.
        game_type: optional filter; one of "prisoners_dilemma" |
            "stag_hunt" | "battle_of_sexes" | "el_farol" | "public_goods".
    """
    from atp.dashboard.tournament.models import TournamentStatus

    status_filter = TournamentStatus(status) if status else None
    tournaments = await service.list_tournaments(
        user=user, status=status_filter, game_type=game_type
    )
    return {
        "tournaments": [
            {
                "id": t.id,
                "name": (getattr(t, "config", {}) or {}).get("name", ""),
                "status": getattr(t, "status", None),
                "game_type": getattr(t, "game_type", None),
                "has_join_token": bool(getattr(t, "join_token", None)),
            }
            for t in tournaments
        ]
    }


async def get_tournament(
    ctx: Any,
    service: Any,
    user: Any,
    tournament_id: int,
) -> dict[str, Any]:
    """MCP tool: get details for a single tournament."""
    t = await service.get_tournament(tournament_id=tournament_id, user=user)
    return {
        "tournament": {
            "id": t.id,
            "name": (getattr(t, "config", {}) or {}).get("name", ""),
            "status": getattr(t, "status", None),
            "has_join_token": bool(getattr(t, "join_token", None)),
        }
    }


async def cancel_tournament(
    ctx: Any,
    service: Any,
    user: Any,
    tournament_id: int,
    reason_detail: str | None = None,
) -> dict[str, Any]:
    """MCP tool: cancel a tournament (admin or owner only)."""
    await service.cancel_tournament(
        user=user,
        tournament_id=tournament_id,
        reason_detail=reason_detail,
    )
    return {"cancelled": True, "tournament_id": tournament_id}


# ---------------------------------------------------------------------------
# FastMCP-registered wrappers for the Plan 2a tools
# ---------------------------------------------------------------------------


@mcp_server.tool()
async def mcp_leave_tournament(
    ctx: Context,
    tournament_id: int,
) -> dict:
    """Leave a tournament. Idempotent."""
    from atp.dashboard.mcp.notifications import (
        resolve_agent_id_from_ctx,
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="mcp_leave_tournament")
    user = await resolve_user_from_ctx(ctx)
    agent_id = resolve_agent_id_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await leave_tournament(
            ctx=ctx,
            service=service,
            user=user,
            tournament_id=tournament_id,
            agent_id=agent_id,
        )


@mcp_server.tool()
async def mcp_get_history(
    ctx: Context,
    tournament_id: int,
    last_n: int | None = None,
) -> dict:
    """Return round history for a tournament."""
    from atp.dashboard.mcp.notifications import (
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="mcp_get_history")
    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await get_history(
            ctx=ctx,
            service=service,
            user=user,
            tournament_id=tournament_id,
            last_n=last_n,
        )


@mcp_server.tool()
async def mcp_list_tournaments(
    ctx: Context,
    status: str | None = None,
    game_type: str | None = None,
) -> dict:
    """List tournaments, optionally filtered by status and/or game_type.

    Args:
        status: optional; one of the TournamentStatus values.
        game_type: optional; one of "prisoners_dilemma" | "stag_hunt" |
            "battle_of_sexes" | "el_farol" | "public_goods".
    """
    from atp.dashboard.mcp.notifications import (
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="mcp_list_tournaments")
    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await list_tournaments(
            ctx=ctx, service=service, user=user, status=status, game_type=game_type
        )


@mcp_server.tool()
async def mcp_get_tournament(
    ctx: Context,
    tournament_id: int,
) -> dict:
    """Get details for a single tournament."""
    from atp.dashboard.mcp.notifications import (
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="mcp_get_tournament")
    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await get_tournament(
            ctx=ctx, service=service, user=user, tournament_id=tournament_id
        )


@mcp_server.tool()
async def mcp_cancel_tournament(
    ctx: Context,
    tournament_id: int,
    reason_detail: str | None = None,
) -> dict:
    """Cancel a tournament (admin or owner only)."""
    from atp.dashboard.mcp.notifications import (
        resolve_user_from_ctx,
        with_service,
    )

    emit_tool_call(ctx, tool="mcp_cancel_tournament")
    user = await resolve_user_from_ctx(ctx)
    async with with_service(ctx, tournament_event_bus) as service:
        return await cancel_tournament(
            ctx=ctx,
            service=service,
            user=user,
            tournament_id=tournament_id,
            reason_detail=reason_detail,
        )
