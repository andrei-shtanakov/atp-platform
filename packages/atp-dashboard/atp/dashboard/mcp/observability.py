"""Structured log events for the MCP handshake and tool-call lifecycle.

This module is the single source of truth for **what** events are emitted
and **what fields** they carry. Field names are part of the public
schema consumed downstream by Prometheus / OpenTelemetry mappers — keep
them stable. The runbook at ``docs/runbooks/mcp-observability.md``
describes each event in operator-friendly terms.

Cold-start incidents like tournament 30 (2026-04-27) lacked any
structured trace through the SSE handshake / auth / tool dispatch
chain, so debugging was reduced to grepping ad-hoc ``logger.info``
lines. These events give on-call a per-request handle (``request_id``)
plus a per-SSE-session handle (``session_id``) to correlate timing
across the lifecycle.

This is Task 4 of ``docs/superpowers/plans/2026-04-27-mcp-server-reliability.md``.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger("atp.mcp.observability")

# ---------------------------------------------------------------------------
# Event names — public schema
# ---------------------------------------------------------------------------

#: Emitted at the entry of ``MCPAuthMiddleware`` for every HTTP request
#: hitting ``/mcp/*``. Fields: ``request_id`` (str), ``client_ip``
#: (str | None), ``user_agent`` (str | None), ``path`` (str).
MCP_HANDSHAKE_STARTED = "mcp_handshake_started"

#: Emitted on auth-pass exit from ``MCPAuthMiddleware``. Fields:
#: ``request_id`` (str), ``user_id`` (int), ``agent_id`` (int | None),
#: ``agent_purpose`` (str | None), ``duration_ms`` (float).
MCP_HANDSHAKE_AUTHORIZED = "mcp_handshake_authorized"

#: Emitted on auth-fail exit from ``MCPAuthMiddleware``. Fields:
#: ``request_id`` (str), ``reason`` (str — one of
#: ``"unauthenticated"`` / ``"user_level_token"`` / ``"benchmark_token"``),
#: ``status`` (int — 401 or 403), ``duration_ms`` (float).
MCP_HANDSHAKE_REJECTED = "mcp_handshake_rejected"

#: Emitted on the first tool invocation per FastMCP session, gated by
#: the ``_first_tool_call_seen`` cache below. Subsequent tool calls
#: from the same session do not re-emit. Fields: ``session_id`` (str),
#: ``request_id`` (str), ``tool`` (str — e.g. ``"join_tournament"``).
#: This is the metric that pinpoints how long after a successful
#: handshake a client actually starts working — the cold-start race
#: lives in this gap.
MCP_FIRST_TOOL_CALL = "mcp_first_tool_call"


# ---------------------------------------------------------------------------
# In-process state for de-duping ``first_tool_call`` per session
# ---------------------------------------------------------------------------

# SSE sessions are long-lived but bounded. 1 h TTL covers any plausible
# tournament session; 4096-entry ceiling caps RAM under bursty churn.
# Single-worker assumption (factory.py ``assert_single_worker``) keeps
# this consistent without locking.
_FIRST_CALL_TTL_S = 3600.0
_FIRST_CALL_MAX = 4096
_first_tool_call_seen: TTLCache[str, bool] = TTLCache(
    maxsize=_FIRST_CALL_MAX, ttl=_FIRST_CALL_TTL_S
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def new_request_id() -> str:
    """Return a 16-char uuid4 slice: short enough for log grepping,
    wide enough (64 bits) to avoid collisions for any single session."""
    return uuid.uuid4().hex[:16]


def now_monotonic_ms() -> float:
    """Monotonic-clock millisecond reading, suitable for measuring
    intra-request durations. Not wall time — do not log directly."""
    return time.monotonic() * 1000.0


def emit_event(event: str, *, request_id: str, **fields: Any) -> None:
    """Log a structured event. Field naming is part of the public schema."""
    logger.info(
        event,
        extra={"event": event, "request_id": request_id, **fields},
    )


def maybe_emit_first_tool_call(*, session_id: str, request_id: str, tool: str) -> bool:
    """Emit ``MCP_FIRST_TOOL_CALL`` exactly once per ``session_id``.

    Returns ``True`` if the event was emitted (this was the first tool
    call for this session), ``False`` otherwise. Callers do not need to
    inspect the return value — it is exposed for tests that want to
    assert dedup behaviour without scraping log records.
    """
    if session_id in _first_tool_call_seen:
        return False
    _first_tool_call_seen[session_id] = True
    emit_event(
        MCP_FIRST_TOOL_CALL,
        request_id=request_id,
        session_id=session_id,
        tool=tool,
    )
    return True


def reset_state() -> None:
    """Clear the per-session dedup cache. For test isolation only —
    production code must never call this."""
    _first_tool_call_seen.clear()


def emit_tool_call(ctx: Any, *, tool: str) -> None:
    """Helper for ``@mcp_server.tool()`` bodies — gates the
    per-session ``MCP_FIRST_TOOL_CALL`` event and pulls correlation
    ids out of the FastMCP context plus the ASGI scope.

    If ``ctx.session_id`` is missing, the event is **skipped**: the
    only stable per-session key on FastMCP's context is
    ``session_id`` itself. Falling back to ``id(ctx)`` would defeat
    the dedup contract because ``id()`` is not stable across calls
    (FastMCP may construct a fresh ``Context`` per tool dispatch).
    A skipped emit is preferable to a noisy one — the cold-start
    metrics are diagnostic, not load-bearing.
    """
    raw_session = getattr(ctx, "session_id", None)
    if not raw_session:
        return
    session_id = str(raw_session)

    request_id = "no-request-id"
    try:
        from fastmcp.server.dependencies import get_http_request

        request = get_http_request()
        resolved = getattr(request.state, "mcp_request_id", None)
        if isinstance(resolved, str) and resolved:
            request_id = resolved
    except Exception:
        # ``get_http_request`` raises outside an HTTP-bound MCP call —
        # nothing we can do, keep the placeholder so the event still
        # fires and operators can see the gap in tooling.
        pass

    maybe_emit_first_tool_call(session_id=session_id, request_id=request_id, tool=tool)
