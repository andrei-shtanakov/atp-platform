"""Authentication gate for the MCP SSE endpoint.

Sits in front of the FastMCP mount and rejects any request that does
not have ``request.state.user_id`` populated by the outer
``JWTUserStateMiddleware``. The actual JWT decoding is done upstream;
this module only enforces presence.

Implemented as a pure ASGI middleware (not Starlette's BaseHTTPMiddleware)
because the MCP mount serves Server-Sent Events. BaseHTTPMiddleware
buffers/wraps the response body and breaks long-lived SSE streams with
``assert message["type"] == "http.response.body"`` once the stream sends
non-body messages. LABS-74 was the symptom of this incompatibility.

LABS-TSA PR-3 extends this gate with purpose-based authorisation: /mcp
is only open to tournament agents, not benchmark agents, user-level
tokens, or admin sessions.
"""

from __future__ import annotations

import json
from typing import Any

from atp.dashboard.mcp.observability import (
    MCP_HANDSHAKE_AUTHORIZED,
    MCP_HANDSHAKE_REJECTED,
    MCP_HANDSHAKE_STARTED,
    emit_event,
    new_request_id,
    now_monotonic_ms,
)


class MCPAuthMiddleware:
    """Pure ASGI middleware that rejects unauthorised MCP requests.

    Must be a plain ASGI callable (not BaseHTTPMiddleware) so it doesn't
    buffer the SSE response stream. On rejection we write a JSON response
    directly; on success we pass through to the inner app unchanged.

    Rejection matrix:
    - no ``user_id`` → 401 ("Bearer JWT required")
    - ``user_id`` set but no ``agent_purpose`` → 403 (user-level / admin)
    - ``agent_purpose != "tournament"`` → 403 (benchmark agent)
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        # Let non-HTTP traffic (lifespan, websockets) pass through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # JWTUserStateMiddleware upstream writes user_id / agent_purpose
        # onto the Starlette request state dict inside scope.
        state = scope.setdefault("state", {})

        # MCP reliability plan Task 4: per-request correlation id +
        # structured observability events. ``mcp_request_id`` is read
        # downstream by tool wrappers (via ``get_http_request().state``)
        # to thread the same id through the ``mcp_first_tool_call`` event.
        request_id = new_request_id()
        state["mcp_request_id"] = request_id
        started_at_ms = now_monotonic_ms()
        client_ip, user_agent = _extract_client_metadata(scope)
        emit_event(
            MCP_HANDSHAKE_STARTED,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
            path=scope.get("path"),
        )

        user_id = state.get("user_id")
        if user_id is None:
            emit_event(
                MCP_HANDSHAKE_REJECTED,
                request_id=request_id,
                reason="unauthenticated",
                status=401,
                duration_ms=now_monotonic_ms() - started_at_ms,
            )
            await self._send_json(
                send,
                status=401,
                body={"error": "unauthorized", "detail": "Bearer JWT required"},
            )
            return

        agent_purpose = state.get("agent_purpose")
        if agent_purpose is None:
            # User-level token or admin session — MCP is strictly for
            # agent-scoped tokens from registered tournament agents.
            emit_event(
                MCP_HANDSHAKE_REJECTED,
                request_id=request_id,
                reason="user_level_token",
                status=403,
                duration_ms=now_monotonic_ms() - started_at_ms,
            )
            await self._send_json(
                send,
                status=403,
                body={
                    "error": "forbidden",
                    "detail": "MCP requires an agent-scoped token (atp_a_*)",
                },
            )
            return

        if agent_purpose != "tournament":
            emit_event(
                MCP_HANDSHAKE_REJECTED,
                request_id=request_id,
                reason="benchmark_token",
                status=403,
                duration_ms=now_monotonic_ms() - started_at_ms,
            )
            await self._send_json(
                send,
                status=403,
                body={
                    "error": "forbidden",
                    "detail": (
                        "MCP is tournament-agents only; "
                        "this token belongs to a benchmark agent"
                    ),
                },
            )
            return

        emit_event(
            MCP_HANDSHAKE_AUTHORIZED,
            request_id=request_id,
            user_id=user_id,
            agent_id=state.get("agent_id"),
            agent_purpose=agent_purpose,
            duration_ms=now_monotonic_ms() - started_at_ms,
        )
        await self.app(scope, receive, send)

    @staticmethod
    async def _send_json(send: Any, *, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": encoded})


def _extract_client_metadata(scope: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull ``client_ip`` and ``user_agent`` from the ASGI scope for
    inclusion in observability events. Both are best-effort: ``None``
    on missing/malformed data rather than raising."""
    headers_raw = scope.get("headers") or []
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in headers_raw}
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        client_ip: str | None = forwarded.split(",")[0].strip() or None
    else:
        client = scope.get("client")
        client_ip = client[0] if isinstance(client, (list, tuple)) and client else None
    user_agent = headers.get("user-agent") or None
    return client_ip, user_agent
