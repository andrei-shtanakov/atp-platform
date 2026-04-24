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
        user_id = state.get("user_id")
        if user_id is None:
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
