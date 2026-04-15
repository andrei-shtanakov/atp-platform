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
"""

from __future__ import annotations

import json
from typing import Any


class MCPAuthMiddleware:
    """Pure ASGI middleware that rejects MCP requests without user_id.

    Must be a plain ASGI callable (not BaseHTTPMiddleware) so it doesn't
    buffer the SSE response stream. On 401 we write a JSON response
    directly; on success we pass through to the inner app unchanged.
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

        # JWTUserStateMiddleware upstream writes user_id onto the Starlette
        # request state dict inside scope. Match how Request.state reads it.
        state = scope.setdefault("state", {})
        user_id = state.get("user_id")
        if user_id is None:
            body = json.dumps(
                {"error": "unauthorized", "detail": "Bearer JWT required"}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
