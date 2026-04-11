"""Authentication gate for the MCP SSE endpoint.

Sits in front of the FastMCP mount and rejects any request that does
not have ``request.state.user_id`` populated by the outer
``JWTUserStateMiddleware``. The actual JWT decoding is done upstream;
this module only enforces presence.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Reject MCP requests without an authenticated user_id."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        user_id = getattr(request.state, "user_id", None)
        if user_id is None:
            return JSONResponse(
                {"error": "unauthorized", "detail": "Bearer JWT required"},
                status_code=401,
            )
        return await call_next(request)
