"""HTTP rate limiting for ATP Dashboard.

Uses slowapi (built on the limits library) for per-endpoint rate limiting.
Key function resolves user_id from JWT for authenticated requests,
falls back to client IP for anonymous requests.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from jwt.exceptions import InvalidTokenError
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

if TYPE_CHECKING:
    from atp.dashboard.v2.config import DashboardConfig

logger = logging.getLogger("atp.dashboard.rate_limit")


def get_rate_limit_key(request: Request) -> str:
    """Extract rate limit key from request.

    Uses user_id from JWT if available, otherwise client IP.
    Respects X-Forwarded-For header for proxy deployments.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return f"user:{user_id}"

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = get_remote_address(request)
    return f"ip:{ip}"


# Module-level limiter — reconfigured by create_limiter() at app startup.
# Starts disabled so decorators can be applied at import time.
limiter: Limiter = Limiter(
    key_func=get_rate_limit_key,
    enabled=False,
)


def create_limiter(config: DashboardConfig) -> Limiter:
    """Configure the module-level limiter from dashboard config.

    Mutates the existing ``limiter`` instance in-place so that
    decorators already bound at import time pick up the new settings.
    """
    from limits.storage import storage_from_string
    from slowapi.wrappers import LimitGroup

    limiter.enabled = config.rate_limit_enabled  # type: ignore[attr-defined]
    # _default_limits must be a list of LimitGroup instances — slowapi's
    # middleware does ``itertools.chain(*self._default_limits)`` which would
    # otherwise iterate a raw string character-by-character and crash with an
    # AttributeError deep inside __evaluate_limits, then get re-wrapped by
    # slowapi's own exception handler (which assumes RateLimitExceeded and
    # reads ``.detail``), surfacing as a cryptic 500 on any non-decorated
    # route. Mirror Limiter.__init__'s construction.
    limiter._default_limits = [  # type: ignore[attr-defined]
        LimitGroup(
            config.rate_limit_default,
            limiter._key_func,  # type: ignore[attr-defined]
            None,
            False,
            None,
            None,
            None,
            1,
            False,
        )
    ]
    if config.rate_limit_storage:
        limiter._storage = storage_from_string(  # type: ignore[attr-defined]
            config.rate_limit_storage
        )
    return limiter


class JWTUserStateMiddleware(BaseHTTPMiddleware):
    """Best-effort: populate ``request.state.user_id`` from a Bearer JWT.

    Runs BEFORE ``SlowAPIMiddleware`` so the rate-limit key function can
    see the authenticated identity and key per-user instead of per-IP
    (which collapses buckets when multiple benchmark participants share
    a NAT).

    Auth is NOT enforced here — invalid / expired / missing tokens are
    silently ignored. Real authentication is still performed by
    ``get_current_user`` on protected routes; this middleware only
    enriches the request with an advisory user_id when trivially
    available.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        token: str | None = None

        # 1. Try Authorization: Bearer header
        header = request.headers.get("authorization", "")
        scheme, _, header_token = header.partition(" ")
        if scheme.lower() == "bearer" and header_token.strip():
            token = header_token.strip()

        # 2. Fallback to atp_token cookie (browser sessions)
        if token is None:
            token = request.cookies.get("atp_token")

        if token:
            # Late import so test monkeypatching of SECRET_KEY takes effect
            # and to avoid a circular import at module load.
            from atp.dashboard import auth as auth_module

            try:
                payload = jwt.decode(
                    token,
                    auth_module.SECRET_KEY,
                    algorithms=[auth_module.ALGORITHM],
                )
            except InvalidTokenError:
                pass
            else:
                user_id = payload.get("user_id")
                if user_id is not None:
                    request.state.user_id = user_id
        return await call_next(request)


async def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Custom 429 response with JSON body and Retry-After header."""
    retry_after = getattr(exc, "retry_after", 60)
    detail = getattr(exc, "detail", str(exc))

    response = JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {detail}",
            "retry_after": retry_after,
        },
    )
    response.headers["Retry-After"] = str(retry_after)
    return response
