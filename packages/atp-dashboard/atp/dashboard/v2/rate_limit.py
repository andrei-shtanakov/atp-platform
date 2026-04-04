"""HTTP rate limiting for ATP Dashboard.

Uses slowapi (built on the limits library) for per-endpoint rate limiting.
Key function resolves user_id from JWT for authenticated requests,
falls back to client IP for anonymous requests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

if TYPE_CHECKING:
    from atp.dashboard.v2.config import DashboardConfig

logger = logging.getLogger("atp.dashboard.rate_limit")

# Module-level limiter — populated by create_limiter(), imported by route files
limiter: Limiter | None = None


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


def create_limiter(config: DashboardConfig) -> Limiter:
    """Create a slowapi Limiter instance from config."""
    global limiter  # noqa: PLW0603
    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=[config.rate_limit_default],
        storage_uri=config.rate_limit_storage,
        enabled=config.rate_limit_enabled,
    )
    return limiter


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Custom 429 response with JSON body and Retry-After header."""
    retry_after = getattr(exc, "retry_after", 60)

    response = JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": retry_after,
        },
    )
    response.headers["Retry-After"] = str(retry_after)
    return response
