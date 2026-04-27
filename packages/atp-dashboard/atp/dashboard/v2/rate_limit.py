"""HTTP rate limiting for ATP Dashboard.

Uses slowapi (built on the limits library) for per-endpoint rate limiting.
Key function resolves user_id from JWT for authenticated requests,
falls back to client IP for anonymous requests.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import jwt
from cachetools import TTLCache
from fastapi import Request
from fastapi.responses import JSONResponse
from jwt.exceptions import InvalidTokenError
from slowapi import Limiter
from slowapi.util import get_remote_address

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


_API_TOKEN_PREFIXES = ("atp_u_", "atp_a_")


# LABS-TSA PR-3: lazy-fallback cache for tokens that were issued before the
# APIToken.agent_purpose column existed. Keyed by token_hash (cryptographic
# hash, safe to keep in RAM — no raw tokens stored here). Process-scoped,
# unbounded: each legacy token contributes one small string; once all
# legacy tokens are rotated, nothing new enters this dict.
_legacy_purpose_cache: dict[str, str] = {}


# MCP reliability plan Task 1 (2026-04-27): in-process LRU cache for the
# JWTUserStateMiddleware._resolve_api_token DB SELECT path. Without it,
# every MCP request — including SSE keepalives and rapid handshake
# bursts — costs one ``api_tokens`` SELECT plus a debounced UPDATE. With
# 30 s TTL, the same token used repeatedly skips the SELECT entirely.
#
# Keyed by token_hash (sha256 hex). Value = (user_id, agent_id,
# agent_purpose) — the triple needed to populate request.state. The
# cache stores only resolved successes; misses (revoked / expired /
# unknown tokens) are not cached so retries always re-check the DB.
#
# Bounded at 1024 entries to cap RAM under token churn (~200 B/entry
# worst case = ~200 KB). Single-worker assumption (see factory.py
# ``assert_single_worker``) keeps cache coherent without locks.
#
# Token-revocation propagation is bounded by the TTL: an admin who
# revokes a token via ``DELETE /api/v1/tokens`` may still see it
# resolve for up to 30 s. Acceptable pre-1.0; revisit if/when a
# customer requires sub-second revocation.
#
# ``last_used_at`` cadence note: the column is updated only on cache
# misses (the SELECT/UPDATE path). With a 30 s TTL and a 60 s update
# debounce, a continuously-used token still records ``last_used_at``
# every ~60 s. A bursty token may lag up to TTL+debounce ≈ 90 s
# behind real activity. ``last_used_at`` is an audit/cleanup field;
# the lag is acceptable. If sub-minute precision is ever needed,
# fire a debounced UPDATE on cache hit (no SELECT required since the
# UPDATE has its own WHERE-clause debounce).
_TOKEN_AUTH_CACHE_TTL_S = 30.0
_TOKEN_AUTH_CACHE_MAX = 1024
# Cached value: (user_id, agent_id, agent_purpose, expires_at). Storing
# ``expires_at`` lets cache hits enforce expiry without a DB roundtrip
# — otherwise a token that expires after caching could resolve for up
# to TTL seconds past its actual expiry.
_token_auth_cache: TTLCache[
    str, tuple[int, int | None, str | None, datetime | None]
] = TTLCache(maxsize=_TOKEN_AUTH_CACHE_MAX, ttl=_TOKEN_AUTH_CACHE_TTL_S)


class JWTUserStateMiddleware:
    """Pure ASGI middleware — populate scope.state.user_id from JWT / API token.

    Must not inherit from BaseHTTPMiddleware: this middleware also wraps
    the /mcp SSE mount, and BaseHTTPMiddleware buffers response bodies in
    a way that crashes long-lived streaming responses with
    ``Unexpected message: {type: http.response.start}`` (LABS-74).

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

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Starlette's Request.state reads/writes scope["state"] dict.
        state = scope.setdefault("state", {})
        state.setdefault("user_id", None)
        state.setdefault("agent_id", None)
        state.setdefault("token_type", None)
        # LABS-TSA PR-3: agent_purpose is only populated for agent-scoped
        # API tokens. NULL means "user-level token or anonymous" — used by
        # MCPAuthMiddleware / benchmark API gating to distinguish admins
        # and user-scoped tokens from agent-scoped ones.
        state.setdefault("agent_purpose", None)

        token = self._extract_token(scope)
        if token:
            if token.startswith(_API_TOKEN_PREFIXES):
                await self._resolve_api_token(state, token)
            else:
                self._resolve_jwt(state, token)

        await self.app(scope, receive, send)

    @staticmethod
    def _extract_token(scope: dict[str, Any]) -> str | None:
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        auth = headers.get("authorization", "")
        scheme, _, header_token = auth.partition(" ")
        if scheme.lower() == "bearer" and header_token.strip():
            return header_token.strip()

        # Fallback to atp_token cookie (browser sessions).
        cookie_header = headers.get("cookie", "")
        for part in cookie_header.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "atp_token" and value:
                return value
        return None

    @staticmethod
    def _resolve_jwt(state: dict[str, Any], token: str) -> None:
        """Populate state['user_id'] from a JWT token. Silently ignore errors."""
        from atp.dashboard import auth as auth_module

        try:
            payload = jwt.decode(
                token,
                auth_module.SECRET_KEY,
                algorithms=[auth_module.ALGORITHM],
            )
        except InvalidTokenError:
            return

        user_id = payload.get("user_id")
        if user_id is not None:
            state["user_id"] = user_id

    @staticmethod
    async def _resolve_api_token(state: dict[str, Any], token: str) -> None:
        """Populate request.state from an API token (atp_u_ / atp_a_ prefix).

        Looks up the token hash in the ``api_tokens`` table.  Updates
        ``last_used_at`` atomically with a WHERE-clause debounce to avoid
        races.  Silently skips on any error (DB not ready, token not found,
        token revoked/expired).

        Hot-path optimisation: ``_token_auth_cache`` (TTL 30 s) skips the
        SELECT for tokens used repeatedly within the window — see the
        cache definition above for revocation-lag tradeoffs.
        """
        from sqlalchemy import select, update

        from atp.dashboard.database import get_database
        from atp.dashboard.tokens import APIToken

        token_hash = hashlib.sha256(token.encode()).hexdigest()

        cached = _token_auth_cache.get(token_hash)
        if cached is not None:
            user_id, agent_id, agent_purpose, cached_expires_at = cached
            if cached_expires_at is not None and cached_expires_at < datetime.now():
                # Token expired since caching — evict and re-resolve via DB so
                # the SELECT/expiry block returns ``None`` properly. Keeping
                # the stale entry would let it resolve for up to TTL seconds
                # past actual expiry.
                _token_auth_cache.pop(token_hash, None)
            else:
                state["user_id"] = user_id
                state["agent_id"] = agent_id
                state["agent_purpose"] = agent_purpose
                state["token_type"] = "api"
                return

        try:
            db = get_database()
            async with db.session() as session:
                result = await session.execute(
                    select(APIToken).where(
                        APIToken.token_hash == token_hash,
                        APIToken.revoked_at.is_(None),
                    )
                )
                api_token = result.scalar_one_or_none()

                if api_token is None:
                    return

                # Check expiry
                if (
                    api_token.expires_at is not None
                    and api_token.expires_at < datetime.now()
                ):
                    return

                state["user_id"] = api_token.user_id
                state["agent_id"] = api_token.agent_id
                state["token_type"] = "api"

                # LABS-TSA PR-3: surface agent_purpose for downstream
                # MCP / benchmark-API gates. Fall back to a one-time
                # Agent.purpose lookup (cached in-process by token_hash)
                # for legacy tokens issued before the snapshot column
                # existed.
                if api_token.agent_id is not None:
                    purpose = api_token.agent_purpose
                    if purpose is None:
                        purpose = _legacy_purpose_cache.get(token_hash)
                        if purpose is None:
                            from atp.dashboard.models import Agent

                            agent = await session.get(Agent, api_token.agent_id)
                            if agent is not None:
                                purpose = agent.purpose
                                _legacy_purpose_cache[token_hash] = purpose
                    state["agent_purpose"] = purpose

                # Populate the hot-path cache with the resolved tuple.
                # Done before the UPDATE so a debounce-related exception
                # below does not leave the cache empty for the next call.
                # ``expires_at`` is included so a future cache hit can
                # enforce expiry locally without a DB roundtrip.
                _token_auth_cache[token_hash] = (
                    api_token.user_id,
                    api_token.agent_id,
                    state.get("agent_purpose"),
                    api_token.expires_at,
                )

                # Debounced last_used_at: skip if updated within last 60s
                now = datetime.now()
                debounce = timedelta(seconds=60)
                await session.execute(
                    update(APIToken)
                    .where(
                        APIToken.id == api_token.id,
                        (
                            APIToken.last_used_at.is_(None)
                            | (APIToken.last_used_at < now - debounce)
                        ),
                    )
                    .values(last_used_at=now)
                )
        except Exception:
            # DB not initialised yet or any other transient error — skip.
            logger.debug(
                "API token resolution skipped (DB error or token not found)",
                exc_info=True,
            )


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
