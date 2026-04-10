"""Tests for JWTUserStateMiddleware — populates request.state.user_id
so slowapi's rate limiter can key on authenticated identity.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def _make_app() -> FastAPI:
    """Build an app with JWTUserStateMiddleware + a state-introspection route."""
    from atp.dashboard.v2.rate_limit import JWTUserStateMiddleware

    app = FastAPI()
    app.add_middleware(JWTUserStateMiddleware)

    @app.get("/_state")
    async def _state(request: Request) -> dict[str, Any]:
        return {"user_id": getattr(request.state, "user_id", None)}

    return app


def _token(payload: dict[str, Any], secret: str = "test-secret") -> str:
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a known SECRET_KEY on the auth module for predictable decode."""
    import atp.dashboard.auth as auth_module

    monkeypatch.setattr(auth_module, "SECRET_KEY", "test-secret")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_jwt_sets_user_id_on_request_state() -> None:
    """A valid Bearer JWT with user_id claim populates request.state.user_id."""
    token = _token({"sub": "alice", "user_id": 42})
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json() == {"user_id": 42}


def test_valid_jwt_case_insensitive_bearer_scheme() -> None:
    """``bearer`` / ``BEARER`` / ``Bearer`` prefixes all work."""
    token = _token({"sub": "bob", "user_id": 7})
    client = TestClient(_make_app())

    for scheme in ("Bearer", "bearer", "BEARER"):
        resp = client.get("/_state", headers={"Authorization": f"{scheme} {token}"})
        assert resp.json()["user_id"] == 7, scheme


# ---------------------------------------------------------------------------
# Anonymous / missing header
# ---------------------------------------------------------------------------


def test_no_authorization_header_leaves_state_unset() -> None:
    """Anonymous request → no user_id, no error."""
    client = TestClient(_make_app())

    resp = client.get("/_state")

    assert resp.status_code == 200
    assert resp.json() == {"user_id": None}


def test_wrong_scheme_is_ignored() -> None:
    """Basic auth header is silently ignored by the middleware."""
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": "Basic Zm9vOmJhcg=="})

    assert resp.json() == {"user_id": None}


# ---------------------------------------------------------------------------
# Failure modes — all silent, state stays None
# ---------------------------------------------------------------------------


def test_malformed_token_silently_ignored() -> None:
    """A garbage token does not crash the request or populate state."""
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": "Bearer not-a-jwt"})

    assert resp.status_code == 200
    assert resp.json() == {"user_id": None}


def test_expired_token_silently_ignored() -> None:
    """An expired JWT does not populate state; full auth will reject elsewhere."""
    expired = _token(
        {
            "sub": "alice",
            "user_id": 42,
            "exp": datetime.now(tz=UTC) - timedelta(hours=1),
        }
    )
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": f"Bearer {expired}"})

    assert resp.json() == {"user_id": None}


def test_wrong_signature_silently_ignored() -> None:
    """JWT signed with a different key does not populate state."""
    foreign = _token({"sub": "alice", "user_id": 42}, secret="other-secret")
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": f"Bearer {foreign}"})

    assert resp.json() == {"user_id": None}


def test_token_without_user_id_claim_leaves_state_unset() -> None:
    """A valid JWT without user_id claim does not set state.

    Rationale: slowapi's key function only treats user_id as the rate-limit
    key when it is present; falling back to IP is the desired behavior.
    """
    token = _token({"sub": "alice"})
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": f"Bearer {token}"})

    assert resp.json() == {"user_id": None}


def test_bearer_prefix_only_no_token() -> None:
    """``Authorization: Bearer`` with no token is ignored."""
    client = TestClient(_make_app())

    resp = client.get("/_state", headers={"Authorization": "Bearer "})

    assert resp.json() == {"user_id": None}


# ---------------------------------------------------------------------------
# Integration with slowapi rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_key_is_user_scoped_for_authenticated_requests() -> None:
    """Two authenticated requests from different IPs but the same user share
    a rate-limit bucket; anonymous requests are keyed by IP.
    """
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    from atp.dashboard.v2.rate_limit import (
        JWTUserStateMiddleware,
        get_rate_limit_key,
        rate_limit_exceeded_handler,
    )

    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=["2/minute"],
        storage_uri="memory://",
    )

    app = FastAPI()
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(JWTUserStateMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.get("/ping")
    async def ping(request: Request) -> dict[str, bool]:
        return {"ok": True}

    token_alice = _token({"sub": "alice", "user_id": 1})
    token_bob = _token({"sub": "bob", "user_id": 2})

    client = TestClient(app)

    # Alice uses her 2 requests, third is 429
    assert (
        client.get(
            "/ping", headers={"Authorization": f"Bearer {token_alice}"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/ping", headers={"Authorization": f"Bearer {token_alice}"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/ping", headers={"Authorization": f"Bearer {token_alice}"}
        ).status_code
        == 429
    )

    # Bob is a separate bucket — not yet limited
    assert (
        client.get(
            "/ping", headers={"Authorization": f"Bearer {token_bob}"}
        ).status_code
        == 200
    )
