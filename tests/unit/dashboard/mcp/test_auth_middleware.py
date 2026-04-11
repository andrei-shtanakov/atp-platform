"""Tests for MCPAuthMiddleware: rejects requests without request.state.user_id."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from atp.dashboard.mcp.auth import MCPAuthMiddleware


def _make_app(set_user_id: int | None) -> FastAPI:
    """Build a tiny FastAPI test app.

    Starlette applies HTTP middlewares in LIFO: the LAST-added wraps
    incoming requests FIRST. We want ``_set_state`` to run BEFORE
    ``MCPAuthMiddleware`` sees the request, so we add
    ``MCPAuthMiddleware`` first and ``_set_state`` second.
    """
    app = FastAPI()
    app.add_middleware(MCPAuthMiddleware)

    @app.middleware("http")
    async def _set_state(request, call_next):  # type: ignore[no-untyped-def]
        if set_user_id is not None:
            request.state.user_id = set_user_id
        return await call_next(request)

    @app.get("/_test")
    async def _test() -> dict:
        return {"ok": True}

    return app


def test_mcp_auth_rejects_when_no_user_id() -> None:
    client = TestClient(_make_app(set_user_id=None))
    response = client.get("/_test")
    assert response.status_code == 401


def test_mcp_auth_accepts_when_user_id_present() -> None:
    client = TestClient(_make_app(set_user_id=42))
    response = client.get("/_test")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
