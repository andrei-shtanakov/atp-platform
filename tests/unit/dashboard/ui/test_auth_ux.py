"""Tests for auth UX: cookie fallback, login/logout."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def client():
    app = create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_token():
    """Create a valid JWT for testing."""
    from atp.dashboard.auth import create_access_token

    return create_access_token(data={"sub": "testuser", "user_id": 999})


@pytest.mark.anyio
async def test_cookie_auth_sets_user_id_on_request_state(
    client: AsyncClient, auth_token: str
):
    """Middleware should read atp_token cookie and set request.state.user_id."""
    resp = await client.get(
        "/ui/tournaments",
        cookies={"atp_token": auth_token},
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_logout_clears_cookie_and_redirects(client: AsyncClient, auth_token: str):
    """POST /ui/logout clears the atp_token cookie and redirects to the
    login page (commit b157996 changed the target from /ui/ to /ui/login
    so the user lands on a page that doesn't immediately re-auth-redirect)."""
    resp = await client.post(
        "/ui/logout",
        cookies={"atp_token": auth_token},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/login"
    set_cookie = resp.headers.get("set-cookie", "")
    assert "atp_token" in set_cookie
