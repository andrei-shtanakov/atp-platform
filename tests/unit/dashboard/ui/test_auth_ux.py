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
