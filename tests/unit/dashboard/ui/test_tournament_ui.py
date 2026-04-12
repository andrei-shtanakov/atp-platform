"""Tests for tournament UI routes."""

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


@pytest.mark.anyio
async def test_tournament_list_returns_200(client: AsyncClient):
    resp = await client.get("/ui/tournaments")
    assert resp.status_code == 200
    assert "Tournaments" in resp.text


@pytest.mark.anyio
async def test_tournament_list_partial_returns_200(client: AsyncClient):
    resp = await client.get("/ui/tournaments?partial=1")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_tournament_detail_404_for_missing(client: AsyncClient):
    resp = await client.get("/ui/tournaments/99999")
    assert resp.status_code == 404
    assert "Not Found" in resp.text
