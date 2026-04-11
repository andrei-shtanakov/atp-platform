"""End-to-end PD tournament test (acceptance criterion for v1 slice)."""

from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_e2e_mcp_server_boots(e2e_mcp_server) -> None:
    base_url, port = e2e_mcp_server
    assert base_url.startswith("http://127.0.0.1:")
    assert isinstance(port, int) and port > 0


@pytest.mark.anyio
async def test_seeded_users_have_jwts(e2e_mcp_server, mcp_seeded_users) -> None:
    assert "alice" in mcp_seeded_users
    assert mcp_seeded_users["alice"]["jwt"]
    assert mcp_seeded_users["bob"]["jwt"]
    assert mcp_seeded_users["admin"]["jwt"]
