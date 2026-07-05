"""Agent-scoped token minting for e2e MCP auth.

Since LABS-TSA PR-3 (#78) the ``/mcp`` mount only admits agent-scoped
API tokens (``atp_a_*``) whose ``agent_purpose`` is ``"tournament"`` —
plain user JWTs are rejected with 403 by ``MCPAuthMiddleware``. E2e
fixtures and bots must therefore register a tournament agent and mint
a real token over the HTTP API (the same path production participants
use) before opening an SSE connection.
"""

from __future__ import annotations

import httpx


async def mint_tournament_agent_token(
    base_url: str, owner_jwt: str, *, agent_name: str
) -> str:
    """Register a tournament agent and mint an agent-scoped token.

    Returns the raw ``atp_a_*`` token accepted by ``MCPAuthMiddleware``.
    The owner must already exist in the database; ``owner_jwt`` is a JWT
    for that user (used for the two REST calls only).
    """
    headers = {"Authorization": f"Bearer {owner_jwt}"}
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        resp = await client.post(
            "/api/v1/agents",
            headers=headers,
            json={
                "name": agent_name,
                "agent_type": "mcp",
                "purpose": "tournament",
            },
        )
        assert resp.status_code == 201, resp.text
        agent_id = resp.json()["id"]

        resp = await client.post(
            "/api/v1/tokens",
            headers=headers,
            json={"agent_id": agent_id, "name": f"{agent_name}-token"},
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["token"]
