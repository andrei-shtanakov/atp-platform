"""Wire tests for the official ``mcp`` SDK v2 client against /mcp/sse.

The /ui/about examples and the El Farol participant kit tell users to
connect with ``Client(sse_client(...), mode="legacy")``. These tests run
that exact shape against a live uvicorn + FastMCP server, so an SDK or
fastmcp upgrade that breaks the documented client path fails here.

Push notifications ride on the MCP logging capability, which the
2026-07-28 protocol revision deprecated: on connections negotiating it,
the server drops ``notifications/message`` sent after a tool call has
returned. ``mode="legacy"`` keeps the initialize handshake and with it
the pre-2026 delivery. The second test pins the default-mode behaviour,
because /ui/about documents it; if upstream changes it, revisit the page.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import httpx
import pytest
from mcp import Client
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, LoggingMessageNotificationParams

from tests.e2e.mcp_auth import mint_tournament_agent_token

pytestmark = [pytest.mark.anyio, pytest.mark.slow]

LEGACY_PROTOCOL = "2025-11-25"
MODERN_PROTOCOL = "2026-07-28"


def _payload(result: CallToolResult) -> dict[str, Any]:
    """Tool output as a dict: structured content first, text JSON second."""
    if isinstance(result.structured_content, dict):
        return result.structured_content
    for item in result.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    return {}


class _EventLog:
    """Collects ``(event, round_number)`` from server log notifications."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    async def handle_log(self, params: LoggingMessageNotificationParams) -> None:
        """``logging_callback`` for ``mcp.Client``."""
        if isinstance(params.data, dict):
            self.events.append(
                (str(params.data.get("event")), params.data.get("round_number"))
            )

    async def wait_for(
        self, event: str, round_number: int | None = None, timeout_s: float = 10.0
    ) -> bool:
        """Poll until ``event`` (for ``round_number``, if given) arrives."""

        def seen() -> bool:
            return any(
                name == event and (round_number is None or rnd == round_number)
                for name, rnd in self.events
            )

        with anyio.move_on_after(timeout_s):
            while not seen():
                await anyio.sleep(0.05)
            return True
        return False


async def _create_pd_tournament(base_url: str, admin_jwt: str, name: str) -> int:
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as rest:
        resp = await rest.post(
            "/api/v1/tournaments",
            headers={"Authorization": f"Bearer {admin_jwt}"},
            json={
                "name": name,
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 2,
                "round_deadline_s": 30,
            },
        )
    assert resp.status_code in (200, 201), resp.text
    return int(resp.json()["id"])


async def test_legacy_mode_client_plays_and_receives_notifications(
    tournament_uvicorn,
) -> None:
    """The documented v2 client shape joins, plays and gets push events."""
    base_url, admin_jwt, bob_jwt = tournament_uvicorn
    sse_url = f"{base_url}/mcp/sse"
    tournament_id = await _create_pd_tournament(base_url, admin_jwt, "sdk-v2-legacy")
    admin_token = await mint_tournament_agent_token(
        base_url, admin_jwt, agent_name="sdk-v2-admin"
    )
    bob_token = await mint_tournament_agent_token(
        base_url, bob_jwt, agent_name="sdk-v2-bob"
    )
    admin_log, bob_log = _EventLog(), _EventLog()

    async with (
        Client(
            sse_client(sse_url, headers={"Authorization": f"Bearer {admin_token}"}),
            mode="legacy",
            logging_callback=admin_log.handle_log,
        ) as admin,
        Client(
            sse_client(sse_url, headers={"Authorization": f"Bearer {bob_token}"}),
            mode="legacy",
            logging_callback=bob_log.handle_log,
        ) as bob,
    ):
        assert admin.protocol_version == LEGACY_PROTOCOL
        for client, name in ((admin, "admin-bot"), (bob, "bob-bot")):
            joined = _payload(
                await client.call_tool(
                    "join_tournament",
                    {"tournament_id": tournament_id, "agent_name": name},
                )
            )
            assert joined.get("joined") is True, joined
        assert await admin_log.wait_for("session_sync")

        state = _payload(
            await admin.call_tool("get_current_state", {"tournament_id": tournament_id})
        )
        assert state.get("game_type") == "prisoners_dilemma", state

        for client, choice in ((admin, "cooperate"), (bob, "defect")):
            await client.call_tool(
                "make_move",
                {"tournament_id": tournament_id, "action": {"choice": choice}},
            )
        # Round 1's round_started already fired when bob joined, so pin
        # round 2: it exists only once both moves resolved round 1, and is
        # delivered by the background forwarder after make_move returned —
        # exactly the path the 2026-07-28 protocol no longer carries.
        assert await admin_log.wait_for("round_started", 2), admin_log.events
        assert await bob_log.wait_for("round_started", 2), bob_log.events


async def test_default_mode_negotiates_modern_protocol_without_push(
    tournament_uvicorn,
) -> None:
    """Default ``mode="auto"`` lands on 2026-07-28: polling works, push does not."""
    base_url, admin_jwt, _bob_jwt = tournament_uvicorn
    sse_url = f"{base_url}/mcp/sse"
    tournament_id = await _create_pd_tournament(base_url, admin_jwt, "sdk-v2-auto")
    token = await mint_tournament_agent_token(
        base_url, admin_jwt, agent_name="sdk-v2-auto"
    )
    log = _EventLog()

    async with Client(
        sse_client(sse_url, headers={"Authorization": f"Bearer {token}"}),
        logging_callback=log.handle_log,
    ) as client:
        assert client.protocol_version == MODERN_PROTOCOL
        joined = _payload(
            await client.call_tool(
                "join_tournament",
                {"tournament_id": tournament_id, "agent_name": "auto-bot"},
            )
        )
        assert joined.get("joined") is True, joined
        # Request/response still works on the modern protocol. (Not
        # get_current_state: with 1 of 2 players joined it errors server-side.)
        details = _payload(
            await client.call_tool(
                "mcp_get_tournament", {"tournament_id": tournament_id}
            )
        )
        assert details["tournament"]["id"] == tournament_id, details
        assert not await log.wait_for("session_sync", timeout_s=1.0), log.events
