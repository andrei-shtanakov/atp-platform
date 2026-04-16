"""Anti-drift tests for the /ui/about quickstart.

``/ui/about`` is the first external onboarding contract of the
platform. If the page's instructions silently drift away from the
actual runtime (routing refactor, renamed MCP tool, schema change),
a first-time visitor gets a 404 / ValidationError and leaves. These
tests are the safety net.

Two layers:

1. **test_about_links_reachable** — cheap, runs in every CI pass.
   Walks every internal UI URL the About page references; asserts
   each one returns a usable response (200 for public, 302 redirect
   for authed pages). Also asserts specific tool/endpoint names
   appear in the About HTML, catching "deleted section" regressions
   without pretending to verify semantic correctness.

2. **test_about_mcp_walkthrough** — full MCP walkthrough using the
   real FastMCP server + MCPAdapter + a live tournament. Covers the
   happy path a new user would follow end-to-end. Reuses the
   ``tournament_uvicorn`` fixture from the SC-1 e2e suite. Marked
   with the same ``asyncio`` backend — if the shared fixture hits
   the LABS-20/74 SSE handshake flake, this test sees it too; that
   is acceptable because we want the real environment, and the
   fixture already has a retry probe baked in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from atp.adapters.mcp import MCPAdapter, MCPAdapterConfig

# tournament_uvicorn + _mint_jwt come from the sibling conftest.py,
# which re-exports them from the SC-1 tournament e2e module.

pytestmark = pytest.mark.anyio


_ABOUT_TEMPLATE = (
    Path(__file__).parents[3]
    / "packages/atp-dashboard/atp/dashboard/v2/templates/ui/about.html"
)


# Tools the About page currently lists under "MCP tools reference".
# Adding/removing a tool without updating this set is a deliberate
# user-facing docs change and must be explicit — not silent.
_EXPECTED_MCP_TOOLS = frozenset(
    {
        "mcp_list_tournaments",
        "mcp_get_tournament",
        "join_tournament",
        "get_current_state",
        "make_move",
        "mcp_get_history",
        "mcp_leave_tournament",
    }
)


# Internal UI links the About page currently directs new users to.
# Each must respond (authed pages 302 to /ui/login, public 200).
_EXPECTED_UI_LINKS = frozenset({"/ui/login", "/ui/agents", "/ui/tokens"})


# Games the About page lists as available (not greyed-out "coming soon").
# Adding one requires matching server-side dispatcher support first.
_EXPECTED_AVAILABLE_GAMES = frozenset({"prisoners_dilemma", "el_farol", "stag_hunt"})


def _extract_internal_links(html: str) -> set[str]:
    """Pull every internal href starting with /ui/ from the rendered page."""
    return set(re.findall(r'href="(/ui/[^"#?]+)"', html))


async def test_about_links_reachable(tournament_uvicorn) -> None:
    """About page renders, references all expected tools/links, and the
    expected internal UI links it includes are reachable (2xx or 302).
    """
    base_url, admin_jwt, _bob_jwt = tournament_uvicorn

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        resp = await client.get("/ui/about")
        assert resp.status_code == 200, resp.text
        body = resp.text

        # Anti-regression on section names + tool catalog.
        assert "MCP" in body, "About page no longer mentions MCP"
        for tool_name in _EXPECTED_MCP_TOOLS:
            assert tool_name in body, (
                f"About page silently dropped tool {tool_name!r}; "
                "update both the page AND _EXPECTED_MCP_TOOLS in this test."
            )
        for link in _EXPECTED_UI_LINKS:
            assert f'href="{link}"' in body, (
                f"About quickstart no longer links to {link}; "
                "if the page was restructured, update _EXPECTED_UI_LINKS."
            )

        # MCP endpoint path referenced in every code sample.
        assert "/mcp/sse" in body

        # Every "available" game must be listed and NOT grey-boxed.
        # The template marks disabled cards with `coming soon`, so we
        # look for the game_id appearing outside that grey styling.
        for game in _EXPECTED_AVAILABLE_GAMES:
            assert game in body, f"About no longer mentions game {game!r}"
            # Crude but effective: in the current template an available
            # card is the only place the game_id appears without a
            # "coming soon" span following it in the same <div>.
            # Checking the presence of the game_id in a .game-card that
            # is NOT styled as opacity:0.55 is enough to catch accidental
            # grey-outs.
            card_pattern = re.compile(
                r'<div class="game-card"(?![^>]*opacity:0\.55)[^>]*>'
                r'[^<]*<div class="game-id"[^>]*>' + re.escape(game),
                re.DOTALL,
            )
            assert card_pattern.search(body), (
                f"About lists {game!r} as available but the card is "
                "styled as coming-soon (opacity:0.55). Either update "
                "the template, or update _EXPECTED_AVAILABLE_GAMES."
            )

        # Internal links from the rendered page all respond.
        found_links = _extract_internal_links(body)
        # We only probe the ones we expect to exist in all environments;
        # extra links (like optional /ui/something-in-progress) don't fail
        # this test.
        for link in _EXPECTED_UI_LINKS & found_links:
            # cookie/auth-protected pages redirect to /ui/login (302); the
            # public ones return 200. Either is fine; a 404 is not.
            probe = await client.get(link, follow_redirects=False)
            assert probe.status_code in (200, 302, 303), (
                f"About links to {link} but GET returns {probe.status_code}"
            )


def _unwrap(raw: dict) -> dict:
    """Mirror of the smoke bot's structuredContent unwrapper."""
    if isinstance(raw, dict) and "structuredContent" in raw:
        sc = raw["structuredContent"]
        if isinstance(sc, dict):
            return sc
    if isinstance(raw, dict) and "content" in raw:
        content = raw["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:
                        parsed = json.loads(item["text"])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
    return raw if isinstance(raw, dict) else {}


async def test_about_mcp_walkthrough(tournament_uvicorn) -> None:
    """Execute the About quickstart: MCP connection → list tools → create
    tournament (REST) → two bots join and play one PD round → round
    resolves.

    Asserts that all tool names the About promises are actually
    registered on the live server, their call signatures match the
    About-documented shape, and a first-round PD action with
    ``{"choice": "cooperate"}`` is accepted by the server (the exact
    payload the About page shows in its examples).
    """
    base_url, admin_jwt, bob_jwt = tournament_uvicorn
    sse_url = f"{base_url}/mcp/sse"

    # Admin creates a 1-round PD tournament via the REST API the
    # About page implicitly uses (it describes creating via UI, but
    # REST is the same endpoint the UI form posts to).
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as rest:
        resp = await rest.post(
            "/api/v1/tournaments",
            headers={"Authorization": f"Bearer {admin_jwt}"},
            json={
                "name": "about-quickstart-walkthrough",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 1,
                "round_deadline_s": 30,
            },
        )
        assert resp.status_code in (200, 201), resp.text
        tournament_id = int(resp.json()["id"])

    async def _mcp_adapter(jwt_token: str) -> MCPAdapter:
        cfg = MCPAdapterConfig(
            transport="sse",
            url=sse_url,
            headers={"Authorization": f"Bearer {jwt_token}"},
            verify_ssl=False,
            startup_timeout=15.0,
            timeout_seconds=15.0,
        )
        adapter = MCPAdapter(cfg)
        await adapter.initialize()
        return adapter

    admin_adapter = await _mcp_adapter(admin_jwt)
    bob_adapter = await _mcp_adapter(bob_jwt)
    try:
        # Every tool the About page promises must actually be
        # registered by the FastMCP server. If a tool is renamed
        # server-side but the About listing isn't touched, THIS
        # test fails — which is the entire point.
        server_tools = set(admin_adapter.tools.keys())
        missing = _EXPECTED_MCP_TOOLS - server_tools
        assert not missing, (
            f"About lists MCP tools not registered on the server: {missing}. "
            f"Live tools: {sorted(server_tools)}"
        )

        # Each participant joins via the documented tool.
        admin_join = _unwrap(
            await admin_adapter.call_tool(
                "join_tournament",
                {"tournament_id": tournament_id, "agent_name": "admin-bot"},
            )
        )
        assert admin_join.get("joined") is True
        bob_join = _unwrap(
            await bob_adapter.call_tool(
                "join_tournament",
                {"tournament_id": tournament_id, "agent_name": "bob-bot"},
            )
        )
        assert bob_join.get("joined") is True

        # get_current_state returns the PD shape the About documents.
        state = _unwrap(
            await admin_adapter.call_tool(
                "get_current_state", {"tournament_id": tournament_id}
            )
        )
        assert state.get("game_type") == "prisoners_dilemma"
        assert "your_turn" in state
        action_schema = state.get("action_schema") or {}
        # About says action is ``{"choice": "cooperate" | "defect"}``.
        # We assert the server agrees by listing both values in its
        # schema echo — not a full schema diff, just a shape probe.
        assert "cooperate" in json.dumps(action_schema)
        assert "defect" in json.dumps(action_schema)

        # make_move with the exact payload the About example uses.
        admin_move = _unwrap(
            await admin_adapter.call_tool(
                "make_move",
                {
                    "tournament_id": tournament_id,
                    "action": {"choice": "cooperate"},
                },
            )
        )
        assert admin_move.get("status") in ("waiting", "round_resolved")

        bob_move = _unwrap(
            await bob_adapter.call_tool(
                "make_move",
                {
                    "tournament_id": tournament_id,
                    "action": {"choice": "defect"},
                },
            )
        )
        # Bob is second to submit for a 1-round tournament → resolve.
        assert bob_move.get("status") == "round_resolved", bob_move

    finally:
        for a in (admin_adapter, bob_adapter):
            try:
                await a.disconnect()
            except Exception:
                pass
