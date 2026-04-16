"""El Farol random-strategy MCP bot for production smoke.

Connects to an ATP dashboard MCP endpoint with an `atp_a_...` agent-scoped
token, joins a pre-created tournament, and plays random valid slot
selections until the tournament ends.

Intentionally simple: no LLM, no reconnect, no retry. If something goes
wrong, the bot raises and the orchestrator sees it. This is a correctness
probe for the server, not a quality demo for strategies.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from atp.adapters.mcp import MCPAdapter, MCPAdapterConfig

logger = logging.getLogger(__name__)

# Mirrors _EL_FAROL_V1_NUM_SLOTS in packages/atp-dashboard/atp/dashboard
# /tournament/service.py. Kept local to avoid a runtime dependency on
# the server-side dashboard package from bot processes.
DEFAULT_NUM_SLOTS = 16
DEFAULT_MAX_SLOTS_PER_DAY = 8


async def run_bot(
    *,
    server_url: str,
    token: str,
    tournament_id: int,
    agent_name: str,
    rounds: int,
    seed: int,
    verify_ssl: bool = True,
    per_round_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Play one tournament to completion with a random strategy.

    Args:
        server_url: Base URL of the ATP dashboard (e.g.
            ``https://atp.pr0sto.space``). The MCP SSE endpoint is
            derived as ``{server_url}/mcp/sse``.
        token: ``atp_a_...`` agent-scoped API token authenticating this
            bot as a distinct participant.
        tournament_id: Tournament the orchestrator already created.
        agent_name: Display name to register under for this tournament.
        rounds: Total rounds the tournament was created with; used only
            for the overall time budget (``rounds * per_round_timeout_s
            + 60s`` grace).
        seed: Per-bot RNG seed so multiple bots in one process don't make
            identical picks on identical state.
        verify_ssl: Pass ``False`` only for local dev against self-signed
            certificates.
        per_round_timeout_s: Hard ceiling on any single make_move call,
            also used to compute the overall bot timeout.

    Returns:
        Summary dict with keys ``agent_name``, ``final_score``,
        ``rounds_played``, ``final_status``. Raised exceptions propagate
        up to the orchestrator and should be treated as failures.
    """
    rng = random.Random(seed)
    config = MCPAdapterConfig(
        transport="sse",
        url=f"{server_url.rstrip('/')}/mcp/sse",
        headers={"Authorization": f"Bearer {token}"},
        verify_ssl=verify_ssl,
        startup_timeout=30.0,
        timeout_seconds=per_round_timeout_s,
    )
    adapter = MCPAdapter(config)

    rounds_played = 0
    final_score: float | None = None
    final_status = "unknown"
    last_round_played = 0
    loop = asyncio.get_event_loop()
    # Overall wall-clock budget for the bot. Generous so slow opponents
    # can't strand us; the orchestrator runs a faster status-based
    # watchdog in parallel.
    deadline = loop.time() + rounds * per_round_timeout_s + 60.0

    try:
        await adapter.initialize()

        # Join the tournament; server records a Participant row.
        join_result = await adapter.call_tool(
            "join_tournament",
            {"tournament_id": tournament_id, "agent_name": agent_name},
        )
        logger.info("[%s] joined: %s", agent_name, join_result)

        # Main play loop: keep polling + submitting until the server tells us
        # the tournament is over (pending=False AND round_number==total_rounds
        # AND we've already played this round).
        while loop.time() < deadline:
            state = await adapter.call_tool(
                "get_current_state", {"tournament_id": tournament_id}
            )
            state_body = _unwrap_tool_result(state)

            game_type = state_body.get("game_type")
            if game_type != "el_farol":
                raise RuntimeError(
                    f"[{agent_name}] unexpected game_type={game_type!r}; "
                    "this bot only handles el_farol."
                )

            pending = state_body.get("pending_submission")
            round_number = int(state_body.get("round_number") or 0)
            total_rounds = int(state_body.get("total_rounds") or 0)

            if not pending:
                final_score = state_body.get("your_cumulative_score")
                # Tournament is over ONLY when we're on the final round AND
                # we already played it. Otherwise we're in a transient
                # between-rounds gap (just-resolved R_n, R_{n+1} not yet
                # created) — keep polling.
                if round_number >= total_rounds and last_round_played >= total_rounds:
                    final_status = "completed"
                    break
                await asyncio.sleep(0.3)
                continue

            # Skip submitting the same round twice in a row (defensive).
            if round_number == last_round_played:
                await asyncio.sleep(0.3)
                continue

            slots = _random_slots(state_body, rng)
            move = await adapter.call_tool(
                "make_move",
                {"tournament_id": tournament_id, "action": {"slots": slots}},
            )
            rounds_played += 1
            last_round_played = round_number
            logger.info(
                "[%s] r%s: slots=%s -> %s",
                agent_name,
                round_number,
                slots,
                _unwrap_tool_result(move).get("status"),
            )
        else:
            final_status = "timeout"

        # Release participation so the same bot user can play a fresh
        # tournament. Server doesn't auto-release on completion (see LABS
        # open issue). Best-effort; don't fail the bot if leave errors.
        if final_status == "completed":
            try:
                await adapter.call_tool(
                    "mcp_leave_tournament",
                    {"tournament_id": tournament_id},
                )
            except Exception as e:
                logger.warning("[%s] leave failed: %s", agent_name, e)

    finally:
        try:
            await adapter.disconnect()
        except Exception:
            pass

    return {
        "agent_name": agent_name,
        "final_score": final_score,
        "rounds_played": rounds_played,
        "final_status": final_status,
    }


def _unwrap_tool_result(raw: dict[str, Any]) -> dict[str, Any]:
    """FastMCP wraps tool returns in ``{"content": [...]}``; unwrap it.

    The tool functions on the server return plain dicts, but the MCP
    transport wraps them. Some MCP clients surface the wrapped form,
    others pre-unwrap. Handle both so the bot is portable.
    """
    if not isinstance(raw, dict):
        return {}
    # FastMCP structured content: ``{"structuredContent": {...}}``.
    if "structuredContent" in raw and isinstance(raw["structuredContent"], dict):
        return raw["structuredContent"]
    # Some clients return ``{"content": [{"type": "text", "text": "..."}]}``.
    if "content" in raw and isinstance(raw["content"], list):
        for item in raw["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                import json

                try:
                    parsed = json.loads(item.get("text", ""))
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    return raw


def _random_slots(state: dict[str, Any], rng: random.Random) -> list[int]:
    """Pick a valid random subset of slots per server constraints."""
    num_slots = int(state.get("num_slots", DEFAULT_NUM_SLOTS))
    max_per_day = int(
        (state.get("action_schema") or {}).get("max_length", DEFAULT_MAX_SLOTS_PER_DAY)
    )
    k = rng.randint(0, min(max_per_day, num_slots))
    return sorted(rng.sample(range(num_slots), k))
