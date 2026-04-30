from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client


def _parse_tool_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "structuredContent") and isinstance(
        result.structuredContent, dict
    ):
        return result.structuredContent

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

    if isinstance(result, dict):
        return result
    return {}


def _choose_random_intervals(
    state: dict[str, Any], rng: random.Random
) -> list[list[int]]:
    """Pick a random El Farol action in the canonical interval shape.

    Returns either ``[]`` ("stay home"), one interval, or two
    non-overlapping non-adjacent intervals. Total covered slots stays
    within ``max_total_slots``.
    """
    num_slots = int(state.get("num_slots", 16))
    schema = state.get("action_schema") or {}
    max_intervals = int(schema.get("max_intervals", 2))
    max_total_slots = int(schema.get("max_total_slots", 8))
    max_total_slots = max(0, min(max_total_slots, num_slots))
    max_intervals = max(0, min(max_intervals, max_total_slots))

    if max_total_slots == 0 or max_intervals == 0:
        return []

    n_intervals = rng.randint(0, max_intervals)
    if n_intervals == 0:
        return []

    if n_intervals == 1:
        length = rng.randint(1, max_total_slots)
        max_start = num_slots - length
        start = rng.randint(0, max_start)
        return [[start, start + length - 1]]

    # Two intervals: pick lengths summing to <= max_total_slots, then
    # place them with at least one empty slot between.
    total = rng.randint(2, max_total_slots)
    len1 = rng.randint(1, total - 1)
    len2 = total - len1
    # Need: start1 + len1 - 1 + 1 < start2 → start2 >= start1 + len1 + 1
    # And: start2 + len2 - 1 < num_slots → start2 <= num_slots - len2
    # Pick start1 first, then start2 in feasible range.
    max_start1 = num_slots - len2 - 1 - len1
    if max_start1 < 0:
        # Can't fit two intervals; fall back to one.
        length = min(total, num_slots)
        max_start = num_slots - length
        start = rng.randint(0, max_start)
        return [[start, start + length - 1]]
    start1 = rng.randint(0, max_start1)
    end1 = start1 + len1 - 1
    min_start2 = end1 + 2
    max_start2 = num_slots - len2
    start2 = rng.randint(min_start2, max_start2)
    end2 = start2 + len2 - 1
    return [[start1, end1], [start2, end2]]


async def main() -> None:
    load_dotenv()
    mcp_url = os.environ["ATP_MCP_URL"]
    token = os.environ["ATP_TOKEN"]
    tournament_id = int(os.environ["TOURNAMENT_ID"])
    agent_name = os.environ.get("AGENT_NAME", "el-farol-random-bot")
    join_token = os.environ.get("JOIN_TOKEN") or None
    seed = int(os.environ.get("BOT_RANDOM_SEED", "42"))

    rng = random.Random(seed)
    headers = {"Authorization": f"Bearer {token}"}

    async with sse_client(mcp_url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            join_payload: dict[str, Any] = {
                "tournament_id": tournament_id,
                "agent_name": agent_name,
            }
            if join_token:
                join_payload["join_token"] = join_token

            join = await session.call_tool("join_tournament", join_payload)
            print("joined:", _parse_tool_result(join))

            last_played_round = 0
            while True:
                raw_state = await session.call_tool(
                    "get_current_state", {"tournament_id": tournament_id}
                )
                state = _parse_tool_result(raw_state)
                status = str(state.get("status", "")).lower()
                if status in {"completed", "cancelled"}:
                    print("tournament status:", status)
                    break

                pending = bool(state.get("pending_submission", False))
                round_number = int(state.get("round_number", 0))
                total_rounds = int(state.get("total_rounds", 0))

                if not pending:
                    if (
                        round_number >= total_rounds
                        and last_played_round >= total_rounds
                    ):
                        print("tournament finished")
                        break
                    await asyncio.sleep(0.5)
                    continue

                if round_number == last_played_round:
                    await asyncio.sleep(0.5)
                    continue

                intervals = _choose_random_intervals(state, rng)
                action_payload: dict[str, Any] = {"intervals": intervals}
                # Optionally add reasoning (max 8000 chars; visible to owner during play)
                # action_payload["reasoning"] = "Random strategy: choosing intervals to optimize threshold"
                move = await session.call_tool(
                    "make_move",
                    {
                        "tournament_id": tournament_id,
                        "action": action_payload,
                    },
                )
                move_payload = _parse_tool_result(move)
                print(f"round {round_number}: intervals={intervals} -> {move_payload}")
                last_played_round = round_number


if __name__ == "__main__":
    asyncio.run(main())
