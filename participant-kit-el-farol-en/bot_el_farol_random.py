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


def _choose_random_slots(state: dict[str, Any], rng: random.Random) -> list[int]:
    num_slots = int(state.get("num_slots", 16))
    max_len = int((state.get("action_schema") or {}).get("max_length", 8))
    max_len = max(0, min(max_len, num_slots))

    visits = rng.randint(0, max_len)
    if visits == 0:
        return []
    return sorted(rng.sample(range(num_slots), visits))


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

                slots = _choose_random_slots(state, rng)
                move = await session.call_tool(
                    "make_move",
                    {"tournament_id": tournament_id, "action": {"slots": slots}},
                )
                move_payload = _parse_tool_result(move)
                print(f"round {round_number}: slots={slots} -> {move_payload}")
                last_played_round = round_number


if __name__ == "__main__":
    asyncio.run(main())
