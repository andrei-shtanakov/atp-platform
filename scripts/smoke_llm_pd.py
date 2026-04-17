"""LLM-strategy smoke: 2-player Prisoner's Dilemma via MCP against a live server.

Mirrors ``smoke_el_farol_prod.py`` but runs two LLM-backed bots instead of
random-strategy ones. Validates that ``llm_mcp_bot.run_bot`` completes a
real PD tournament end-to-end and that reasoning ends up on the server
via the PR #44 field.

Env vars (typically loaded from ``~/.atp-smoke-tokens.env``):
    OPENAI_API_KEY       required (or ANTHROPIC_API_KEY if LLM_PROVIDER=anthropic)
    ATP_ADMIN_TOKEN      user-level token that creates the tournament
    ATP_BOT_TOKEN_0      agent-scoped token for bot #0
    ATP_BOT_TOKEN_1      agent-scoped token for bot #1
    LLM_PROVIDER         optional, "openai" (default) | "anthropic"
    LLM_MODEL            optional, "gpt-4o-mini" (default) | "claude-haiku-4-5-20251001"

CLI:
    uv run python scripts/smoke_llm_pd.py \\
        --server=https://atp.pr0sto.space --rounds=10 --round-deadline=60

Exit 0 = tournament completed, both bots played all rounds.
Exit non-zero = bot or server error.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo-game"))
from bots.llm_mcp_bot import run_bot  # noqa: E402

NUM_BOTS = 2
BOT_AGENT_PREFIX = "smoke-llm-pd"

logger = logging.getLogger("smoke_llm_pd")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, help="ATP dashboard base URL")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--round-deadline", type=int, default=60)
    p.add_argument(
        "--name", default=None, help="Tournament name (default: timestamp-based)"
    )
    p.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Skip TLS verification (local dev against self-signed cert)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _load_tokens() -> tuple[str, list[str]]:
    admin = os.environ.get("ATP_ADMIN_TOKEN")
    if not admin:
        sys.exit("ATP_ADMIN_TOKEN is not set; source ~/.atp-smoke-tokens.env first")
    bots: list[str] = []
    for i in range(NUM_BOTS):
        tok = os.environ.get(f"ATP_BOT_TOKEN_{i}")
        if not tok:
            sys.exit(f"ATP_BOT_TOKEN_{i} is not set")
        bots.append(tok)
    return admin, bots


async def _create_tournament(
    *,
    server: str,
    admin_token: str,
    rounds: int,
    round_deadline: int,
    name: str,
    verify_ssl: bool,
) -> int:
    async with httpx.AsyncClient(verify=verify_ssl, timeout=30) as client:
        resp = await client.post(
            f"{server.rstrip('/')}/api/v1/tournaments",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "name": name,
                "game_type": "prisoners_dilemma",
                "num_players": NUM_BOTS,
                "total_rounds": rounds,
                "round_deadline_s": round_deadline,
            },
        )
        if resp.status_code not in (200, 201):
            sys.exit(f"Tournament create failed: {resp.status_code} {resp.text}")
        body = resp.json()
        return int(body["id"])


async def _wait_for_completion(
    *,
    server: str,
    admin_token: str,
    tournament_id: int,
    timeout_s: float,
    verify_ssl: bool,
) -> dict[str, Any]:
    """Poll tournament status until COMPLETED or the deadline expires."""
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(verify=verify_ssl, timeout=10) as client:
        while time.monotonic() < deadline:
            resp = await client.get(
                f"{server.rstrip('/')}/api/v1/tournaments/{tournament_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            if resp.status_code != 200:
                await asyncio.sleep(1.0)
                continue
            body = resp.json()
            status = body.get("status")
            if status == "completed":
                return body
            if status == "cancelled":
                reason = body.get("cancelled_reason")
                raise RuntimeError(
                    f"tournament {tournament_id} cancelled "
                    f"(reason={reason!r}); bots likely failed to join"
                )
            await asyncio.sleep(1.0)
    raise TimeoutError(
        f"tournament {tournament_id} did not complete within {timeout_s:.0f}s"
    )


async def _run() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit(
            "neither OPENAI_API_KEY nor ANTHROPIC_API_KEY set; "
            "the LLM bot needs at least one"
        )

    admin_token, bot_tokens = _load_tokens()
    name = args.name or f"smoke-llm-pd-{int(time.time())}"
    verify_ssl = not args.no_verify_ssl

    t_id = await _create_tournament(
        server=args.server,
        admin_token=admin_token,
        rounds=args.rounds,
        round_deadline=args.round_deadline,
        name=name,
        verify_ssl=verify_ssl,
    )
    print(f"created tournament id={t_id} name={name!r}")
    print(f"watch live: {args.server.rstrip('/')}/ui/tournaments/{t_id}")

    bot_tasks = [
        asyncio.create_task(
            run_bot(
                server_url=args.server,
                token=tok,
                tournament_id=t_id,
                agent_name=f"{BOT_AGENT_PREFIX}-{i}",
                rounds=args.rounds,
                seed=1000 + i,
                verify_ssl=verify_ssl,
                per_round_timeout_s=float(args.round_deadline),
            )
        )
        for i, tok in enumerate(bot_tokens)
    ]

    timeout_s = float(args.round_deadline) * args.rounds + 60.0
    watcher = asyncio.create_task(
        _wait_for_completion(
            server=args.server,
            admin_token=admin_token,
            tournament_id=t_id,
            timeout_s=timeout_s,
            verify_ssl=verify_ssl,
        )
    )

    try:
        final = await watcher
    except Exception as e:
        for task in bot_tasks:
            task.cancel()
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    bot_results = await asyncio.gather(*bot_tasks, return_exceptions=True)

    print(f"tournament {t_id} status={final.get('status')!r}")
    failures = 0
    for res in bot_results:
        if isinstance(res, Exception):
            print(f"  bot FAIL: {res!r}", file=sys.stderr)
            failures += 1
        else:
            print(
                f"  {res['agent_name']:22s} "
                f"rounds_played={res['rounds_played']:2d} "
                f"score={res['final_score']} "
                f"status={res['final_status']}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
