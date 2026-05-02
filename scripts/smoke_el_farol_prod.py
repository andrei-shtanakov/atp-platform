"""Production smoke: N=5 El Farol tournament with random-strategy MCP bots.

Validates that a merged el_farol tournament actually runs end-to-end on a
live ATP dashboard. Admin creates the tournament over REST, then 5
``atp_a_...``-authenticated bots join + play via MCP until completion.

Env vars (typically loaded from ~/.atp-smoke-tokens.env):
    ATP_ADMIN_TOKEN         user-level token used to create the tournament
    ATP_BOT_TOKEN_0..4      agent-scoped tokens for 5 distinct users

CLI:
    uv run python scripts/smoke_el_farol_prod.py \\
        --server=https://atp.pr0sto.space \\
        --rounds=20 \\
        --round-deadline=60

Exit 0 = tournament reached COMPLETED status and all 5 scores were written.
Exit non-zero = server or bot error; stderr carries a traceback.
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
from bots.el_farol_mcp_bot import run_bot  # noqa: E402

NUM_BOTS = 5
BOT_AGENT_PREFIX = "smoke-bot"

logger = logging.getLogger("smoke_el_farol_prod")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, help="ATP dashboard base URL")
    p.add_argument("--rounds", type=int, default=20)
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
                "game_type": "el_farol",
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


async def _check_winners_page(
    *,
    server: str,
    tournament_id: int,
    expected_agent_names: list[str],
    verify_ssl: bool,
) -> None:
    """Fetch the winners page for the just-completed tournament and
    assert each expected agent name appears in the rendered HTML.

    Raises ``RuntimeError`` on non-200 status or missing agent names so
    the caller can convert it into a smoke failure exit.
    """
    winners_url = f"{server.rstrip('/')}/ui/tournaments/{tournament_id}/winners"
    async with httpx.AsyncClient(verify=verify_ssl, timeout=10) as client:
        resp = await client.get(winners_url)
    if resp.status_code != 200:
        raise RuntimeError(
            f"winners page HTTP {resp.status_code} for tournament "
            f"{tournament_id}; body={resp.text[:200]!r}"
        )
    missing = [name for name in expected_agent_names if name not in resp.text]
    if missing:
        raise RuntimeError(
            f"winners page rendered but missing agent names: {missing!r}"
        )


async def _run() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    admin_token, bot_tokens = _load_tokens()
    name = args.name or f"smoke-el-farol-{int(time.time())}"
    verify_ssl = not args.no_verify_ssl

    # 1. Admin creates tournament.
    t_id = await _create_tournament(
        server=args.server,
        admin_token=admin_token,
        rounds=args.rounds,
        round_deadline=args.round_deadline,
        name=name,
        verify_ssl=verify_ssl,
    )
    print(f"created tournament id={t_id} name={name!r}")

    # 2. 5 bots play in parallel. Staggered seeds so random choices diverge.
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

    # 3. Tournament-status watchdog in parallel, so we can fail fast if a
    # bot hangs without tripping global cancellation.
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

    # Let bot tasks drain (they should see status flip and exit cleanly).
    bot_results = await asyncio.gather(*bot_tasks, return_exceptions=True)

    print(f"tournament {t_id} status={final.get('status')!r}")

    # Smoke-check that the public winners poster renders for the
    # just-completed tournament — catches route deploy regressions.
    expected_agent_names = [f"{BOT_AGENT_PREFIX}-{i}" for i in range(NUM_BOTS)]
    try:
        await _check_winners_page(
            server=args.server,
            tournament_id=t_id,
            expected_agent_names=expected_agent_names,
            verify_ssl=verify_ssl,
        )
        print(f"  ✓ winners page renders ({len(expected_agent_names)} agents)")
    except Exception as e:
        print(f"FAIL: winners page check: {e}", file=sys.stderr)
        return 1

    failures = 0
    for res in bot_results:
        if isinstance(res, Exception):
            print(f"  bot FAIL: {res!r}", file=sys.stderr)
            failures += 1
        else:
            print(
                f"  {res['agent_name']:18s} "
                f"rounds_played={res['rounds_played']:2d} "
                f"score={res['final_score']} "
                f"status={res['final_status']}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
