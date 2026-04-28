#!/usr/bin/env python3
"""Minimal repro for MCP/SSE concurrent ``tools/list`` race.

Goal
----
Validate or refute the hypothesis that under concurrent SSE handshakes
to the same FastMCP endpoint, the Nth client's ``tools/list`` returns
empty / incomplete — the symptom we observed in tournament runs where
the third Claude SDK bot reliably failed cold-start.

This script isolates the race from our application stack: no auth
middleware, no observability events, no real tools. Just FastMCP + SSE
+ ``mcp`` client lib. If the race reproduces here, it lives inside
FastMCP itself; if it does not, the race depends on something specific
to our app integration (auth chain, tool count, scope state, …).

Repro mirrors the production sequence
-------------------------------------

In ``claude_el_farol_3bots.py`` bots spawn staggered (``SPAWN_STAGGER_S``)
and each one's SSE session **stays alive** while the next bot opens
its handshake. So the race is NOT "open N parallel handshakes from
zero". It is "open one handshake, keep it alive, open another, keep
it alive, … open Nth handshake — what does it see?".

Usage
-----
::

    uv run python scripts/repro_mcp_concurrent_tools_list.py
    uv run python scripts/repro_mcp_concurrent_tools_list.py --bots 5 --stagger 2.0
    uv run python scripts/repro_mcp_concurrent_tools_list.py --bots 10 --stagger 0.5

Exit code 0 = all clients saw the full tool list (race did NOT
reproduce). Exit code 1 = at least one client saw fewer tools than
expected (race reproduced — investigate the upstream FastMCP issue).
"""

from __future__ import annotations

import argparse
import socket
import sys
from dataclasses import dataclass

import anyio
import uvicorn
from fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------------------------------------------------------------------------
# FastMCP server with 3 dummy tools
# ---------------------------------------------------------------------------


def build_server(num_tools: int = 9) -> FastMCP:
    """A FastMCP instance with N trivial tools — shape match for the
    prod tool registry (9 tools today). No DB, no auth, no
    notifications. Default of 9 mirrors atp-platform's MCP surface
    so the repro stresses the same ``tools/list`` payload size.

    Tools are registered programmatically (not via decorators) so the
    count is configurable from CLI. Their bodies are no-ops; this
    test only exercises the registration / list response path.
    """
    server = FastMCP("repro-server")

    # FastMCP requires the tool callable to have a typed signature so
    # it can build the schema. A unique closure per index avoids
    # registering the same callable repeatedly.
    for i in range(num_tools):
        name = f"tool_{i:02d}"

        async def _impl(x: int = 0, _captured_name: str = name) -> dict:
            """Dummy tool — echoes argument with the registered name."""
            return {"name": _captured_name, "x": x}

        # Manually rename the callable so FastMCP picks the right name.
        _impl.__name__ = name
        server.tool()(_impl)

    return server


def expected_tool_names(num_tools: int) -> set[str]:
    return {f"tool_{i:02d}" for i in range(num_tools)}


def pick_free_port() -> int:
    """Bind to ephemeral port, close, return the number — racy in
    theory but fine for a one-shot repro on developer machines."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Concurrent SSE client tasks
# ---------------------------------------------------------------------------


@dataclass
class ClientResult:
    label: str
    tools_seen: list[str]
    error: str | None = None


async def hold_sse_session(
    label: str,
    sse_url: str,
    hold_event: anyio.Event,
    results: list[ClientResult],
    list_tools_budget_s: float,
) -> None:
    """Connect SSE, initialize, list_tools (under a timeout that
    matches Claude SDK's ToolSearch budget), then hold the session
    open until the test driver releases the gate. Mimics the prod
    bot lifetime: handshake then long-lived activity.

    The ``list_tools_budget_s`` cap simulates Claude SDK's
    deferred-tool retry behaviour — SDK tries ToolSearch a few times
    over ~5 seconds and aborts. Raw ``mcp`` lib otherwise waits
    indefinitely, which would mask the prod symptom.
    """
    try:
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Apply Claude-SDK-style budget here. If FastMCP is
                # slow under concurrent load, this surfaces as a
                # timeout the same way SDK's ToolSearch gives up.
                with anyio.fail_after(list_tools_budget_s):
                    tools_result = await session.list_tools()
                tool_names = sorted(t.name for t in tools_result.tools)
                results.append(ClientResult(label=label, tools_seen=tool_names))
                print(
                    f"[{label}] saw {len(tool_names)} tools",
                    flush=True,
                )
                # Hold the session alive while subsequent clients
                # do their own handshake — this is what makes the
                # repro match the prod scenario.
                await hold_event.wait()
    except TimeoutError:
        results.append(
            ClientResult(
                label=label,
                tools_seen=[],
                error=f"list_tools timed out after {list_tools_budget_s}s "
                "(matches Claude SDK ToolSearch give-up)",
            )
        )
        print(f"[{label}] TIMEOUT after {list_tools_budget_s}s", flush=True)
    except Exception as exc:  # noqa: BLE001 — we want to record any failure shape
        results.append(
            ClientResult(
                label=label,
                tools_seen=[],
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        print(f"[{label}] FAILED: {exc!r}", flush=True)


async def run_repro(
    *,
    bots: int,
    stagger: float,
    num_tools: int,
    list_tools_budget_s: float,
) -> int:
    """Stand up the server, fire N clients with staggered start,
    keep all sessions alive while last client handshakes, then
    diff the tool lists. Returns process exit code."""
    server = build_server(num_tools=num_tools)
    expected = expected_tool_names(num_tools)
    port = pick_free_port()
    sse_url = f"http://127.0.0.1:{port}/sse"

    config = uvicorn.Config(
        server.http_app(transport="sse"),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    uv_server = uvicorn.Server(config)

    results: list[ClientResult] = []
    hold_event = anyio.Event()

    async with anyio.create_task_group() as tg:
        # Background uvicorn — anyio shields it from the rest of the
        # task group's cancellation, but on success we cancel cleanly
        # via tg.cancel_scope.cancel() at the end.
        tg.start_soon(uv_server.serve)

        # Wait until uvicorn is actually accepting connections.
        for _ in range(50):
            await anyio.sleep(0.1)
            if uv_server.started:
                break
        else:
            print("uvicorn failed to start within 5s", file=sys.stderr)
            return 2

        print(
            f"server up at {sse_url}; spawning {bots} clients with {stagger}s stagger",
            flush=True,
        )

        for i in range(bots):
            if i > 0:
                await anyio.sleep(stagger)
            tg.start_soon(
                hold_sse_session,
                f"BOT{i + 1}",
                sse_url,
                hold_event,
                results,
                list_tools_budget_s,
            )

        # Give the last client a comfortable budget to finish its
        # initialize+list_tools; the prod symptom is "client sees
        # empty/short tool list", not "client hangs forever", so
        # 10s is plenty.
        await anyio.sleep(10)

        # Release all held sessions; they exit cleanly, the task
        # group can wind down.
        hold_event.set()
        await anyio.sleep(0.5)

        # Stop the server. ``uvicorn.Server.should_exit`` is the
        # public way to ask it to drain.
        uv_server.should_exit = True
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    # Diff
    print()
    print("=" * 60)
    print(f"Summary: {len(results)} client results")
    failed = []
    for r in results:
        status = "OK"
        if r.error is not None:
            status = f"ERROR ({r.error})"
            failed.append(r)
        elif set(r.tools_seen) != expected:
            missing = expected - set(r.tools_seen)
            extra = set(r.tools_seen) - expected
            parts = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if extra:
                parts.append(f"extra={sorted(extra)}")
            status = "MISMATCH " + " ".join(parts)
            failed.append(r)
        print(f"  {r.label}: {status}")
    print("=" * 60)

    if failed:
        print(
            f"\nRACE REPRODUCED — {len(failed)} of {len(results)} clients "
            "saw an incomplete or failed tool list.",
            flush=True,
        )
        return 1
    print(
        f"\nNo race observed — all {len(results)} clients saw the full tool list.",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bots",
        type=int,
        default=3,
        help="number of concurrent SSE clients (default: 3)",
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=2.0,
        help="seconds between client starts (default: 2.0, matches prod default)",
    )
    parser.add_argument(
        "--num-tools",
        type=int,
        default=9,
        help="number of dummy tools registered on the server "
        "(default: 9, matches prod tool registry)",
    )
    parser.add_argument(
        "--list-tools-budget-s",
        type=float,
        default=5.0,
        help="per-client list_tools timeout in seconds (default: 5.0, "
        "approximates Claude SDK ToolSearch give-up window)",
    )
    args = parser.parse_args()

    # ``anyio.run`` does not forward kwargs to the entry coroutine —
    # wrap with a closure so argparse values reach run_repro cleanly.
    async def _entry() -> int:
        return await run_repro(
            bots=args.bots,
            stagger=args.stagger,
            num_tools=args.num_tools,
            list_tools_budget_s=args.list_tools_budget_s,
        )

    return anyio.run(_entry)


if __name__ == "__main__":
    raise SystemExit(main())
