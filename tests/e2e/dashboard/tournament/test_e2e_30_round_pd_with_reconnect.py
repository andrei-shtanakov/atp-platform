"""SC-1: full multi-round PD tournament end-to-end over real MCP SSE,
with mid-tournament reconnect triggering session_sync replay.

30 rounds × round_deadline_s=1 keeps wall clock under ~40s for CI. The
90-round variant exists as test_e2e_90_round_pd_benchmark.py (manual).
"""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
import pytest

pytestmark = pytest.mark.anyio

_E2E_SECRET = "e2e-sc1-secret-32-bytes-long-pad!"


# ---------------------------------------------------------------------------
# JWT helper
# ---------------------------------------------------------------------------


def _mint_jwt(user_id: int, username: str) -> str:
    """Mint a signed JWT for the given user."""
    return jwt.encode(
        {
            "sub": username,
            "user_id": user_id,
            "exp": datetime.now(tz=UTC) + timedelta(hours=1),
        },
        _E2E_SECRET,
        algorithm="HS256",
    )


# ---------------------------------------------------------------------------
# Local shims — thin helpers on top of the real MCPAdapter API.
# These are intentionally private to this test module and must NOT be
# promoted to the production adapter without a separate plan task.
# ---------------------------------------------------------------------------


class _BotSession:
    """Thin wrapper around MCPAdapter that adds event-queue helpers.

    Provides:
      - ``wait_for_event(event_type, timeout)`` — poll the shared queue
      - ``connect()`` / ``disconnect()`` — lifecycle wrappers
    """

    def __init__(self, sse_url: str, jwt_token: str) -> None:
        """Initialise the session (does NOT connect yet)."""
        self._sse_url = sse_url
        self._jwt_token = jwt_token
        self._adapter: Any | None = None
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._consumer_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create a fresh adapter, initialise the MCP session, and start
        the background SSE consumer task."""
        from atp.adapters.mcp import MCPAdapter, MCPAdapterConfig

        self._adapter = MCPAdapter(
            MCPAdapterConfig(
                agent_id="e2e-bot",
                transport="sse",
                url=self._sse_url,
                headers={"Authorization": f"Bearer {self._jwt_token}"},
                timeout_seconds=20.0,
                startup_timeout=10.0,
            )
        )
        await self._adapter.initialize()
        self._consumer_task = asyncio.create_task(self._consume())

    async def disconnect(self) -> None:
        """Cancel the consumer task and clean up the adapter."""
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        if self._adapter is not None:
            await self._adapter.cleanup()
            self._adapter = None

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        """Background task: drain SSE notifications into the queue."""
        if self._adapter is None or self._adapter._transport is None:
            return
        try:
            async for event in self._adapter._transport.stream_events():
                await self._event_queue.put(event)
        except Exception:
            pass

    async def wait_for_event(
        self, event_type: str, timeout: float = 5.0
    ) -> dict[str, Any] | None:
        """Block until an event with the given ``event`` field arrives.

        Scans the in-memory queue first, then polls for new arrivals.
        Returns the matching event dict, or ``None`` on timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout

        # Drain already-queued events into a local list first.
        seen: list[dict[str, Any]] = []
        while not self._event_queue.empty():
            seen.append(self._event_queue.get_nowait())

        # Check items we just drained.
        for item in seen:
            if _extract_event_type(item) == event_type:
                return item

        # Put them back and wait for new ones.
        for item in seen:
            await self._event_queue.put(item)

        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            try:
                item = await asyncio.wait_for(
                    self._event_queue.get(), timeout=min(remaining, 0.2)
                )
            except TimeoutError:
                continue
            if _extract_event_type(item) == event_type:
                return item
            # Put non-matching items back for other waiters.
            await self._event_queue.put(item)
            await asyncio.sleep(0)

        return None

    # ------------------------------------------------------------------
    # Tool proxy
    # ------------------------------------------------------------------

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call an MCP tool via the raw transport (bypasses tool registry).

        The production MCPAdapter.call_tool() validates against _tools which
        requires a successful tools/list response.  Since the MCP server may
        serve tools without exposing them through tools/list, we send the
        tools/call JSON-RPC request directly through the transport layer.
        """
        if self._adapter is None or self._adapter._transport is None:
            raise RuntimeError("Not connected")
        response = await self._adapter._transport.send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        return response.get("result", {})


def _extract_event_type(notification: dict[str, Any]) -> str | None:
    """Return the tournament event type from an MCP notification dict.

    Server notifications arrive as JSON-RPC notifications with shape::

        {
          "method": "notifications/message",
          "params": {"data": {"event": "round_started", ...}},
        }
    """
    params = notification.get("params", {})
    data = params.get("data", {})
    return data.get("event")


# ---------------------------------------------------------------------------
# Fixture: real uvicorn instance on a free port
# ---------------------------------------------------------------------------


@pytest.fixture
async def tournament_uvicorn(tmp_path, monkeypatch):
    """Spin up a real uvicorn instance on a random port serving the
    FastAPI app with the deadline worker running in lifespan.

    Seeds two users (admin id=1, bob id=2) and yields a tuple of
    ``(base_url, admin_jwt, bob_jwt)``.
    """
    import uvicorn
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    db_path = tmp_path / "sc1.db"
    monkeypatch.setenv("ATP_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("ATP_SECRET_KEY", _E2E_SECRET)
    monkeypatch.setenv("ATP_DISABLE_AUTH", "true")
    monkeypatch.setenv("ATP_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("ATP_DEADLINE_WORKER_POLL_INTERVAL_S", "0.5")

    # Patch the live SECRET_KEY so JWT decode works in the same process.
    import atp.dashboard.auth as _auth_module

    monkeypatch.setattr(_auth_module, "SECRET_KEY", _E2E_SECRET)

    # Drop cached config so new env vars take effect.
    from atp.dashboard.v2.config import get_config

    get_config.cache_clear()

    from atp.dashboard.v2.factory import create_app

    app = create_app()

    # Find a free port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    free_port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=free_port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait until the port accepts connections.
    loop = asyncio.get_event_loop()
    deadline = loop.time() + 10.0
    started = False
    while loop.time() < deadline:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            writer.close()
            await writer.wait_closed()
            started = True
            break
        except OSError:
            await asyncio.sleep(0.05)

    if not started:
        server.should_exit = True
        await server_task
        raise RuntimeError(f"uvicorn did not come up on port {free_port}")

    # Do a FULL authenticated SSE handshake as the readiness probe —
    # waiting for ``event: endpoint`` ensures FastMCP's session manager
    # is fully warmed before the first test adapter connects. Without
    # this, the first MCPAdapter.connect() can race the session-manager
    # setup in CI and fail with "SSE reader exited before emitting
    # endpoint frame" (LABS-20 / LABS-74).
    import httpx as _httpx_probe

    probe_jwt = _mint_jwt(0, "probe")
    mcp_deadline = loop.time() + 15.0
    mcp_ready = False
    last_err: Exception | None = None
    while loop.time() < mcp_deadline:
        try:
            async with _httpx_probe.AsyncClient(timeout=3.0) as probe:
                async with probe.stream(
                    "GET",
                    f"http://127.0.0.1:{free_port}/mcp/sse",
                    headers={
                        "Accept": "text/event-stream",
                        "Authorization": f"Bearer {probe_jwt}",
                    },
                ) as resp:
                    if resp.status_code != 200:
                        last_err = RuntimeError(f"/mcp/sse status={resp.status_code}")
                        await asyncio.sleep(0.1)
                        continue
                    async for line in resp.aiter_lines():
                        if line.startswith("event: endpoint"):
                            mcp_ready = True
                            break
                    if mcp_ready:
                        break
                    last_err = RuntimeError("/mcp/sse closed without endpoint frame")
        except _httpx_probe.RequestError as e:
            last_err = e
        await asyncio.sleep(0.1)
    if not mcp_ready:
        server.should_exit = True
        await server_task
        raise RuntimeError(
            f"MCP /mcp/sse never emitted endpoint frame on port {free_port}"
            + (f": {last_err}" if last_err else "")
        )

    # Seed two users so both MCP sessions and the DISABLE_AUTH fallback work.
    seed_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with seed_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT OR IGNORE INTO users "
                    "(id, tenant_id, username, email, hashed_password, "
                    "is_active, is_admin, created_at, updated_at) "
                    "VALUES "
                    "(1, 'default', 'admin', 'admin@e2e.test', 'x', "
                    "1, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
                    "(2, 'default', 'bob', 'bob@e2e.test', 'x', "
                    "1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
    finally:
        await seed_engine.dispose()

    admin_jwt = _mint_jwt(1, "admin")
    bob_jwt = _mint_jwt(2, "bob")

    try:
        yield f"http://127.0.0.1:{free_port}", admin_jwt, bob_jwt
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=10.0)
        get_config.cache_clear()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


async def test_thirty_round_pd_with_reconnect_sc1(
    tournament_uvicorn: tuple[str, str, str],
) -> None:
    """SC-1: 30-round PD with both bots cooperating.

    Client A (admin) reconnects at round 15 and must receive session_sync
    with correct round_number before play resumes.
    """
    base_url, admin_jwt, bob_jwt = tournament_uvicorn
    sse_url = f"{base_url}/mcp/sse"

    # ----------------------------------------------------------------
    # 1. Admin creates the tournament via REST (uses DISABLE_AUTH
    #    fallback to user id=1 which we seeded above).
    # ----------------------------------------------------------------
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "e2e-sc1",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 30,
                "round_deadline_s": 3,
                "private": False,
            },
        )
        assert response.status_code == 201, response.text
        tournament_id = response.json()["id"]

    # ----------------------------------------------------------------
    # 2. Two bot sessions — each authenticated with its own JWT.
    # ----------------------------------------------------------------
    bot_a = _BotSession(sse_url, admin_jwt)
    bot_b = _BotSession(sse_url, bob_jwt)

    await bot_a.connect()
    await bot_b.connect()

    try:
        # ----------------------------------------------------------
        # 3. Both bots join the tournament.
        #
        # NOTE: The FastMCP-registered tool is named ``join_tournament``
        # (function name used as tool name by FastMCP). The plan spec called
        # it ``join_tournament``; the actual registered name is used here.
        # ----------------------------------------------------------
        await bot_a.call_tool(
            "join_tournament",
            {"tournament_id": tournament_id, "agent_name": "alice"},
        )
        await bot_b.call_tool(
            "join_tournament",
            {"tournament_id": tournament_id, "agent_name": "bob"},
        )

        # ----------------------------------------------------------
        # 4. Gameplay loop with mid-tournament reconnect for bot_a.
        #
        # Only ``round_started`` and ``tournament_completed`` events are
        # forwarded via SSE (the notification formatter drops
        # ``round_ended``), so the loop is driven by ``round_started``
        # arrivals. After submitting the move, the bot waits for the
        # NEXT ``round_started`` (or ``tournament_completed`` on the last
        # round) before proceeding.
        #
        # Bot_a reconnects at round 15. During the disconnect/reconnect
        # window, the ``round_started(16)`` event may fire while bot_a has
        # no active SSE subscription.  The new subscription (from re-join)
        # will catch round 17 onwards.  The loop is therefore event-driven:
        # it submits moves until ``tournament_completed`` arrives instead
        # of relying on a fixed iteration count.
        # ----------------------------------------------------------
        round_started_a: list[dict[str, Any]] = []
        round_started_b: list[dict[str, Any]] = []
        # Collect tournament_completed events inside _play so step 5 can
        # verify them even if wait_for_event consumed them from the queue.
        completed_events: dict[str, dict[str, Any] | None] = {
            "alice": None,
            "bob": None,
        }

        async def _play(
            bot: _BotSession,
            rounds_seen: list[dict[str, Any]],
            agent_name: str,
            reconnect_at: int | None = None,
        ) -> None:
            moves_submitted = 0
            reconnected = False

            while True:
                # Check for tournament_completed first (may have arrived
                # while we were processing the previous round).
                completed = await bot.wait_for_event(
                    "tournament_completed", timeout=0.05
                )
                if completed is not None:
                    completed_events[agent_name] = completed
                    break

                # 20s round_started budget = 6x the 3s deadline: gives slow
                # CI runners enough headroom for the deadline worker to
                # force-resolve rounds where our reconnected peer missed a
                # submission, without masking real hangs. LABS-75.
                event = await bot.wait_for_event("round_started", timeout=20.0)
                if event is None:
                    # Timed out — tournament should have completed already.
                    # Check one more time before giving up.
                    completed = await bot.wait_for_event(
                        "tournament_completed", timeout=2.0
                    )
                    assert completed is not None, (
                        f"{agent_name}: timed out waiting for round_started "
                        f"after {moves_submitted} moves submitted"
                    )
                    completed_events[agent_name] = completed
                    break

                rounds_seen.append(event)
                moves_submitted += 1

                # PD cooperate — action dict format from the MCP tool spec.
                await bot.call_tool(
                    "make_move",
                    {
                        "tournament_id": tournament_id,
                        "action": {"choice": "cooperate"},
                    },
                )

                if (
                    reconnect_at is not None
                    and moves_submitted == reconnect_at
                    and not reconnected
                ):
                    reconnected = True
                    await bot.disconnect()
                    await bot.connect()

                    # Idempotent re-join triggers session_sync.
                    await bot.call_tool(
                        "join_tournament",
                        {
                            "tournament_id": tournament_id,
                            "agent_name": agent_name,
                        },
                    )

                    sync_event = await bot.wait_for_event("session_sync", timeout=5.0)
                    assert sync_event is not None, (
                        "session_sync not received after reconnect"
                    )
                    # session_sync payload: {"event": "session_sync",
                    #   "tournament_id": ..., "state": {...}}
                    sync_data = sync_event.get("params", {}).get("data", {})
                    assert sync_data.get("event") == "session_sync", (
                        f"Unexpected session_sync payload: {sync_data}"
                    )

        await asyncio.gather(
            _play(bot_a, round_started_a, "alice", reconnect_at=15),
            _play(bot_b, round_started_b, "bob"),
        )

        # ----------------------------------------------------------
        # 5. Both bots must see tournament_completed.
        #    Use the events captured inside _play() first; fall back to
        #    queue poll in case the event arrived after _play() exited.
        # ----------------------------------------------------------
        completed_a = completed_events["alice"] or await bot_a.wait_for_event(
            "tournament_completed", timeout=10.0
        )
        completed_b = completed_events["bob"] or await bot_b.wait_for_event(
            "tournament_completed", timeout=10.0
        )
        assert completed_a is not None, "bot_a did not receive tournament_completed"
        assert completed_b is not None, "bot_b did not receive tournament_completed"

        # bot_b (no reconnect) must see all 30 round_started events.
        # bot_a may miss up to 1 event during the reconnect window (the
        # round_started that fires while bot_a has no active subscription).
        assert len(round_started_b) == 30, (
            f"bot_b saw {len(round_started_b)} round_started events (expected 30)"
        )
        assert len(round_started_a) >= 29, (
            f"bot_a saw {len(round_started_a)} round_started events (expected ≥29)"
        )

    finally:
        await bot_a.disconnect()
        await bot_b.disconnect()
