"""End-to-end PD tournament test (acceptance criterion for v1 slice)."""

from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_e2e_mcp_server_boots(e2e_mcp_server) -> None:
    base_url, port = e2e_mcp_server
    assert base_url.startswith("http://127.0.0.1:")
    assert isinstance(port, int) and port > 0


@pytest.mark.anyio
async def test_seeded_users_have_jwts(e2e_mcp_server, mcp_seeded_users) -> None:
    assert "alice" in mcp_seeded_users
    assert mcp_seeded_users["alice"]["jwt"]
    assert mcp_seeded_users["bob"]["jwt"]
    assert mcp_seeded_users["admin"]["jwt"]


@pytest.mark.anyio
async def test_admin_creates_tournament_directly(
    e2e_mcp_server, mcp_seeded_users
) -> None:
    """Until Plan 2 adds the REST admin endpoint, the test creates the
    tournament by calling the service directly inside the same process.
    The MCP path (for participants) is exercised separately.
    """
    from atp.dashboard.database import get_database
    from atp.dashboard.models import User
    from atp.dashboard.tournament.events import TournamentEventBus
    from atp.dashboard.tournament.service import TournamentService

    db = get_database()
    async with db.session() as session:
        admin = await session.get(User, mcp_seeded_users["admin"]["id"])
        assert admin is not None
        svc = TournamentService(session, TournamentEventBus())
        t = await svc.create_tournament(
            admin=admin,
            name="e2e-pd",
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
        )
        await session.commit()
        assert t.id is not None
        assert t.game_type == "prisoners_dilemma"
        assert t.total_rounds == 3


@pytest.mark.anyio
async def test_two_bots_play_pd_tournament_end_to_end(
    e2e_mcp_server, mcp_seeded_users
) -> None:
    """ACCEPTANCE TEST FOR THE V1 SLICE.

    Two MCPAdapter bots connect to the platform via SSE, join a
    3-round Prisoner's Dilemma tournament, play 'alice always
    cooperates' vs 'bob always defects', and verify final scores
    match the PD payoff matrix (alice=0, bob=15).
    """
    import asyncio

    from sqlalchemy import select

    from atp.adapters.mcp import MCPAdapter, MCPAdapterConfig
    from atp.dashboard.database import get_database
    from atp.dashboard.models import User
    from atp.dashboard.tournament.events import TournamentEventBus
    from atp.dashboard.tournament.models import Participant, Round, Tournament
    from atp.dashboard.tournament.service import TournamentService

    base_url, _ = e2e_mcp_server

    # 1. Admin creates the tournament directly.
    db = get_database()
    async with db.session() as session:
        admin = await session.get(User, mcp_seeded_users["admin"]["id"])
        assert admin is not None
        svc = TournamentService(session, TournamentEventBus())
        t = await svc.create_tournament(
            admin=admin,
            name="e2e-pd-acceptance",
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
        )
        await session.commit()
        tournament_id = t.id
        alice_user_id = mcp_seeded_users["alice"]["id"]
        bob_user_id = mcp_seeded_users["bob"]["id"]

    # 2. Two MCPAdapter bots connect via SSE. Phase 0.2 verified that
    # FastMCP mounted at /mcp exposes the SSE handshake at /mcp/sse.
    sse_url = f"{base_url}/mcp/sse"

    def _make_adapter(agent_id: str, jwt_token: str) -> MCPAdapter:
        return MCPAdapter(
            MCPAdapterConfig(
                agent_id=agent_id,
                transport="sse",
                url=sse_url,
                headers={"Authorization": f"Bearer {jwt_token}"},
                timeout_seconds=20.0,
                startup_timeout=10.0,
            )
        )

    alice_bot = _make_adapter("alice-bot", mcp_seeded_users["alice"]["jwt"])
    bob_bot = _make_adapter("bob-bot", mcp_seeded_users["bob"]["jwt"])

    alice_received: list[dict] = []
    bob_received: list[dict] = []

    async def _consume(adapter: MCPAdapter, received: list[dict]) -> None:
        assert adapter._transport is not None
        try:
            async for event in adapter._transport.stream_events():
                received.append(event)
        except Exception:
            pass

    await alice_bot.initialize()
    await bob_bot.initialize()

    alice_consumer = asyncio.create_task(_consume(alice_bot, alice_received))
    bob_consumer = asyncio.create_task(_consume(bob_bot, bob_received))
    # Let the consumer tasks register before any notifications fly.
    await asyncio.sleep(0)

    async def _wait_for_round_started(received: list[dict], target_round: int) -> None:
        deadline = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < deadline:
            for n in received:
                data = n.get("params", {}).get("data", {})
                if (
                    data.get("event") == "round_started"
                    and data.get("round_number") == target_round
                ):
                    return
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"never got round_started for round {target_round}; received: {received}"
        )

    async def _wait_for_tournament_completed(
        received: list[dict],
    ) -> dict:
        deadline = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < deadline:
            for n in received:
                data = n.get("params", {}).get("data", {})
                if data.get("event") == "tournament_completed":
                    return data
            await asyncio.sleep(0.05)
        raise TimeoutError("never got tournament_completed notification")

    try:
        # 3. Both join the tournament (triggers _start_tournament).
        await alice_bot.call_tool(
            "join_tournament",
            {
                "tournament_id": tournament_id,
                "agent_name": "alice-always-cooperate",
            },
        )
        await bob_bot.call_tool(
            "join_tournament",
            {
                "tournament_id": tournament_id,
                "agent_name": "bob-always-defect",
            },
        )

        # 4. Play 3 rounds.
        for round_n in range(1, 4):
            await _wait_for_round_started(alice_received, round_n)
            await _wait_for_round_started(bob_received, round_n)

            await alice_bot.call_tool(
                "make_move",
                {
                    "tournament_id": tournament_id,
                    "action": {"choice": "cooperate"},
                },
            )
            await bob_bot.call_tool(
                "make_move",
                {
                    "tournament_id": tournament_id,
                    "action": {"choice": "defect"},
                },
            )

        # 5. Wait for tournament_completed on alice's stream (bob would
        # do too — alice is enough to confirm the final fan-out).
        completed = await _wait_for_tournament_completed(alice_received)
        final_scores = {int(k): v for k, v in completed["final_scores"].items()}
        assert final_scores[alice_user_id] == 0.0
        assert final_scores[bob_user_id] == 15.0

    finally:
        alice_consumer.cancel()
        bob_consumer.cancel()
        with contextlib_suppress(asyncio.CancelledError):
            await alice_consumer
        with contextlib_suppress(asyncio.CancelledError):
            await bob_consumer
        await alice_bot.cleanup()
        await bob_bot.cleanup()

    # 6. Direct DB sanity check — all state persisted.
    db = get_database()
    async with db.session() as session:
        tournament = await session.get(Tournament, tournament_id)
        assert tournament is not None
        assert tournament.status == "completed"

        rounds = (
            (
                await session.execute(
                    select(Round)
                    .where(Round.tournament_id == tournament_id)
                    .order_by(Round.round_number)
                )
            )
            .scalars()
            .all()
        )
        assert len(rounds) == 3
        assert all(r.status == "completed" for r in rounds)

        parts = (
            (
                await session.execute(
                    select(Participant).where(
                        Participant.tournament_id == tournament_id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(parts) == 2
        by_user = {p.user_id: p for p in parts}
        assert by_user[alice_user_id].total_score == 0.0
        assert by_user[bob_user_id].total_score == 15.0


def contextlib_suppress(*exceptions):
    import contextlib

    return contextlib.suppress(*exceptions)
