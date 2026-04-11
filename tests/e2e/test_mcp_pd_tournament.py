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
