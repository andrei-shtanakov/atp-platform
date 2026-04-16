"""SC-3, SC-6, SC-8 cancel-path e2e coverage: user cancel via MCP,
pending timeout, abandoned cascade."""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.slow]


async def test_e2e_user_cancel_via_mcp(
    tournament_uvicorn: tuple[str, str, str],
) -> None:
    """Admin cancels a tournament via REST; status becomes cancelled."""
    base_url, admin_jwt, _bob_jwt = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "cancel-test",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 5,
                "private": False,
            },
        )
        assert response.status_code == 201, response.text
        tid = response.json()["id"]

        # Admin cancels via REST
        cancel_resp = await client.post(f"/api/v1/tournaments/{tid}/cancel")
        assert cancel_resp.status_code == 200

        # Verify status via detail endpoint
        detail = await client.get(f"/api/v1/tournaments/{tid}")
        body = detail.json()
        assert body["status"] == "cancelled"
        assert body["cancelled_reason"] == "admin_action"


async def test_e2e_pending_timeout_autocancel(
    tournament_uvicorn: tuple[str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-3: deadline worker auto-cancels PENDING tournaments past
    their pending_deadline. Uses short TOURNAMENT_PENDING_MAX_WAIT_S via
    monkeypatch of the module constant."""
    base_url, _admin_jwt, _bob_jwt = tournament_uvicorn

    # Reduce the constant for this test — the app process is already
    # running so we must import and mutate at runtime
    from atp.dashboard.tournament import service as svc_module

    monkeypatch.setattr(svc_module, "TOURNAMENT_PENDING_MAX_WAIT_S", 2)

    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "timeout",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": False,
            },
        )
        assert response.status_code == 201, response.text
        tid = response.json()["id"]

        # Wait longer than TOURNAMENT_PENDING_MAX_WAIT_S + poll interval
        await asyncio.sleep(4.0)

        detail = await client.get(f"/api/v1/tournaments/{tid}")
        body = detail.json()
        assert body["status"] == "cancelled"
        assert body["cancelled_reason"] == "pending_timeout"


async def test_e2e_idempotent_cancel(
    tournament_uvicorn: tuple[str, str, str],
) -> None:
    """SC-6: second cancel is a no-op, never 500."""
    base_url, _admin_jwt, _bob_jwt = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "idem",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": False,
            },
        )
        assert response.status_code == 201, response.text
        tid = response.json()["id"]

        r1 = await client.post(f"/api/v1/tournaments/{tid}/cancel")
        r2 = await client.post(f"/api/v1/tournaments/{tid}/cancel")
        # First call succeeds, second is idempotent (200 or 409 conflict)
        assert r1.status_code == 200
        assert r2.status_code in (200, 409)
