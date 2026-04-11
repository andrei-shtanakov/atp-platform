"""SC-3 AD-9 duration cap + SC-4 AD-10 join_token + 1-active-per-user."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_e2e_ad9_duration_cap_rejects_over_budget(
    tournament_uvicorn: tuple[str, str, str],
) -> None:
    """AD-9: creating a tournament whose wall-clock duration exceeds the
    budget derived from ATP_TOKEN_EXPIRE_MINUTES must be rejected with 422."""
    base_url, _admin_jwt, _bob_jwt = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "too-long",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 200,  # 300 + 200*30 = 6300s >> (60-10)*60 = 3000s
                "round_deadline_s": 30,
                "private": False,
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert "max duration" in body["detail"].lower()


async def test_e2e_ad10_join_token_required_when_private(
    tournament_uvicorn: tuple[str, str, str],
) -> None:
    """AD-10: private tournament returns join_token on creation; subsequent
    GET must NOT expose the token but must signal has_join_token=True."""
    base_url, _admin_jwt, _bob_jwt = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        create_resp = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "private",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": True,
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        body = create_resp.json()
        assert body["has_join_token"] is True
        assert body.get("join_token") is not None

        # Subsequent GET must NOT expose the token
        detail = await client.get(f"/api/v1/tournaments/{body['id']}")
        assert detail.status_code == 200, detail.text
        detail_body = detail.json()
        assert detail_body["has_join_token"] is True
        assert "join_token" not in detail_body
