"""SC-8: operator cancels a stuck tournament via curl-equivalent
REST call. The MCP half of SC-8 is covered in test_e2e_cancel_paths."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_e2e_rest_cancel_returns_200(
    tournament_uvicorn: tuple[str, str, str],
) -> None:
    """SC-8: POST /api/v1/tournaments/{id}/cancel returns 200 and the
    subsequent GET confirms status=cancelled, cancelled_reason=admin_action."""
    base_url, _admin_jwt, _bob_jwt = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        create_resp = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "ops-cancel",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": False,
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        tid = create_resp.json()["id"]

        # curl-equivalent: POST with Authorization header (ATP_DISABLE_AUTH=true
        # in the e2e fixture so the token value is not validated)
        response = await client.post(
            f"/api/v1/tournaments/{tid}/cancel",
            headers={"Authorization": "Bearer test-admin-token"},
        )
        assert response.status_code == 200, response.text

        detail = await client.get(f"/api/v1/tournaments/{tid}")
        body = detail.json()
        assert body["status"] == "cancelled"
        assert body["cancelled_reason"] == "admin_action"
