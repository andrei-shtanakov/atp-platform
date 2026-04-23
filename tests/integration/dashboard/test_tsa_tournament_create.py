"""LABS-TSA PR-4 — tournament creation with builtin roster + private cap."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    """Create a v2 app bound to the shared in-memory test database."""
    app = create_test_app()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


async def _register_tournament_agent(
    client: AsyncClient, auth_headers: dict[str, str], name: str = "creator-t"
) -> int:
    resp = await client.post(
        "/api/v1/agents",
        headers=auth_headers,
        json={"name": name, "agent_type": "mcp", "purpose": "tournament"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


class TestCreateWithRoster:
    @pytest.mark.anyio
    async def test_private_tournament_with_builtin_roster(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # creator first registers at least one tournament agent to
            # satisfy the "creator commit" validator
            await _register_tournament_agent(client, auth_headers)

            resp = await client.post(
                "/api/v1/tournaments",
                headers=auth_headers,
                json={
                    "name": "private-with-roster",
                    "game_type": "el_farol",
                    "num_players": 3,
                    "total_rounds": 2,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [
                        {"builtin_strategy": "el_farol/traditionalist"},
                        {"builtin_strategy": "el_farol/contrarian"},
                    ],
                },
            )
            assert resp.status_code == 201, resp.text
            body = resp.json()
            assert body["join_token"] is not None

    @pytest.mark.anyio
    async def test_private_tournament_rejected_without_tournament_agent(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # creator has zero tournament-purpose agents AND does not
            # commit a full builtin roster (roster < num_players)
            resp = await client.post(
                "/api/v1/tournaments",
                headers=auth_headers,
                json={
                    "name": "no-commit",
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 2,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [],
                },
            )
            assert resp.status_code == 400, resp.text

    @pytest.mark.anyio
    async def test_unknown_builtin_rejected(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await _register_tournament_agent(client, auth_headers)
            resp = await client.post(
                "/api/v1/tournaments",
                headers=auth_headers,
                json={
                    "name": "bad-roster",
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 1,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [{"builtin_strategy": "el_farol/nope"}],
                },
            )
            assert resp.status_code == 400, resp.text
            assert "unknown builtin" in resp.text.lower()

    @pytest.mark.anyio
    async def test_cross_game_builtin_rejected(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await _register_tournament_agent(client, auth_headers)
            resp = await client.post(
                "/api/v1/tournaments",
                headers=auth_headers,
                json={
                    "name": "cross-game",
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 1,
                    "round_deadline_s": 30,
                    "private": True,
                    # tit_for_tat is a PD strategy, not El Farol
                    "roster": [{"builtin_strategy": "prisoners_dilemma/tit_for_tat"}],
                },
            )
            assert resp.status_code == 400, resp.text

    @pytest.mark.anyio
    async def test_oversized_roster_rejected(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await _register_tournament_agent(client, auth_headers)
            resp = await client.post(
                "/api/v1/tournaments",
                headers=auth_headers,
                json={
                    "name": "too-many",
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 1,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [
                        {"builtin_strategy": "el_farol/traditionalist"},
                        {"builtin_strategy": "el_farol/contrarian"},
                        {"builtin_strategy": "el_farol/gambler"},
                    ],
                },
            )
            assert resp.status_code == 400, resp.text


class TestConcurrentPrivateCap:
    @pytest.mark.anyio
    async def test_fourth_private_tournament_rejected(
        self, v2_app, auth_headers, monkeypatch
    ) -> None:
        monkeypatch.setenv("ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER", "3")
        get_config.cache_clear()
        try:
            async with AsyncClient(
                transport=ASGITransport(app=v2_app), base_url="http://test"
            ) as client:
                await _register_tournament_agent(client, auth_headers)
                # Full builtin roster covers num_players so creator-commit
                # invariant holds without the creator joining each one.
                body = {
                    "name": "cap-probe",
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 1,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [
                        {"builtin_strategy": "el_farol/traditionalist"},
                        {"builtin_strategy": "el_farol/contrarian"},
                    ],
                }
                for i in range(3):
                    resp = await client.post(
                        "/api/v1/tournaments", headers=auth_headers, json=body
                    )
                    assert resp.status_code == 201, (i, resp.text)
                resp = await client.post(
                    "/api/v1/tournaments", headers=auth_headers, json=body
                )
                assert resp.status_code == 429, resp.text
        finally:
            get_config.cache_clear()
