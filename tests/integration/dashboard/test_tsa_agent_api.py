"""Integration tests for the purpose-aware agent API (LABS-TSA PR-2)."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
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


class TestAgentPurposeAPI:
    @pytest.mark.anyio
    async def test_create_tournament_agent(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={
                    "name": "my-tournament-bot",
                    "agent_type": "mcp",
                    "purpose": "tournament",
                },
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["purpose"] == "tournament"

    @pytest.mark.anyio
    async def test_default_purpose_is_benchmark(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={"name": "legacy-bot", "agent_type": "mcp"},
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["purpose"] == "benchmark"

    @pytest.mark.anyio
    async def test_filter_list_by_purpose(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={"name": "b1", "agent_type": "http"},
            )
            await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={
                    "name": "t1",
                    "agent_type": "mcp",
                    "purpose": "tournament",
                },
            )
            resp = await client.get(
                "/api/v1/agents?purpose=tournament", headers=auth_headers
            )
            names = {a["name"] for a in resp.json()}
            assert names == {"t1"}


class TestAgentQuota:
    @pytest.mark.anyio
    async def test_tournament_quota_rejects_sixth(
        self, v2_app, auth_headers, monkeypatch
    ) -> None:
        monkeypatch.setenv("ATP_MAX_TOURNAMENT_AGENTS_PER_USER", "5")
        # Must clear cached config
        from atp.dashboard.v2.config import get_config

        get_config.cache_clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            for i in range(5):
                resp = await client.post(
                    "/api/v1/agents",
                    headers=auth_headers,
                    json={
                        "name": f"t{i}",
                        "agent_type": "mcp",
                        "purpose": "tournament",
                    },
                )
                assert resp.status_code == 201, (i, resp.text)
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={
                    "name": "t5",
                    "agent_type": "mcp",
                    "purpose": "tournament",
                },
            )
            assert resp.status_code == 429
            assert "tournament agent quota" in resp.text.lower()

    @pytest.mark.anyio
    async def test_benchmark_quota_independent(
        self, v2_app, auth_headers, monkeypatch
    ) -> None:
        monkeypatch.setenv("ATP_MAX_BENCHMARK_AGENTS_PER_USER", "2")
        monkeypatch.setenv("ATP_MAX_TOURNAMENT_AGENTS_PER_USER", "2")
        from atp.dashboard.v2.config import get_config

        get_config.cache_clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # 2 benchmark agents — fill quota
            for i in range(2):
                resp = await client.post(
                    "/api/v1/agents",
                    headers=auth_headers,
                    json={"name": f"b{i}", "agent_type": "http"},
                )
                assert resp.status_code == 201

            # 3rd benchmark rejected
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={"name": "b2", "agent_type": "http"},
            )
            assert resp.status_code == 429

            # But tournament slot still open
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={
                    "name": "t0",
                    "agent_type": "mcp",
                    "purpose": "tournament",
                },
            )
            assert resp.status_code == 201
