"""LABS-TSA PR-3 — MCP and benchmark-API auth gating by agent purpose.

Covers:
- APIToken.agent_purpose snapshot at issuance (Task 3.2).
- JWTUserStateMiddleware surfacing agent_purpose on request.state (Task 3.3).
- MCPAuthMiddleware rejecting non-tournament tokens (Task 3.4).
- Benchmark API rejecting tournament tokens (Task 3.5).
- Legacy-token lazy fallback when agent_purpose IS NULL (Task 3.6).
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.tokens import APIToken
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


async def _mint_agent_token(
    client: AsyncClient,
    auth_headers: dict[str, str],
    *,
    agent_name: str,
    purpose: str,
    token_name: str,
) -> tuple[int, str]:
    """Register an agent and mint an agent-scoped token.

    Returns (agent_id, raw_token).
    """
    resp = await client.post(
        "/api/v1/agents",
        headers=auth_headers,
        json={
            "name": agent_name,
            "agent_type": "mcp" if purpose == "tournament" else "http",
            "purpose": purpose,
        },
    )
    assert resp.status_code == 201, resp.text
    agent_id = resp.json()["id"]

    resp = await client.post(
        "/api/v1/tokens",
        headers=auth_headers,
        json={"agent_id": agent_id, "name": token_name},
    )
    assert resp.status_code == 201, resp.text
    return agent_id, resp.json()["token"]


async def _mint_user_token(
    client: AsyncClient, auth_headers: dict[str, str], *, token_name: str
) -> str:
    """Mint a user-scoped token (no agent_id)."""
    resp = await client.post(
        "/api/v1/tokens", headers=auth_headers, json={"name": token_name}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["token"]


class TestTokenSnapshot:
    @pytest.mark.anyio
    async def test_issuing_tournament_agent_token_records_purpose(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            agent_id, _raw = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="t1",
                purpose="tournament",
                token_name="t1-token",
            )

        stmt = select(APIToken).where(APIToken.agent_id == agent_id)
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.agent_purpose == "tournament"

    @pytest.mark.anyio
    async def test_issuing_benchmark_agent_token_records_purpose(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            agent_id, _raw = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="b1",
                purpose="benchmark",
                token_name="b1-token",
            )

        stmt = select(APIToken).where(APIToken.agent_id == agent_id)
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.agent_purpose == "benchmark"

    @pytest.mark.anyio
    async def test_user_level_token_has_null_purpose(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/tokens",
                headers=auth_headers,
                json={"name": "user-tok"},
            )
            assert resp.status_code == 201

        stmt = select(APIToken).where(APIToken.agent_id.is_(None))
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.agent_purpose is None


class TestMCPAuthGate:
    @pytest.mark.anyio
    async def test_benchmark_token_rejected_by_mcp(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            _agent_id, token = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="bench1",
                purpose="benchmark",
                token_name="bench-tok",
            )
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code == 403, resp.text
            assert "benchmark-agents only" in resp.text.lower() or (
                "this token belongs to a benchmark agent" in resp.text.lower()
            )

    @pytest.mark.anyio
    async def test_user_level_token_rejected_by_mcp(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            token = await _mint_user_token(client, auth_headers, token_name="user-tok")
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code == 403, resp.text
            assert "agent-scoped token" in resp.text.lower()

    @pytest.mark.anyio
    async def test_tournament_token_accepted_by_mcp(self, v2_app, auth_headers) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            _agent_id, token = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="t_ok",
                purpose="tournament",
                token_name="t-ok-tok",
            )
            # A bare GET to /mcp/ may 404/406/etc. because MCP requires
            # a full SSE handshake. We only assert the auth gate does not
            # 403 for a valid tournament token.
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code != 403, resp.text


class TestBenchmarkAPIGate:
    @pytest.mark.anyio
    async def test_tournament_token_rejected_by_benchmark_api(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            _agent_id, token = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="t-for-bench",
                purpose="tournament",
                token_name="t-for-bench-tok",
            )
            resp = await client.get(
                "/api/v1/benchmarks",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 403, resp.text
            assert "benchmark-agents only" in resp.text.lower()

    @pytest.mark.anyio
    async def test_benchmark_token_accepted_by_benchmark_api(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            _agent_id, token = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="b-for-bench",
                purpose="benchmark",
                token_name="b-for-bench-tok",
            )
            resp = await client.get(
                "/api/v1/benchmarks",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200, resp.text

    @pytest.mark.anyio
    async def test_user_token_accepted_by_benchmark_api(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            token = await _mint_user_token(
                client, auth_headers, token_name="u-for-bench"
            )
            resp = await client.get(
                "/api/v1/benchmarks",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200, resp.text


class TestLegacyTokenLookup:
    @pytest.mark.anyio
    async def test_token_with_null_agent_purpose_still_resolves(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        """A token issued before PR-3 has agent_purpose=NULL; middleware
        must fall back to Agent.purpose lookup and cache the result."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            agent_id, token = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="legacy-t",
                purpose="tournament",
                token_name="legacy-tok",
            )

        # Simulate a pre-PR-3 token: NULL out the agent_purpose snapshot.
        await db_session.execute(
            update(APIToken)
            .where(APIToken.agent_id == agent_id)
            .values(agent_purpose=None)
        )
        await db_session.commit()

        # Clear any cached resolution so the test exercises the lookup path.
        from atp.dashboard.v2.rate_limit import (
            _legacy_purpose_cache,
            _token_auth_cache,
        )

        _legacy_purpose_cache.clear()
        _token_auth_cache.clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code != 403, resp.text

    @pytest.mark.anyio
    async def test_legacy_benchmark_token_still_rejected_by_mcp(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        """Even with agent_purpose=NULL on an benchmark agent's token,
        the fallback should resolve to 'benchmark' and reject at /mcp."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            agent_id, token = await _mint_agent_token(
                client,
                auth_headers,
                agent_name="legacy-b",
                purpose="benchmark",
                token_name="legacy-b-tok",
            )

        await db_session.execute(
            update(APIToken)
            .where(APIToken.agent_id == agent_id)
            .values(agent_purpose=None)
        )
        await db_session.commit()

        from atp.dashboard.v2.rate_limit import _legacy_purpose_cache

        _legacy_purpose_cache.clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code == 403, resp.text
