"""Regression test for LABS-54 Phase 2: legacy POST /api/agents returns 410."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore


@pytest.fixture
def v2_app(test_database: Database):
    app = create_test_app(use_v2_routes=True)

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


@pytest.mark.anyio
async def test_legacy_post_api_agents_returns_410(v2_app) -> None:
    """POST /api/agents is deprecated — must return 410 Gone with
    standard deprecation headers and a pointer to the successor
    endpoint. The check is intentionally auth-agnostic: the endpoint
    is gone for all callers.
    """
    async with AsyncClient(
        transport=ASGITransport(app=v2_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agents",
            json={"name": "test", "agent_type": "http"},
        )
    assert response.status_code == 410
    assert response.headers.get("Deprecation") == "true"
    assert "Sunset" in response.headers
    link_header = response.headers.get("Link")
    assert link_header is not None
    assert "/api/v1/agents" in link_header
    assert 'rel="successor-version"' in link_header
    body = response.json()
    assert "/api/v1/agents" in body["detail"]
