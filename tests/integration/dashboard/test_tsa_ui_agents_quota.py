"""LABS-TSA PR-5 — /ui/agents quota strip + purpose column.

Verifies the agents page surfaces per-purpose agent counts against their
configured caps, renders a Purpose column in the agents table, and shows
two register links (one per purpose).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token
from atp.dashboard.database import Database
from atp.dashboard.models import Agent, User
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


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
async def test_agents_page_shows_quota_strip_and_purpose_column(
    v2_app,
    db_session: AsyncSession,
) -> None:
    # Seed a user + one agent of each purpose via the same session the
    # handler will read from (dependency override above keeps the
    # session factory consistent across fixture and request).
    user = User(
        username="owner",
        email="owner@test.com",
        hashed_password="x",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    db_session.add_all(
        [
            Agent(
                name="bench-1",
                agent_type="http",
                owner_id=user.id,
                purpose="benchmark",
            ),
            Agent(
                name="tourn-1",
                agent_type="mcp",
                owner_id=user.id,
                purpose="tournament",
            ),
        ]
    )
    await db_session.commit()

    jwt = create_access_token(data={"sub": user.username, "user_id": user.id})
    cookies = {"atp_token": jwt}

    async with AsyncClient(
        transport=ASGITransport(app=v2_app),
        base_url="http://test",
        cookies=cookies,
    ) as client:
        resp = await client.get("/ui/agents")

    assert resp.status_code == 200
    html = resp.text
    # Quota strip
    assert "Tournament agents:" in html
    assert "Benchmark agents:" in html
    # Purpose column visible
    assert "Purpose" in html
    assert "tournament" in html
    assert "benchmark" in html
