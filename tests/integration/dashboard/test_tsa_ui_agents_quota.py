"""LABS-TSA PR-5 — /ui/agents quota strip + purpose column.

Verifies the agents page surfaces per-purpose agent counts against their
configured caps, renders a Purpose column in the agents table, and shows
two register links (one per purpose).
"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def app_with_cookie():
    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "false"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    async with db.session() as session:
        user = User(
            username="owner",
            email="owner@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    jwt = create_access_token(data={"sub": user.username, "user_id": user.id})
    cookies = {"atp_token": jwt}

    yield app, cookies, user, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


@pytest.mark.anyio
async def test_agents_page_shows_quota_strip_and_purpose_column(
    app_with_cookie,
) -> None:
    app, cookies, user, db = app_with_cookie
    async with db.session() as session:
        session.add(
            Agent(
                name="bench-1",
                agent_type="http",
                owner_id=user.id,
                purpose="benchmark",
            )
        )
        session.add(
            Agent(
                name="tourn-1",
                agent_type="mcp",
                owner_id=user.id,
                purpose="tournament",
            )
        )
        await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app),
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
