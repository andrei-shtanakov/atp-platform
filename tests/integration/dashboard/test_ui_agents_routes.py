"""Regression tests for the cookie-authenticated /ui/agents form endpoint.

The /api/v1/agents endpoint expects a JSON body, but the 'New Agent' form on
the dashboard submits application/x-www-form-urlencoded. A dedicated UI
handler translates form fields into the shared create_agent_for_user helper.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User
from atp.dashboard.tokens import APIToken
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def app_with_cookie():
    """App + cookie-authenticated client state for UI tests."""
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
            username="uiowner",
            email="uiowner@test.com",
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
async def test_ui_post_agents_form_creates_agent(app_with_cookie: tuple) -> None:
    """POST /ui/agents with form-encoded body creates an agent and redirects."""
    app, cookies, user, db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post(
            "/ui/agents",
            data={
                "name": "AlexSDK",
                "version": "latest",
                "agent_type": "mcp",
                "description": "Test Agent MCP Communication",
            },
        )

    assert resp.status_code == 303, resp.text
    assert resp.headers["location"] == "/ui/agents"

    async with db.session() as session:
        result = await session.execute(
            select(Agent).where(Agent.owner_id == user.id, Agent.name == "AlexSDK")
        )
        agent = result.scalar_one()
    assert agent.agent_type == "mcp"
    assert agent.version == "latest"
    assert agent.description == "Test Agent MCP Communication"


@pytest.mark.anyio
async def test_ui_post_agents_duplicate_redirects_with_error(
    app_with_cookie: tuple,
) -> None:
    """Submitting the same name+version twice redirects back with ?error=."""
    app, cookies, _user, _db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        first = await client.post(
            "/ui/agents",
            data={"name": "Dup", "version": "v1", "agent_type": "cli"},
        )
        assert first.status_code == 303

        second = await client.post(
            "/ui/agents",
            data={"name": "Dup", "version": "v1", "agent_type": "cli"},
        )

    assert second.status_code == 303
    assert second.headers["location"].startswith("/ui/agents?error=")


@pytest.mark.anyio
async def test_ui_post_agents_anonymous_redirects_to_login(
    app_with_cookie: tuple,
) -> None:
    """No cookie → redirect to /ui/login (never leaks a 401 JSON body)."""
    app, _cookies, _user, _db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ui/agents",
            data={"name": "Anon", "agent_type": "cli"},
        )
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/login"


@pytest.mark.anyio
async def test_ui_create_agent_token_shows_raw_once(
    app_with_cookie: tuple,
) -> None:
    """POST /ui/agents/{id}/tokens returns 201 with raw token in body."""
    app, cookies, user, db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        create = await client.post(
            "/ui/agents",
            data={"name": "TokHost", "agent_type": "cli"},
        )
        assert create.status_code == 303

    async with db.session() as session:
        result = await session.execute(
            select(Agent).where(Agent.owner_id == user.id, Agent.name == "TokHost")
        )
        agent = result.scalar_one()

    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post(
            f"/ui/agents/{agent.id}/tokens",
            data={"name": "ci-bot"},
        )

    assert resp.status_code == 201, resp.text
    body = resp.text
    assert "Token created" in body
    assert "atp_a_" in body

    async with db.session() as session:
        stored = await session.execute(
            select(APIToken).where(APIToken.agent_id == agent.id)
        )
        tok = stored.scalar_one()
    assert tok.name == "ci-bot"
    assert tok.token_prefix.startswith("atp_a_")


@pytest.mark.anyio
async def test_ui_revoke_agent_token_redirects_back(
    app_with_cookie: tuple,
) -> None:
    """POST /ui/tokens/{id}/revoke redirects to owning agent and marks revoked."""
    app, cookies, user, db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        await client.post("/ui/agents", data={"name": "RevHost", "agent_type": "cli"})

    async with db.session() as session:
        result = await session.execute(
            select(Agent).where(Agent.owner_id == user.id, Agent.name == "RevHost")
        )
        agent = result.scalar_one()

    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        await client.post(f"/ui/agents/{agent.id}/tokens", data={"name": "ephemeral"})

    async with db.session() as session:
        tok = (
            await session.execute(select(APIToken).where(APIToken.agent_id == agent.id))
        ).scalar_one()

    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post(f"/ui/tokens/{tok.id}/revoke")

    assert resp.status_code == 303
    assert resp.headers["location"] == f"/ui/agents/{agent.id}"

    async with db.session() as session:
        fresh = (
            await session.execute(select(APIToken).where(APIToken.id == tok.id))
        ).scalar_one()
    assert fresh.revoked_at is not None
