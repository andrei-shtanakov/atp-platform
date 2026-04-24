"""Regression tests for /ui/agents/new — the form route that was missing."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database
from atp.dashboard.models import Agent, User
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


@pytest.fixture
async def owner_cookies(db_session: AsyncSession) -> tuple[User, dict[str, str]]:
    user = User(
        username="owner",
        email="o@t.com",
        hashed_password=get_password_hash("pass"),
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    jwt = create_access_token(data={"sub": user.username, "user_id": user.id})
    return user, {"atp_token": jwt}


class TestAgentNewForm:
    """GET /ui/agents/new renders, doesn't collide with /ui/agents/{id}."""

    @pytest.mark.anyio
    async def test_get_form_renders_for_authenticated_user(
        self, v2_app, owner_cookies
    ) -> None:
        user, cookies = owner_cookies
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.get("/ui/agents/new?purpose=tournament")
        # Pre-fix behaviour: 422 "unable to parse 'new' as integer" because
        # /ui/agents/{agent_id:int} caught the path. Post-fix: 200 + form.
        assert resp.status_code == 200, resp.text
        html = resp.text
        assert "<form" in html.lower()
        assert 'name="name"' in html
        assert 'value="tournament"' in html  # hidden input

    @pytest.mark.anyio
    async def test_get_form_defaults_to_benchmark(self, v2_app, owner_cookies) -> None:
        _, cookies = owner_cookies
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.get("/ui/agents/new")
        assert resp.status_code == 200
        assert 'value="benchmark"' in resp.text

    @pytest.mark.anyio
    async def test_unauthenticated_redirects_to_login(self, v2_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/ui/agents/new", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"].startswith("/ui/login")


class TestAgentNewSubmit:
    """POST /ui/agents/new creates an Agent and redirects to /ui/agents."""

    @pytest.mark.anyio
    async def test_post_creates_tournament_agent(
        self, v2_app, db_session: AsyncSession, owner_cookies
    ) -> None:
        user, cookies = owner_cookies
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/ui/agents/new",
                data={
                    "name": "qa-bot-1",
                    "agent_type": "mcp",
                    "purpose": "tournament",
                },
                follow_redirects=False,
            )
        assert resp.status_code == 303, resp.text
        assert resp.headers["location"] == "/ui/agents"

        from sqlalchemy import select

        agent = await db_session.scalar(
            select(Agent).where(Agent.owner_id == user.id, Agent.name == "qa-bot-1")
        )
        assert agent is not None
        assert agent.purpose == "tournament"
        assert agent.agent_type == "mcp"

    @pytest.mark.anyio
    async def test_post_without_name_renders_error(self, v2_app, owner_cookies) -> None:
        _, cookies = owner_cookies
        async with AsyncClient(
            transport=ASGITransport(app=v2_app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/ui/agents/new",
                data={"name": "", "agent_type": "mcp", "purpose": "tournament"},
            )
        assert resp.status_code == 400
        assert "Error" in resp.text
