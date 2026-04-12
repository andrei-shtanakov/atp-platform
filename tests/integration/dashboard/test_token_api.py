"""Integration tests for the token management API endpoints."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app

_TOKEN_URL = "/api/v1/tokens"


@pytest.fixture
async def app_with_db():
    """Create app with in-memory DB, a user, and an agent owned by that user."""
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
            username="tokenuser",
            email="tokenuser@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.flush()

        agent = Agent(
            name="test-agent",
            agent_type="cli",
            owner_id=user.id,
        )
        session.add(agent)
        await session.commit()
        await session.refresh(user)
        await session.refresh(agent)

    jwt = create_access_token(data={"sub": user.username, "user_id": user.id})
    headers = {"Authorization": f"Bearer {jwt}"}

    yield app, headers, user, agent, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


class TestCreateToken:
    @pytest.mark.anyio
    async def test_create_user_level_token(self, app_with_db: tuple) -> None:
        """Create a user-level token → 201, starts with atp_u_, length 38."""
        app, headers, _user, _agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _TOKEN_URL,
                json={"name": "ci-runner", "expires_in_days": 30},
                headers=headers,
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert "token" in data
        assert data["token"].startswith("atp_u_")
        assert len(data["token"]) == 38
        assert data["agent_id"] is None
        assert data["name"] == "ci-runner"

    @pytest.mark.anyio
    async def test_create_agent_scoped_token(self, app_with_db: tuple) -> None:
        """Create an agent-scoped token → 201, starts with atp_a_, has agent_id."""
        app, headers, _user, agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _TOKEN_URL,
                json={"name": "agent-token", "agent_id": agent.id},
                headers=headers,
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["token"].startswith("atp_a_")
        assert len(data["token"]) == 38
        assert data["agent_id"] == agent.id


class TestListTokens:
    @pytest.mark.anyio
    async def test_list_tokens_no_raw_token(self, app_with_db: tuple) -> None:
        """List tokens → 200, no raw token field in response."""
        app, headers, _user, _agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a token first
            await client.post(
                _TOKEN_URL,
                json={"name": "list-test"},
                headers=headers,
            )
            resp = await client.get(_TOKEN_URL, headers=headers)

        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert len(items) >= 1
        for item in items:
            assert "token" not in item
            assert "token_prefix" in item


class TestRevokeToken:
    @pytest.mark.anyio
    async def test_revoke_token(self, app_with_db: tuple) -> None:
        """Revoke token → 200, revoked_at is set."""
        app, headers, _user, _agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _TOKEN_URL,
                json={"name": "to-revoke"},
                headers=headers,
            )
            assert create_resp.status_code == 201
            token_id = create_resp.json()["id"]

            revoke_resp = await client.delete(
                f"{_TOKEN_URL}/{token_id}", headers=headers
            )

        assert revoke_resp.status_code == 200, revoke_resp.text
        data = revoke_resp.json()
        assert data["revoked_at"] is not None
        assert "token" not in data

    @pytest.mark.anyio
    async def test_revoke_already_revoked_token_409(self, app_with_db: tuple) -> None:
        """Revoking an already-revoked token → 409."""
        app, headers, _user, _agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _TOKEN_URL,
                json={"name": "double-revoke"},
                headers=headers,
            )
            token_id = create_resp.json()["id"]
            await client.delete(f"{_TOKEN_URL}/{token_id}", headers=headers)
            second_resp = await client.delete(
                f"{_TOKEN_URL}/{token_id}", headers=headers
            )

        assert second_resp.status_code == 409


class TestTokenLimits:
    @pytest.mark.anyio
    async def test_user_token_limit_enforced(self, app_with_db: tuple) -> None:
        """5 user tokens OK, 6th returns 409 (default max_user_tokens=5)."""
        app, headers, _user, _agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(5):
                resp = await client.post(
                    _TOKEN_URL,
                    json={"name": f"token-{i}"},
                    headers=headers,
                )
                assert resp.status_code == 201, f"Token {i} failed: {resp.text}"

            sixth_resp = await client.post(
                _TOKEN_URL,
                json={"name": "token-overflow"},
                headers=headers,
            )

        assert sixth_resp.status_code == 409, sixth_resp.text

    @pytest.mark.anyio
    async def test_agent_token_limit_enforced(self, app_with_db: tuple) -> None:
        """3 agent tokens OK, 4th returns 409 (default max_tokens_per_agent=3)."""
        app, headers, _user, agent, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(3):
                resp = await client.post(
                    _TOKEN_URL,
                    json={"name": f"agent-token-{i}", "agent_id": agent.id},
                    headers=headers,
                )
                assert resp.status_code == 201, f"Agent token {i} failed: {resp.text}"

            fourth_resp = await client.post(
                _TOKEN_URL,
                json={"name": "agent-token-overflow", "agent_id": agent.id},
                headers=headers,
            )

        assert fourth_resp.status_code == 409, fourth_resp.text
