"""Integration tests for the agent management API endpoints."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app

_AGENTS_URL = "/api/v1/agents"


@pytest.fixture
async def app_with_db():
    """Create app with in-memory DB, a user, and JWT auth headers."""
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
            username="agentowner",
            email="agentowner@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    jwt = create_access_token(data={"sub": user.username, "user_id": user.id})
    headers = {"Authorization": f"Bearer {jwt}"}

    yield app, headers, user, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


class TestCreateAgent:
    @pytest.mark.anyio
    async def test_create_agent_201_with_owner_id(self, app_with_db: tuple) -> None:
        """Create agent → 201, has owner_id matching the requesting user."""
        app, headers, user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _AGENTS_URL,
                json={"name": "my-agent", "agent_type": "cli", "version": "1.0"},
                headers=headers,
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["name"] == "my-agent"
        assert data["version"] == "1.0"
        assert data["agent_type"] == "cli"
        assert data["owner_id"] == user.id

    @pytest.mark.anyio
    async def test_create_agent_default_version(self, app_with_db: tuple) -> None:
        """Create agent without version → defaults to 'latest'."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _AGENTS_URL,
                json={"name": "default-agent", "agent_type": "http"},
                headers=headers,
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["version"] == "latest"

    @pytest.mark.anyio
    async def test_duplicate_name_version_409(self, app_with_db: tuple) -> None:
        """Create same name+version twice → second returns 409."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post(
                _AGENTS_URL,
                json={"name": "dupe-agent", "agent_type": "cli", "version": "v1"},
                headers=headers,
            )
            assert first.status_code == 201, first.text

            second = await client.post(
                _AGENTS_URL,
                json={"name": "dupe-agent", "agent_type": "cli", "version": "v1"},
                headers=headers,
            )
        assert second.status_code == 409, second.text


class TestListAgents:
    @pytest.mark.anyio
    async def test_list_agents_returns_created(self, app_with_db: tuple) -> None:
        """List agents → returns previously created agents."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                _AGENTS_URL,
                json={"name": "list-agent-1", "agent_type": "cli"},
                headers=headers,
            )
            await client.post(
                _AGENTS_URL,
                json={"name": "list-agent-2", "agent_type": "http"},
                headers=headers,
            )
            resp = await client.get(_AGENTS_URL, headers=headers)

        assert resp.status_code == 200, resp.text
        items = resp.json()
        names = [item["name"] for item in items]
        assert "list-agent-1" in names
        assert "list-agent-2" in names

    @pytest.mark.anyio
    async def test_list_agents_excludes_deleted(self, app_with_db: tuple) -> None:
        """Soft-deleted agents do not appear in list."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _AGENTS_URL,
                json={"name": "to-delete-agent", "agent_type": "cli"},
                headers=headers,
            )
            agent_id = create_resp.json()["id"]
            await client.delete(f"{_AGENTS_URL}/{agent_id}", headers=headers)
            resp = await client.get(_AGENTS_URL, headers=headers)

        items = resp.json()
        ids = [item["id"] for item in items]
        assert agent_id not in ids


class TestSoftDelete:
    @pytest.mark.anyio
    async def test_soft_delete_returns_200(self, app_with_db: tuple) -> None:
        """Soft delete → 200, deleted_at is not None in DB (agent gone from list)."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _AGENTS_URL,
                json={"name": "soft-delete-agent", "agent_type": "cli"},
                headers=headers,
            )
            assert create_resp.status_code == 201
            agent_id = create_resp.json()["id"]

            del_resp = await client.delete(f"{_AGENTS_URL}/{agent_id}", headers=headers)

        assert del_resp.status_code == 200, del_resp.text

    @pytest.mark.anyio
    async def test_double_delete_404(self, app_with_db: tuple) -> None:
        """Deleting already-deleted agent → 404."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _AGENTS_URL,
                json={"name": "double-del-agent", "agent_type": "cli"},
                headers=headers,
            )
            agent_id = create_resp.json()["id"]
            await client.delete(f"{_AGENTS_URL}/{agent_id}", headers=headers)
            second_del = await client.delete(
                f"{_AGENTS_URL}/{agent_id}", headers=headers
            )

        assert second_del.status_code == 404


class TestAgentLimit:
    @pytest.mark.anyio
    async def test_agent_limit_enforced(self, app_with_db: tuple) -> None:
        """10 agents OK, 11th returns 409 (default max_agents_per_user=10)."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(10):
                resp = await client.post(
                    _AGENTS_URL,
                    json={"name": f"limit-agent-{i}", "agent_type": "cli"},
                    headers=headers,
                )
                assert resp.status_code == 201, f"Agent {i} failed: {resp.text}"

            eleventh = await client.post(
                _AGENTS_URL,
                json={"name": "limit-agent-overflow", "agent_type": "cli"},
                headers=headers,
            )
        assert eleventh.status_code == 409, eleventh.text


class TestGetAndUpdateAgent:
    @pytest.mark.anyio
    async def test_get_agent(self, app_with_db: tuple) -> None:
        """Get agent by id → 200 with correct data."""
        app, headers, user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _AGENTS_URL,
                json={"name": "get-me", "agent_type": "cli", "version": "v2"},
                headers=headers,
            )
            agent_id = create_resp.json()["id"]
            resp = await client.get(f"{_AGENTS_URL}/{agent_id}", headers=headers)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["id"] == agent_id
        assert data["name"] == "get-me"
        assert data["owner_id"] == user.id

    @pytest.mark.anyio
    async def test_get_nonexistent_agent_404(self, app_with_db: tuple) -> None:
        """Get non-existent agent → 404."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"{_AGENTS_URL}/999999", headers=headers)
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_update_agent_description(self, app_with_db: tuple) -> None:
        """Patch agent description → 200, updated."""
        app, headers, _user, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _AGENTS_URL,
                json={"name": "patchable", "agent_type": "cli"},
                headers=headers,
            )
            agent_id = create_resp.json()["id"]
            patch_resp = await client.patch(
                f"{_AGENTS_URL}/{agent_id}",
                json={"description": "Updated desc"},
                headers=headers,
            )

        assert patch_resp.status_code == 200, patch_resp.text
        assert patch_resp.json()["description"] == "Updated desc"
