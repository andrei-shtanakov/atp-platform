"""Test API token authentication through middleware."""

import os
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from atp.dashboard.auth import get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.tokens import APIToken, generate_api_token, hash_token
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def app_with_db():
    """Create app with in-memory DB and seed a user + API token."""
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
            username="testuser",
            email="test@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
        )
        session.add(user)
        await session.flush()

        raw_token = generate_api_token(agent_scoped=False)
        api_token = APIToken(
            user_id=user.id,
            name="test-token",
            token_prefix=raw_token[:12],
            token_hash=hash_token(raw_token),
        )
        session.add(api_token)
        await session.commit()

    yield app, raw_token, db
    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


class TestAPITokenMiddleware:
    @pytest.mark.anyio
    async def test_api_token_sets_user_id(self, app_with_db: tuple) -> None:
        """A valid API token authenticates and returns the user."""
        app, raw_token, _db = app_with_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {raw_token}"},
            )
            assert resp.status_code == 200
            assert resp.json()["username"] == "testuser"

    @pytest.mark.anyio
    async def test_revoked_token_rejected(self, app_with_db: tuple) -> None:
        """A revoked API token is rejected with 401."""
        app, raw_token, db = app_with_db
        async with db.session() as session:
            await session.execute(
                update(APIToken)
                .where(APIToken.token_hash == hash_token(raw_token))
                .values(revoked_at=datetime.now())
            )
            await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {raw_token}"},
            )
            assert resp.status_code == 401
