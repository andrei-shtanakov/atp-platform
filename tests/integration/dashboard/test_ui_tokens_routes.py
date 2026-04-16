"""Regression tests for the cookie-authenticated /ui/tokens form.

Mirror of test_ui_agents_routes — validates that the UI form for
creating user-level (``atp_u_``) tokens exists, works, and surfaces
errors without falling back to curl. Companion of PR #34's About
quickstart: users who follow the docs must not need a terminal to
mint their first token.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.tokens import APIToken
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
            username="tokenowner",
            email="tokenowner@test.com",
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
async def test_ui_post_tokens_creates_user_level_token(app_with_cookie) -> None:
    """POST /ui/tokens with form body mints a user-level token and
    returns the raw value once."""
    app, cookies, user, db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post(
            "/ui/tokens",
            data={"name": "my-cli", "expires_in_days": "7"},
        )

    assert resp.status_code == 201, resp.text
    body = resp.text
    # Raw token (shown once) must contain the user-level prefix.
    assert "atp_u_" in body, "raw user-level token not surfaced in response"

    async with db.session() as session:
        row = (
            await session.execute(
                select(APIToken).where(
                    APIToken.user_id == user.id,
                    APIToken.name == "my-cli",
                )
            )
        ).scalar_one()
        assert row.agent_id is None, "user-level token must have agent_id=NULL"
        assert row.token_prefix.startswith("atp_u_")


@pytest.mark.anyio
async def test_ui_post_tokens_rejects_non_int_expiry(app_with_cookie) -> None:
    """Non-numeric expires_in_days shows an error, no token written."""
    app, cookies, user, db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post(
            "/ui/tokens",
            data={"name": "bad", "expires_in_days": "seven"},
        )

    assert resp.status_code == 400
    assert "whole number" in resp.text.lower()

    async with db.session() as session:
        rows = (
            (await session.execute(select(APIToken).where(APIToken.user_id == user.id)))
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.anyio
async def test_ui_post_tokens_anonymous_redirects_to_login(app_with_cookie) -> None:
    """Without the cookie the handler redirects to login."""
    app, _cookies, _user, _db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ui/tokens",
            data={"name": "x"},
            follow_redirects=False,
        )
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/login"


@pytest.mark.anyio
async def test_ui_get_tokens_shows_create_form(app_with_cookie) -> None:
    """GET /ui/tokens must render the 'Create user-level token' form.

    Anti-regression: if a template refactor accidentally drops the form
    we want a loud failure, because the public About quickstart assumes
    this UI exists.
    """
    app, cookies, _user, _db = app_with_cookie
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.get("/ui/tokens")

    assert resp.status_code == 200
    assert "Create user-level token" in resp.text
    assert 'action="/ui/tokens"' in resp.text
    assert 'name="name"' in resp.text
