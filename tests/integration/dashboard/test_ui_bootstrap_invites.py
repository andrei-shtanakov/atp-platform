"""Regression tests for /ui/setup (first-admin bootstrap) and /ui/invites UI."""

import os

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.tokens import Invite
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def empty_app():
    """App with an in-memory DB that has no users yet."""
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

    yield app, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


@pytest.fixture
async def admin_app():
    """App with an existing admin + cookie."""
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
        admin = User(
            username="root",
            email="root@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
            is_admin=True,
        )
        session.add(admin)
        await session.commit()
        await session.refresh(admin)

    jwt = create_access_token(data={"sub": admin.username, "user_id": admin.id})
    cookies = {"atp_token": jwt}

    yield app, cookies, admin, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


@pytest.mark.anyio
async def test_login_redirects_to_setup_when_empty(empty_app: tuple) -> None:
    app, _ = empty_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/login")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/setup"


@pytest.mark.anyio
async def test_register_redirects_to_setup_when_empty(empty_app: tuple) -> None:
    app, _ = empty_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/register")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/setup"


@pytest.mark.anyio
async def test_setup_renders_form_when_empty(empty_app: tuple) -> None:
    app, _ = empty_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/setup")
    assert resp.status_code == 200
    assert 'name="username"' in resp.text
    assert 'name="password_confirm"' in resp.text


@pytest.mark.anyio
async def test_setup_creates_first_admin(empty_app: tuple) -> None:
    app, db = empty_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ui/setup",
            data={
                "username": "founder",
                "email": "founder@test.com",
                "password": "supersecret",
                "password_confirm": "supersecret",
            },
        )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/login"

    async with db.session() as session:
        user = (
            await session.execute(select(User).where(User.username == "founder"))
        ).scalar_one()
    assert user.is_admin is True


@pytest.mark.anyio
async def test_setup_rejects_password_mismatch(empty_app: tuple) -> None:
    app, db = empty_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ui/setup",
            data={
                "username": "x",
                "email": "x@test.com",
                "password": "password1",
                "password_confirm": "password2",
            },
        )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/ui/setup?error=")

    async with db.session() as session:
        count = (await session.execute(select(User))).scalars().all()
    assert count == []


@pytest.mark.anyio
async def test_setup_redirects_once_user_exists(admin_app: tuple) -> None:
    app, _cookies, _admin, _db = admin_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/setup")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/login"


@pytest.mark.anyio
async def test_ui_create_invite_happy_path(admin_app: tuple) -> None:
    app, cookies, _admin, db = admin_app
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post("/ui/invites", data={"expires_in_days": "14"})
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/ui/invites?new_code=")

    async with db.session() as session:
        invites = (await session.execute(select(Invite))).scalars().all()
    assert len(invites) == 1
    assert invites[0].max_uses == 1


@pytest.mark.anyio
async def test_ui_create_invite_requires_admin(admin_app: tuple) -> None:
    app, _admin_cookies, _admin, db = admin_app

    async with db.session() as session:
        other = User(
            username="plain",
            email="plain@test.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
            is_admin=False,
        )
        session.add(other)
        await session.commit()
        await session.refresh(other)
    jwt = create_access_token(data={"sub": other.username, "user_id": other.id})

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies={"atp_token": jwt}
    ) as client:
        resp = await client.post("/ui/invites", data={})
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/login"

    async with db.session() as session:
        invites = (await session.execute(select(Invite))).scalars().all()
    assert invites == []


@pytest.mark.anyio
async def test_logout_clears_cookie_and_redirects(admin_app: tuple) -> None:
    """POST /ui/logout → 303 to /ui/login + Set-Cookie that clears atp_token."""
    app, cookies, _admin, _db = admin_app
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post("/ui/logout")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/login"

    set_cookie = resp.headers.get("set-cookie", "")
    assert "atp_token=" in set_cookie
    assert 'atp_token=""' in set_cookie or "Max-Age=0" in set_cookie
    assert "Path=/" in set_cookie
    assert "samesite=strict" in set_cookie.lower()


@pytest.mark.anyio
async def test_ui_deactivate_invite(admin_app: tuple) -> None:
    app, cookies, _admin, db = admin_app
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        await client.post("/ui/invites", data={})

    async with db.session() as session:
        inv = (await session.execute(select(Invite))).scalar_one()
    assert inv.use_count == 0
    assert inv.max_uses == 1

    async with AsyncClient(
        transport=transport, base_url="http://test", cookies=cookies
    ) as client:
        resp = await client.post(f"/ui/invites/{inv.id}/deactivate")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/invites"

    async with db.session() as session:
        fresh = (
            await session.execute(select(Invite).where(Invite.id == inv.id))
        ).scalar_one()
    assert fresh.max_uses == fresh.use_count
