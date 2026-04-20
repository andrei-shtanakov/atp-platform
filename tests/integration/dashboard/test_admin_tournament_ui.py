"""Integration tests for /ui/admin/tournaments/*.

Self-contained: builds an app with in-memory DB, seeds an admin and a
regular user, and exposes JWT headers for both. Avoids collision with
other integration tests that rely on their own database setup.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def admin_ui_ctx() -> AsyncGenerator[dict, None]:
    """Yield an app + client + seeded admin/regular headers."""
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
            username="admin_ui_test",
            email="admin_ui@test.com",
            hashed_password=get_password_hash("pass"),
            is_admin=True,
            is_active=True,
        )
        regular = User(
            username="regular_ui_test",
            email="regular_ui@test.com",
            hashed_password=get_password_hash("pass"),
            is_admin=False,
            is_active=True,
        )
        session.add_all([admin, regular])
        await session.commit()
        await session.refresh(admin)
        await session.refresh(regular)

    admin_jwt = create_access_token(
        data={"sub": admin.username, "user_id": admin.id}, is_admin=True
    )
    regular_jwt = create_access_token(
        data={"sub": regular.username, "user_id": regular.id}, is_admin=False
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "admin_headers": {"Authorization": f"Bearer {admin_jwt}"},
            "regular_headers": {"Authorization": f"Bearer {regular_jwt}"},
            "admin_id": admin.id,
            "regular_id": regular.id,
            "db": db,
        }

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


@pytest.mark.anyio
async def test_admin_landing_rejects_anonymous(admin_ui_ctx):
    """No Authorization header → 401."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_admin_landing_rejects_non_admin(admin_ui_ctx):
    """Authenticated regular user → 403."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin", headers=admin_ui_ctx["regular_headers"])
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_landing_renders_for_admin(admin_ui_ctx):
    """Authenticated admin → 200 with admin content."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin", headers=admin_ui_ctx["admin_headers"])
    assert resp.status_code == 200
    assert "Admin" in resp.text
    assert "Tournaments" in resp.text


@pytest.mark.anyio
async def test_admin_tournaments_list_rejects_regular_user(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments", headers=admin_ui_ctx["regular_headers"]
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_tournaments_list_renders_for_admin(admin_ui_ctx):
    """Empty DB case: page still renders with a 'no tournaments yet' fallback."""
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments", headers=admin_ui_ctx["admin_headers"]
    )
    assert resp.status_code == 200
    assert "Tournaments (admin)" in resp.text
    assert "New tournament" in resp.text
    assert "No tournaments yet" in resp.text


@pytest.mark.anyio
async def test_admin_new_tournament_form_renders(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments/new", headers=admin_ui_ctx["admin_headers"]
    )
    assert resp.status_code == 200
    assert 'name="name"' in resp.text
    assert 'name="game_type"' in resp.text
    assert 'name="num_players"' in resp.text
    assert 'name="total_rounds"' in resp.text
    assert 'name="round_deadline_s"' in resp.text
    assert "el_farol" in resp.text


@pytest.mark.anyio
async def test_admin_create_tournament_redirects_to_detail(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "name": "El Farol smoke A",
            "game_type": "el_farol",
            "num_players": "6",
            "total_rounds": "10",
            "round_deadline_s": "30",
        },
        headers=admin_ui_ctx["admin_headers"],
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/ui/admin/tournaments/")


@pytest.mark.anyio
async def test_admin_create_tournament_rejects_invalid_input(admin_ui_ctx):
    """El Farol requires 2 <= num_players <= 20; 100 must 400."""
    client = admin_ui_ctx["client"]
    resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "name": "El Farol too big",
            "game_type": "el_farol",
            "num_players": "100",
            "total_rounds": "10",
            "round_deadline_s": "30",
        },
        headers=admin_ui_ctx["admin_headers"],
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert (
        "Could not create" in resp.text
        or "Validation" in resp.text
        or "el_farol" in resp.text
    )
