"""Integration tests for the invite management API endpoints."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.rbac import Role, RolePermission, UserRole
from atp.dashboard.tenancy.models import Tenant
from atp.dashboard.tokens import Invite
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app

_INVITE_URL = "/api/v1/invites"
_REGISTER_URL = "/api/auth/register"


@pytest.fixture
async def app_invite_mode():
    """Create app in invite registration mode with in-memory DB and admin user."""
    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "false"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    os.environ["ATP_REGISTRATION_MODE"] = "invite"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(
            lambda c: Tenant.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: Role.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: RolePermission.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: UserRole.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: Invite.__table__.create(c, checkfirst=True)  # type: ignore
        )
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
        registration_mode="invite",
    )
    app = create_app(config=config)

    async with db.session() as session:
        admin = User(
            username="admin",
            email="admin@test.com",
            hashed_password=get_password_hash("adminpass"),
            is_admin=True,
            is_active=True,
        )
        session.add(admin)
        await session.commit()
        await session.refresh(admin)

    jwt = create_access_token(data={"sub": admin.username, "user_id": admin.id})
    admin_headers = {"Authorization": f"Bearer {jwt}"}

    yield app, admin_headers, admin, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()
    os.environ.pop("ATP_REGISTRATION_MODE", None)


class TestCreateInvite:
    @pytest.mark.anyio
    async def test_create_invite_returns_201_with_code(
        self, app_invite_mode: tuple
    ) -> None:
        """Create invite → 201, code starts with atp_inv_."""
        app, headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _INVITE_URL,
                json={"expires_in_days": 7},
                headers=headers,
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert "code" in data
        assert data["code"].startswith("atp_inv_")
        assert data["use_count"] == 0
        assert data["max_uses"] == 1

    @pytest.mark.anyio
    async def test_create_invite_requires_admin(self, app_invite_mode: tuple) -> None:
        """Non-admin user cannot create invite → 403."""
        app, _admin_headers, _admin, db = app_invite_mode
        async with db.session() as session:
            regular = User(
                username="regular",
                email="regular@test.com",
                hashed_password=get_password_hash("pass"),
                is_admin=False,
                is_active=True,
            )
            session.add(regular)
            await session.commit()
            await session.refresh(regular)

        jwt = create_access_token(data={"sub": regular.username, "user_id": regular.id})
        regular_headers = {"Authorization": f"Bearer {jwt}"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _INVITE_URL,
                json={"expires_in_days": 7},
                headers=regular_headers,
            )
        assert resp.status_code == 403, resp.text


class TestListInvites:
    @pytest.mark.anyio
    async def test_list_invites_returns_200(self, app_invite_mode: tuple) -> None:
        """List invites → 200, returns list."""
        app, headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(_INVITE_URL, json={"expires_in_days": 3}, headers=headers)
            resp = await client.get(_INVITE_URL, headers=headers)
        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert isinstance(items, list)
        assert len(items) >= 1


class TestDeactivateInvite:
    @pytest.mark.anyio
    async def test_deactivate_invite(self, app_invite_mode: tuple) -> None:
        """Deactivate invite → 200, max_uses equals use_count."""
        app, headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                _INVITE_URL, json={"expires_in_days": 7}, headers=headers
            )
            assert create_resp.status_code == 201
            invite_id = create_resp.json()["id"]

            deactivate_resp = await client.delete(
                f"{_INVITE_URL}/{invite_id}", headers=headers
            )
        assert deactivate_resp.status_code == 200, deactivate_resp.text
        data = deactivate_resp.json()
        assert data["max_uses"] == data["use_count"]

    @pytest.mark.anyio
    async def test_deactivate_nonexistent_invite_404(
        self, app_invite_mode: tuple
    ) -> None:
        """Deactivate non-existent invite → 404."""
        app, headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(f"{_INVITE_URL}/99999", headers=headers)
        assert resp.status_code == 404, resp.text


class TestInviteGatedRegistration:
    @pytest.mark.anyio
    async def test_register_with_valid_invite_returns_201(
        self, app_invite_mode: tuple
    ) -> None:
        """Register with valid invite → 201."""
        app, admin_headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Admin creates invite
            invite_resp = await client.post(
                _INVITE_URL,
                json={"expires_in_days": 7},
                headers=admin_headers,
            )
            assert invite_resp.status_code == 201
            code = invite_resp.json()["code"]

            # New user registers with that invite
            reg_resp = await client.post(
                _REGISTER_URL,
                json={
                    "username": "newuser",
                    "email": "new@test.com",
                    "password": "securepass",
                    "invite_code": code,
                },
            )
        assert reg_resp.status_code == 201, reg_resp.text
        data = reg_resp.json()
        assert data["username"] == "newuser"

    @pytest.mark.anyio
    async def test_register_without_invite_in_invite_mode_400(
        self, app_invite_mode: tuple
    ) -> None:
        """Register without invite code in invite mode → 400 'Invite code required'."""
        app, _admin_headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _REGISTER_URL,
                json={
                    "username": "noinvite",
                    "email": "noinvite@test.com",
                    "password": "securepass",
                },
            )
        assert resp.status_code == 400, resp.text
        assert "Invite code required" in resp.text

    @pytest.mark.anyio
    async def test_register_with_invalid_invite_400(
        self, app_invite_mode: tuple
    ) -> None:
        """Register with invalid invite code → 400."""
        app, _admin_headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                _REGISTER_URL,
                json={
                    "username": "badinvite",
                    "email": "bad@test.com",
                    "password": "securepass",
                    "invite_code": "atp_inv_notreal",
                },
            )
        assert resp.status_code == 400, resp.text
        assert "Invalid or expired invite code" in resp.text

    @pytest.mark.anyio
    async def test_invite_use_count_increments_after_registration(
        self, app_invite_mode: tuple
    ) -> None:
        """After successful registration, invite use_count is incremented."""
        app, admin_headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invite_resp = await client.post(
                _INVITE_URL,
                json={"expires_in_days": 7},
                headers=admin_headers,
            )
            code = invite_resp.json()["code"]
            invite_id = invite_resp.json()["id"]

            await client.post(
                _REGISTER_URL,
                json={
                    "username": "usedcode",
                    "email": "used@test.com",
                    "password": "securepass",
                    "invite_code": code,
                },
            )

            # List to check use_count
            list_resp = await client.get(_INVITE_URL, headers=admin_headers)
        invites = list_resp.json()
        used_invite = next((i for i in invites if i["id"] == invite_id), None)
        assert used_invite is not None
        assert used_invite["use_count"] == 1

    @pytest.mark.anyio
    async def test_invite_cannot_be_reused_after_exhausted(
        self, app_invite_mode: tuple
    ) -> None:
        """Single-use invite cannot be used twice → 400 on second attempt."""
        app, admin_headers, _admin, _db = app_invite_mode
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invite_resp = await client.post(
                _INVITE_URL,
                json={"expires_in_days": 7},
                headers=admin_headers,
            )
            code = invite_resp.json()["code"]

            # First use succeeds
            await client.post(
                _REGISTER_URL,
                json={
                    "username": "firstuse",
                    "email": "first@test.com",
                    "password": "securepass",
                    "invite_code": code,
                },
            )

            # Second use fails
            second_resp = await client.post(
                _REGISTER_URL,
                json={
                    "username": "seconduse",
                    "email": "second@test.com",
                    "password": "securepass",
                    "invite_code": code,
                },
            )
        assert second_resp.status_code == 400, second_resp.text
