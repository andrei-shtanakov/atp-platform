"""Integration tests for the tenant management API routes.

These tests verify the /api/v2/tenants endpoints for tenant
CRUD operations, quota management, and settings configuration.
All endpoints require admin privileges.
"""

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.tenancy.models import Tenant, TenantQuotas, TenantSettings
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def test_database():
    """Create and configure a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    # Create all tables including tenant table
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(
            lambda c: Tenant.__table__.create(c, checkfirst=True)  # type: ignore
        )
    # Set as global database so auth functions use it
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore


@pytest.fixture
async def async_session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async with test_database.session() as session:
        yield session


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes."""
    app = create_test_app(use_v2_routes=True)

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
async def admin_user(async_session: AsyncSession) -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=True,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def regular_user(async_session: AsyncSession) -> User:
    """Create a regular (non-admin) user for testing."""
    user = User(
        username="regular_test",
        email="regular@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=False,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Generate JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.username, "user_id": admin_user.id}
    )


@pytest.fixture
def regular_token(regular_user: User) -> str:
    """Generate JWT token for regular user."""
    return create_access_token(
        data={"sub": regular_user.username, "user_id": regular_user.id}
    )


@pytest.fixture
async def test_tenant(async_session: AsyncSession) -> Tenant:
    """Create a test tenant."""
    tenant = Tenant(
        id="test-tenant",
        name="Test Tenant",
        plan="pro",
        description="Test tenant for testing",
        quotas_json=TenantQuotas().model_dump(),
        settings_json=TenantSettings().model_dump(),
        schema_name="tenant_test_tenant",
        is_active=True,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        contact_email="test@tenant.com",
    )
    async_session.add(tenant)
    await async_session.commit()
    await async_session.refresh(tenant)
    return tenant


@pytest.fixture
async def multiple_tenants(async_session: AsyncSession) -> list[Tenant]:
    """Create multiple tenants for testing."""
    tenants = []
    for i in range(5):
        tenant = Tenant(
            id=f"tenant{i}",
            name=f"Tenant {i}",
            plan=["free", "pro", "enterprise"][i % 3],
            description=f"Test tenant {i}",
            quotas_json=TenantQuotas().model_dump(),
            settings_json=TenantSettings().model_dump(),
            schema_name=f"tenant_tenant{i}",
            is_active=i != 3,  # tenant3 is inactive
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        async_session.add(tenant)
        tenants.append(tenant)
    await async_session.commit()
    for tenant in tenants:
        await async_session.refresh(tenant)
    return tenants


class TestTenantListEndpoint:
    """Test GET /api/tenants endpoint."""

    @pytest.mark.anyio
    async def test_list_tenants_requires_auth(self, v2_app):
        """Test listing tenants requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/tenants")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_list_tenants_requires_admin(
        self, v2_app, regular_user, regular_token
    ):
        """Test listing tenants requires admin privileges."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants",
                headers={"Authorization": f"Bearer {regular_token}"},
            )
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_list_tenants_empty(self, v2_app, admin_user, admin_token):
        """Test listing tenants when none exist."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
            assert data["items"] == []

    @pytest.mark.anyio
    async def test_list_tenants_with_data(
        self, v2_app, admin_user, admin_token, multiple_tenants
    ):
        """Test listing tenants returns correct data."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            # By default, only active tenants are returned
            assert data["total"] == 4  # 5 total, 1 inactive

    @pytest.mark.anyio
    async def test_list_tenants_include_inactive(
        self, v2_app, admin_user, admin_token, multiple_tenants
    ):
        """Test listing tenants with inactive included."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants?active_only=false",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 5

    @pytest.mark.anyio
    async def test_list_tenants_filter_by_plan(
        self, v2_app, admin_user, admin_token, multiple_tenants
    ):
        """Test listing tenants filtered by plan."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants?plan=pro",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            for item in data["items"]:
                assert item["plan"] == "pro"

    @pytest.mark.anyio
    async def test_list_tenants_pagination(
        self, v2_app, admin_user, admin_token, multiple_tenants
    ):
        """Test listing tenants with pagination."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants?limit=2&offset=0",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 2
            assert data["limit"] == 2
            assert data["offset"] == 0


class TestTenantCreateEndpoint:
    """Test POST /api/tenants endpoint."""

    @pytest.mark.anyio
    async def test_create_tenant_requires_admin(
        self, v2_app, regular_user, regular_token
    ):
        """Test creating tenant requires admin privileges."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/tenants",
                json={"id": "new-tenant", "name": "New Tenant"},
                headers={"Authorization": f"Bearer {regular_token}"},
            )
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_create_tenant_success(self, v2_app, admin_user, admin_token):
        """Test creating a tenant successfully."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/tenants",
                json={
                    "id": "acme-corp",
                    "name": "Acme Corporation",
                    "plan": "enterprise",
                    "description": "Test description",
                    "contact_email": "admin@acme.com",
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "acme-corp"
            assert data["name"] == "Acme Corporation"
            assert data["plan"] == "enterprise"
            assert data["is_active"] is True

    @pytest.mark.anyio
    async def test_create_tenant_with_custom_quotas(
        self, v2_app, admin_user, admin_token
    ):
        """Test creating tenant with custom quotas."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/tenants",
                json={
                    "id": "acme2",
                    "name": "Acme 2",
                    "quotas": {
                        "max_tests_per_day": 500,
                        "max_parallel_runs": 20,
                        "max_storage_gb": 100.0,
                    },
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["quotas"]["max_tests_per_day"] == 500
            assert data["quotas"]["max_parallel_runs"] == 20

    @pytest.mark.anyio
    async def test_create_tenant_with_custom_settings(
        self, v2_app, admin_user, admin_token
    ):
        """Test creating tenant with custom settings."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/tenants",
                json={
                    "id": "acme3",
                    "name": "Acme 3",
                    "settings": {
                        "require_mfa": True,
                        "retention_days": 365,
                        "sso_enabled": True,
                    },
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["settings"]["require_mfa"] is True
            assert data["settings"]["retention_days"] == 365

    @pytest.mark.anyio
    async def test_create_tenant_duplicate(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test creating duplicate tenant fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/tenants",
                json={"id": "test-tenant", "name": "Duplicate"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 409

    @pytest.mark.anyio
    async def test_create_tenant_invalid_id(self, v2_app, admin_user, admin_token):
        """Test creating tenant with invalid ID fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # ID with uppercase
            response = await client.post(
                "/api/tenants",
                json={"id": "Invalid-ID", "name": "Test"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 422

    @pytest.mark.anyio
    async def test_create_tenant_reserved_id(self, v2_app, admin_user, admin_token):
        """Test creating tenant with reserved ID fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/tenants",
                json={"id": "admin", "name": "Admin Tenant"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400


class TestTenantGetEndpoint:
    """Test GET /api/tenants/{tenant_id} endpoint."""

    @pytest.mark.anyio
    async def test_get_tenant_success(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test getting a tenant by ID."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/tenants/{test_tenant.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == test_tenant.id
            assert data["name"] == test_tenant.name

    @pytest.mark.anyio
    async def test_get_tenant_not_found(self, v2_app, admin_user, admin_token):
        """Test getting non-existent tenant returns 404."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 404


class TestTenantUpdateEndpoint:
    """Test PUT /api/tenants/{tenant_id} endpoint."""

    @pytest.mark.anyio
    async def test_update_tenant_success(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test updating a tenant."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.put(
                f"/api/tenants/{test_tenant.id}",
                json={
                    "name": "Updated Tenant Name",
                    "plan": "enterprise",
                    "description": "Updated description",
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Tenant Name"
            assert data["plan"] == "enterprise"

    @pytest.mark.anyio
    async def test_update_tenant_not_found(self, v2_app, admin_user, admin_token):
        """Test updating non-existent tenant returns 404."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/tenants/nonexistent",
                json={"name": "New Name"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 404


class TestTenantDeleteEndpoint:
    """Test DELETE /api/tenants/{tenant_id} endpoint."""

    @pytest.mark.anyio
    async def test_delete_tenant_soft(
        self, v2_app, admin_user, admin_token, test_tenant, async_session
    ):
        """Test soft deleting a tenant."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                f"/api/tenants/{test_tenant.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 204

    @pytest.mark.anyio
    async def test_delete_tenant_hard(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test hard deleting a tenant."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                f"/api/tenants/{test_tenant.id}?hard_delete=true",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 204

    @pytest.mark.anyio
    async def test_delete_tenant_not_found(self, v2_app, admin_user, admin_token):
        """Test deleting non-existent tenant returns 404."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                "/api/tenants/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 404


class TestTenantQuotasEndpoint:
    """Test /api/tenants/{tenant_id}/quotas endpoints."""

    @pytest.mark.anyio
    async def test_get_tenant_quotas(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test getting tenant quotas."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/tenants/{test_tenant.id}/quotas",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "max_tests_per_day" in data
            assert "max_parallel_runs" in data

    @pytest.mark.anyio
    async def test_update_tenant_quotas(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test updating tenant quotas."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.put(
                f"/api/tenants/{test_tenant.id}/quotas",
                json={
                    "max_tests_per_day": 1000,
                    "max_parallel_runs": 50,
                    "max_storage_gb": 500.0,
                    "max_agents": 100,
                    "llm_budget_monthly": 1000.0,
                    "max_users": 50,
                    "max_suites": 200,
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["max_tests_per_day"] == 1000
            assert data["max_parallel_runs"] == 50


class TestTenantSettingsEndpoint:
    """Test /api/tenants/{tenant_id}/settings endpoints."""

    @pytest.mark.anyio
    async def test_get_tenant_settings(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test getting tenant settings."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/tenants/{test_tenant.id}/settings",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "default_timeout_seconds" in data
            assert "allow_external_agents" in data

    @pytest.mark.anyio
    async def test_update_tenant_settings(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test updating tenant settings."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.put(
                f"/api/tenants/{test_tenant.id}/settings",
                json={
                    "default_timeout_seconds": 600,
                    "require_mfa": True,
                    "sso_enabled": True,
                    "sso_provider": "okta",
                    "retention_days": 180,
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["default_timeout_seconds"] == 600
            assert data["require_mfa"] is True


class TestTenantUsageEndpoint:
    """Test /api/tenants/{tenant_id}/usage endpoint."""

    @pytest.mark.anyio
    async def test_get_tenant_usage(self, v2_app, admin_user, admin_token, test_tenant):
        """Test getting tenant usage statistics."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/tenants/{test_tenant.id}/usage",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "tenant" in data
            assert "usage" in data
            assert "quotas_exceeded" in data
            assert data["tenant"]["id"] == test_tenant.id


class TestTenantQuotaStatusEndpoint:
    """Test /api/tenants/{tenant_id}/quota-status endpoint."""

    @pytest.mark.anyio
    async def test_get_tenant_quota_status(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test getting tenant quota status."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/tenants/{test_tenant.id}/quota-status",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == test_tenant.id
            assert "checks" in data
            assert "any_exceeded" in data
            assert isinstance(data["checks"], list)


class TestTenantActivationEndpoints:
    """Test tenant activation/deactivation endpoints."""

    @pytest.mark.anyio
    async def test_deactivate_tenant(
        self, v2_app, admin_user, admin_token, test_tenant
    ):
        """Test deactivating a tenant."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/api/tenants/{test_tenant.id}/deactivate",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["is_active"] is False

    @pytest.mark.anyio
    async def test_activate_tenant(self, v2_app, admin_user, admin_token, test_tenant):
        """Test activating a deactivated tenant."""
        # First deactivate
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                f"/api/tenants/{test_tenant.id}/deactivate",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

            # Then activate
            response = await client.post(
                f"/api/tenants/{test_tenant.id}/activate",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["is_active"] is True
