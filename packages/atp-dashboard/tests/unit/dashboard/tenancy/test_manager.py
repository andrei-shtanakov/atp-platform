"""Tests for TenantManager."""

from pathlib import Path

import pytest

from atp.dashboard.tenancy.manager import (
    TenantDeleteError,
    TenantExistsError,
    TenantManager,
    TenantNotFoundError,
)
from atp.dashboard.tenancy.models import (
    DEFAULT_TENANT_ID,
    Tenant,
    TenantQuotas,
    TenantSettings,
)


class TestTenantManagerValidation:
    """Tests for TenantManager validation methods."""

    @pytest.fixture
    def manager(self, temp_db_path: str) -> TenantManager:
        """Create a TenantManager with a test engine."""
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(temp_db_path)
        return TenantManager(engine)

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    def test_validate_tenant_id_valid(self, manager: TenantManager) -> None:
        """Test validation of valid tenant IDs."""
        valid_ids = [
            "a",
            "ab",
            "acme",
            "acme-corp",
            "acme123",
            "a1b2c3",
            "my-tenant-123",
        ]
        for tenant_id in valid_ids:
            # Should not raise
            manager._validate_tenant_id(tenant_id)

    def test_validate_tenant_id_empty(self, manager: TenantManager) -> None:
        """Test validation rejects empty tenant ID."""
        with pytest.raises(ValueError, match="cannot be empty"):
            manager._validate_tenant_id("")

    def test_validate_tenant_id_too_long(self, manager: TenantManager) -> None:
        """Test validation rejects too long tenant ID."""
        with pytest.raises(ValueError, match="50 characters or less"):
            manager._validate_tenant_id("a" * 51)

    def test_validate_tenant_id_reserved(self, manager: TenantManager) -> None:
        """Test validation rejects reserved tenant IDs."""
        reserved_ids = ["default", "public", "admin", "system", "root", "master"]
        for tenant_id in reserved_ids:
            with pytest.raises(ValueError, match="reserved"):
                manager._validate_tenant_id(tenant_id)

    def test_validate_tenant_id_invalid_format(self, manager: TenantManager) -> None:
        """Test validation rejects invalid format."""
        invalid_ids = [
            "-acme",  # starts with hyphen
            "acme-",  # ends with hyphen
            "Acme",  # uppercase
            "acme corp",  # space
            "acme_corp",  # underscore
            "acme.corp",  # period
            "acme@corp",  # special char
        ]
        for tenant_id in invalid_ids:
            with pytest.raises(ValueError, match="invalid"):
                manager._validate_tenant_id(tenant_id)

    def test_get_schema_name(self, manager: TenantManager) -> None:
        """Test schema name generation."""
        assert manager._get_schema_name("default") == "public"
        assert manager._get_schema_name("acme") == "tenant_acme"
        assert manager._get_schema_name("acme-corp") == "tenant_acme_corp"


class TestTenantManagerOperations:
    """Tests for TenantManager CRUD operations."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    @pytest.fixture
    async def engine(self, temp_db_path: str):
        """Create a test engine with tables."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atp.dashboard.models import Base

        engine = create_async_engine(temp_db_path)

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda c: Tenant.__table__.create(c, checkfirst=True))

        yield engine

        await engine.dispose()

    @pytest.fixture
    def manager(self, engine) -> TenantManager:
        """Create a TenantManager."""
        return TenantManager(engine)

    @pytest.mark.anyio
    async def test_create_tenant(self, manager: TenantManager) -> None:
        """Test creating a tenant."""
        tenant = await manager.create_tenant(
            "acme",
            "Acme Corporation",
            plan="enterprise",
            description="Test tenant",
        )
        assert tenant.id == "acme"
        assert tenant.name == "Acme Corporation"
        assert tenant.plan == "enterprise"
        assert tenant.description == "Test tenant"
        assert tenant.is_active is True
        assert tenant.schema_name == "tenant_acme"

    @pytest.mark.anyio
    async def test_create_tenant_with_quotas(self, manager: TenantManager) -> None:
        """Test creating a tenant with custom quotas."""
        quotas = TenantQuotas(
            max_tests_per_day=500,
            max_parallel_runs=20,
        )
        tenant = await manager.create_tenant(
            "acme2",
            "Acme 2",
            quotas=quotas,
        )
        assert tenant.quotas.max_tests_per_day == 500
        assert tenant.quotas.max_parallel_runs == 20

    @pytest.mark.anyio
    async def test_create_tenant_with_settings(self, manager: TenantManager) -> None:
        """Test creating a tenant with custom settings."""
        settings = TenantSettings(
            require_mfa=True,
            retention_days=365,
        )
        tenant = await manager.create_tenant(
            "acme3",
            "Acme 3",
            settings=settings,
        )
        assert tenant.settings.require_mfa is True
        assert tenant.settings.retention_days == 365

    @pytest.mark.anyio
    async def test_create_tenant_duplicate(self, manager: TenantManager) -> None:
        """Test creating a duplicate tenant fails."""
        await manager.create_tenant("acme", "Acme")

        with pytest.raises(TenantExistsError, match="already exists"):
            await manager.create_tenant("acme", "Another Acme")

    @pytest.mark.anyio
    async def test_get_tenant(self, manager: TenantManager) -> None:
        """Test getting a tenant by ID."""
        await manager.create_tenant("acme", "Acme")

        tenant = await manager.get_tenant("acme")
        assert tenant is not None
        assert tenant.id == "acme"
        assert tenant.name == "Acme"

    @pytest.mark.anyio
    async def test_get_tenant_not_found(self, manager: TenantManager) -> None:
        """Test getting a non-existent tenant."""
        tenant = await manager.get_tenant("nonexistent")
        assert tenant is None

    @pytest.mark.anyio
    async def test_list_tenants(self, manager: TenantManager) -> None:
        """Test listing tenants."""
        await manager.create_tenant("acme1", "Acme 1")
        await manager.create_tenant("acme2", "Acme 2", plan="pro")
        await manager.create_tenant("acme3", "Acme 3", plan="enterprise")

        tenants = await manager.list_tenants()
        assert len(tenants) == 3

    @pytest.mark.anyio
    async def test_list_tenants_filter_by_plan(self, manager: TenantManager) -> None:
        """Test listing tenants filtered by plan."""
        await manager.create_tenant("acme1", "Acme 1", plan="free")
        await manager.create_tenant("acme2", "Acme 2", plan="pro")
        await manager.create_tenant("acme3", "Acme 3", plan="enterprise")

        pro_tenants = await manager.list_tenants(plan="pro")
        assert len(pro_tenants) == 1
        assert pro_tenants[0].plan == "pro"

    @pytest.mark.anyio
    async def test_list_tenants_pagination(self, manager: TenantManager) -> None:
        """Test listing tenants with pagination."""
        for i in range(5):
            await manager.create_tenant(f"tenant{i}", f"Tenant {i}")

        page1 = await manager.list_tenants(limit=2, offset=0)
        assert len(page1) == 2

        page2 = await manager.list_tenants(limit=2, offset=2)
        assert len(page2) == 2

        page3 = await manager.list_tenants(limit=2, offset=4)
        assert len(page3) == 1

    @pytest.mark.anyio
    async def test_update_tenant(self, manager: TenantManager) -> None:
        """Test updating a tenant."""
        await manager.create_tenant("acme", "Acme")

        tenant = await manager.update_tenant(
            "acme",
            name="Acme Corporation",
            plan="enterprise",
            description="Updated description",
        )
        assert tenant.name == "Acme Corporation"
        assert tenant.plan == "enterprise"
        assert tenant.description == "Updated description"

    @pytest.mark.anyio
    async def test_update_tenant_not_found(self, manager: TenantManager) -> None:
        """Test updating a non-existent tenant."""
        with pytest.raises(TenantNotFoundError, match="not found"):
            await manager.update_tenant("nonexistent", name="New Name")

    @pytest.mark.anyio
    async def test_delete_tenant_soft(self, manager: TenantManager) -> None:
        """Test soft deleting a tenant."""
        await manager.create_tenant("acme", "Acme")

        await manager.delete_tenant("acme", confirm=True)

        tenant = await manager.get_tenant("acme")
        assert tenant is not None
        assert tenant.is_active is False

    @pytest.mark.anyio
    async def test_delete_tenant_hard(self, manager: TenantManager) -> None:
        """Test hard deleting a tenant."""
        await manager.create_tenant("acme", "Acme")

        await manager.delete_tenant("acme", confirm=True, hard_delete=True)

        tenant = await manager.get_tenant("acme")
        assert tenant is None

    @pytest.mark.anyio
    async def test_delete_tenant_requires_confirm(self, manager: TenantManager) -> None:
        """Test deleting requires confirmation."""
        await manager.create_tenant("acme", "Acme")

        with pytest.raises(ValueError, match="confirm=True"):
            await manager.delete_tenant("acme")

    @pytest.mark.anyio
    async def test_delete_default_tenant_fails(self, manager: TenantManager) -> None:
        """Test cannot delete default tenant."""
        with pytest.raises(TenantDeleteError, match="Cannot delete"):
            await manager.delete_tenant(DEFAULT_TENANT_ID, confirm=True)

    @pytest.mark.anyio
    async def test_delete_tenant_not_found(self, manager: TenantManager) -> None:
        """Test deleting a non-existent tenant."""
        with pytest.raises(TenantNotFoundError, match="not found"):
            await manager.delete_tenant("nonexistent", confirm=True)

    @pytest.mark.anyio
    async def test_ensure_default_tenant(self, manager: TenantManager) -> None:
        """Test ensuring default tenant exists."""
        tenant = await manager.ensure_default_tenant()
        assert tenant.id == DEFAULT_TENANT_ID
        assert tenant.name == "Default Tenant"
        assert tenant.plan == "enterprise"

        # Running again should return the same tenant
        tenant2 = await manager.ensure_default_tenant()
        assert tenant2.id == tenant.id

    @pytest.mark.anyio
    async def test_list_tenants_active_only(self, manager: TenantManager) -> None:
        """Test listing only active tenants."""
        await manager.create_tenant("acme1", "Acme 1")
        await manager.create_tenant("acme2", "Acme 2")
        await manager.delete_tenant("acme2", confirm=True)  # soft delete

        active_tenants = await manager.list_tenants(active_only=True)
        assert len(active_tenants) == 1
        assert active_tenants[0].id == "acme1"

        all_tenants = await manager.list_tenants(active_only=False)
        assert len(all_tenants) == 2
