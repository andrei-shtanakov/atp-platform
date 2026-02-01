"""Tests for tenant migration utilities."""

from pathlib import Path

import pytest

from atp.dashboard.tenancy.migration import (
    create_default_tenant,
    ensure_tenant_table,
    migrate_existing_data_to_default_tenant,
    run_tenant_migration,
    verify_tenant_isolation,
)
from atp.dashboard.tenancy.models import DEFAULT_TENANT_ID, Tenant


class TestMigration:
    """Tests for migration utilities."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    @pytest.fixture
    async def engine(self, temp_db_path: str):
        """Create a test engine."""
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(temp_db_path)
        yield engine
        await engine.dispose()

    @pytest.fixture
    async def engine_with_tables(self, temp_db_path: str):
        """Create a test engine with tables."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atp.dashboard.models import Base

        engine = create_async_engine(temp_db_path)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine
        await engine.dispose()

    @pytest.mark.anyio
    async def test_ensure_tenant_table(self, engine) -> None:
        """Test ensuring tenant table exists."""
        await ensure_tenant_table(engine)

        # Verify table exists by trying to query it
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(engine) as session:
            result = await session.execute(text("SELECT COUNT(*) FROM tenants"))
            count = result.scalar()
            assert count == 0

    @pytest.mark.anyio
    async def test_create_default_tenant(self, engine_with_tables) -> None:
        """Test creating default tenant."""
        # First ensure tenant table exists
        await ensure_tenant_table(engine_with_tables)

        tenant = await create_default_tenant(engine_with_tables)

        assert tenant.id == DEFAULT_TENANT_ID
        assert tenant.name == "Default Tenant"
        assert tenant.plan == "enterprise"
        assert tenant.is_active is True
        assert tenant.schema_name == "public"

    @pytest.mark.anyio
    async def test_create_default_tenant_idempotent(self, engine_with_tables) -> None:
        """Test creating default tenant is idempotent."""
        await ensure_tenant_table(engine_with_tables)

        tenant1 = await create_default_tenant(engine_with_tables)
        tenant2 = await create_default_tenant(engine_with_tables)

        assert tenant1.id == tenant2.id

    @pytest.mark.anyio
    async def test_migrate_existing_data(self, engine_with_tables) -> None:
        """Test migrating existing data to default tenant.

        Note: Since tenant_id is now NOT NULL with a default value,
        we test migration of records that have empty string tenant_id.
        """
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession

        await ensure_tenant_table(engine_with_tables)
        await create_default_tenant(engine_with_tables)

        # Insert test data with empty tenant_id (simulating pre-migration data)
        async with AsyncSession(engine_with_tables) as session:
            # Insert an agent with empty tenant_id
            await session.execute(
                text(
                    "INSERT INTO agents (name, agent_type, config, tenant_id, "
                    "created_at, updated_at) "
                    "VALUES ('test-agent', 'http', '{}', '', "
                    "datetime('now'), datetime('now'))"
                )
            )
            await session.commit()

        # Run migration
        result = await migrate_existing_data_to_default_tenant(engine_with_tables)

        assert result["agents"] == 1

        # Verify data was migrated
        async with AsyncSession(engine_with_tables) as session:
            row = await session.execute(
                text("SELECT tenant_id FROM agents WHERE name = 'test-agent'")
            )
            tenant_id = row.scalar()
            assert tenant_id == DEFAULT_TENANT_ID

    @pytest.mark.anyio
    async def test_run_tenant_migration(self, engine_with_tables) -> None:
        """Test running complete tenant migration."""
        await run_tenant_migration(engine_with_tables)

        # Verify default tenant exists
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(engine_with_tables) as session:
            tenant = await session.get(Tenant, DEFAULT_TENANT_ID)
            assert tenant is not None
            assert tenant.is_active is True

    @pytest.mark.anyio
    async def test_verify_tenant_isolation(self, engine_with_tables) -> None:
        """Test verifying tenant isolation."""
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession

        await ensure_tenant_table(engine_with_tables)
        await create_default_tenant(engine_with_tables)

        # Insert test data
        async with AsyncSession(engine_with_tables) as session:
            await session.execute(
                text(
                    "INSERT INTO agents (name, agent_type, config, tenant_id, "
                    "created_at, updated_at) "
                    f"VALUES ('agent1', 'http', '{{}}', '{DEFAULT_TENANT_ID}', "
                    "datetime('now'), datetime('now'))"
                )
            )
            await session.execute(
                text(
                    "INSERT INTO agents (name, agent_type, config, tenant_id, "
                    "created_at, updated_at) "
                    "VALUES ('agent2', 'http', '{}', 'other-tenant', "
                    "datetime('now'), datetime('now'))"
                )
            )
            await session.commit()

        # Verify isolation
        default_counts = await verify_tenant_isolation(
            engine_with_tables, DEFAULT_TENANT_ID
        )
        assert default_counts["agents"] == 1

        other_counts = await verify_tenant_isolation(engine_with_tables, "other-tenant")
        assert other_counts["agents"] == 1


class TestMigrationEdgeCases:
    """Tests for migration edge cases."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    @pytest.fixture
    async def engine_with_tables(self, temp_db_path: str):
        """Create a test engine with tables."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atp.dashboard.models import Base

        engine = create_async_engine(temp_db_path)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine
        await engine.dispose()

    @pytest.mark.anyio
    async def test_migrate_empty_string_tenant_id(self, engine_with_tables) -> None:
        """Test migrating records with empty string tenant_id."""
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession

        await ensure_tenant_table(engine_with_tables)
        await create_default_tenant(engine_with_tables)

        # Insert data with empty string tenant_id
        async with AsyncSession(engine_with_tables) as session:
            await session.execute(
                text(
                    "INSERT INTO agents (name, agent_type, config, tenant_id, "
                    "created_at, updated_at) "
                    "VALUES ('empty-tenant-agent', 'http', '{}', '', "
                    "datetime('now'), datetime('now'))"
                )
            )
            await session.commit()

        # Run migration
        result = await migrate_existing_data_to_default_tenant(engine_with_tables)
        assert result["agents"] == 1

        # Verify
        async with AsyncSession(engine_with_tables) as session:
            row = await session.execute(
                text("SELECT tenant_id FROM agents WHERE name = 'empty-tenant-agent'")
            )
            tenant_id = row.scalar()
            assert tenant_id == DEFAULT_TENANT_ID

    @pytest.mark.anyio
    async def test_migrate_already_migrated_data(self, engine_with_tables) -> None:
        """Test migration is idempotent."""
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession

        await ensure_tenant_table(engine_with_tables)
        await create_default_tenant(engine_with_tables)

        # Insert data with valid tenant_id
        async with AsyncSession(engine_with_tables) as session:
            await session.execute(
                text(
                    "INSERT INTO agents (name, agent_type, config, tenant_id, "
                    "created_at, updated_at) "
                    f"VALUES ('migrated-agent', 'http', '{{}}', '{DEFAULT_TENANT_ID}', "
                    "datetime('now'), datetime('now'))"
                )
            )
            await session.commit()

        # Run migration - should not change anything
        result = await migrate_existing_data_to_default_tenant(engine_with_tables)
        assert result["agents"] == 0
