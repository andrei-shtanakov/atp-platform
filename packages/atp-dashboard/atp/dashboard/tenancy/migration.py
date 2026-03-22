"""Migration utilities for multi-tenancy.

This module provides utilities for migrating existing data to
the multi-tenant schema and ensuring database compatibility.
"""

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession

from atp.dashboard.tenancy.models import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    DEFAULT_TENANT_SCHEMA,
    Tenant,
    TenantQuotas,
    TenantSettings,
)

logger = logging.getLogger(__name__)


async def ensure_tenant_table(engine: AsyncEngine) -> None:
    """Ensure the tenants table exists.

    Creates the tenants table if it doesn't exist.

    Args:
        engine: SQLAlchemy async engine.
    """
    async with engine.begin() as conn:
        # Create only the Tenant table
        await conn.run_sync(
            # pyrefly: ignore[missing-attribute]
            lambda sync_conn: Tenant.__table__.create(sync_conn, checkfirst=True)
        )
    logger.info("Ensured tenants table exists")


async def create_default_tenant(engine: AsyncEngine) -> Tenant:
    """Create the default tenant if it doesn't exist.

    The default tenant is used for single-tenant deployments
    and for migrating existing data.

    Args:
        engine: SQLAlchemy async engine.

    Returns:
        The default Tenant object.
    """
    async with AsyncSession(engine) as session:
        # Check if default tenant exists
        existing = await session.get(Tenant, DEFAULT_TENANT_ID)
        if existing:
            logger.debug("Default tenant already exists")
            return existing

        # Create default tenant with generous quotas
        tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            name=DEFAULT_TENANT_NAME,
            plan="enterprise",
            description="Default tenant for single-tenant deployments",
            quotas_json=TenantQuotas(
                max_tests_per_day=10000,
                max_parallel_runs=100,
                max_storage_gb=1000.0,
                max_agents=1000,
                llm_budget_monthly=10000.00,
                max_users=1000,
                max_suites=1000,
            ).model_dump(),
            settings_json=TenantSettings().model_dump(),
            schema_name=DEFAULT_TENANT_SCHEMA,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        session.add(tenant)
        await session.commit()
        await session.refresh(tenant)

        logger.info("Created default tenant")
        return tenant


async def migrate_existing_data_to_default_tenant(
    engine: AsyncEngine,
) -> dict[str, int]:
    """Migrate existing data to the default tenant.

    Updates all records without a tenant_id to use the default tenant.
    This is safe to run multiple times.

    Args:
        engine: SQLAlchemy async engine.

    Returns:
        Dictionary mapping table names to number of records migrated.
    """
    is_postgres = "postgresql" in str(engine.url)
    migrated: dict[str, int] = {}

    # Tables that have tenant_id column
    tables = ["users", "agents", "suite_executions", "suite_definitions"]

    async with engine.begin() as conn:
        for table in tables:
            # Check if tenant_id column exists
            has_tenant_id = await _check_column_exists(
                conn, table, "tenant_id", is_postgres
            )

            if not has_tenant_id:
                logger.debug(f"Table {table} does not have tenant_id column, skipping")
                migrated[table] = 0
                continue

            # Update records without tenant_id
            update_sql = text(
                f"UPDATE {table} SET tenant_id = :tenant_id "
                "WHERE tenant_id IS NULL OR tenant_id = ''"
            )
            result = await conn.execute(update_sql, {"tenant_id": DEFAULT_TENANT_ID})
            migrated[table] = result.rowcount
            if result.rowcount > 0:
                logger.info(
                    f"Migrated {result.rowcount} records in {table} to default tenant"
                )

    total = sum(migrated.values())
    logger.info(f"Total records migrated to default tenant: {total}")
    return migrated


async def _check_column_exists(
    conn: AsyncConnection,
    table: str,
    column: str,
    is_postgres: bool,
) -> bool:
    """Check if a column exists in a table.

    Args:
        conn: Database connection.
        table: Table name.
        column: Column name.
        is_postgres: Whether this is PostgreSQL.

    Returns:
        True if column exists, False otherwise.
    """
    if is_postgres:
        check_sql = text(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table}' AND column_name = '{column}'"
        )
        result = await conn.execute(check_sql)
        return len(result.fetchall()) > 0
    else:
        # SQLite
        check_sql = text(f"PRAGMA table_info({table})")
        result = await conn.execute(check_sql)
        for row in result.fetchall():
            if row[1] == column:
                return True
        return False


async def run_tenant_migration(engine: AsyncEngine) -> None:
    """Run the complete tenant migration.

    This function:
    1. Ensures the tenants table exists
    2. Creates the default tenant
    3. Migrates existing data to the default tenant

    Args:
        engine: SQLAlchemy async engine.
    """
    logger.info("Starting tenant migration")

    # Step 1: Ensure tenants table exists
    await ensure_tenant_table(engine)

    # Step 2: Create default tenant
    await create_default_tenant(engine)

    # Step 3: Migrate existing data
    await migrate_existing_data_to_default_tenant(engine)

    logger.info("Tenant migration complete")


async def verify_tenant_isolation(
    engine: AsyncEngine, tenant_id: str
) -> dict[str, int]:
    """Verify tenant data isolation.

    Counts records in each table for the specified tenant.
    Useful for auditing and testing isolation.

    Args:
        engine: SQLAlchemy async engine.
        tenant_id: Tenant ID to check.

    Returns:
        Dictionary mapping table names to record counts.
    """
    tables = ["users", "agents", "suite_executions", "suite_definitions"]
    counts: dict[str, int] = {}

    async with engine.begin() as conn:
        for table in tables:
            count_sql = text(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id = :tenant_id"
            )
            result = await conn.execute(count_sql, {"tenant_id": tenant_id})
            counts[table] = result.scalar() or 0

    return counts
