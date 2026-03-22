"""Tenant management for multi-tenancy support.

This module provides the TenantManager class for creating, deleting,
and managing tenant schemas and data.
"""

import logging
import re
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from atp.dashboard.models import Base
from atp.dashboard.tenancy.models import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    DEFAULT_TENANT_SCHEMA,
    Tenant,
    TenantQuotas,
    TenantSettings,
)

logger = logging.getLogger(__name__)


class TenantError(Exception):
    """Base exception for tenant operations."""

    pass


class TenantExistsError(TenantError):
    """Raised when trying to create a tenant that already exists."""

    pass


class TenantNotFoundError(TenantError):
    """Raised when a tenant is not found."""

    pass


class TenantDeleteError(TenantError):
    """Raised when tenant deletion fails."""

    pass


class TenantManager:
    """Manages tenant lifecycle and schema operations.

    Provides methods for creating, deleting, and managing tenant schemas.
    Uses schema-per-tenant isolation for PostgreSQL and row-level isolation
    for SQLite (development mode).
    """

    # Valid tenant ID pattern: lowercase letters, numbers, hyphens
    TENANT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$")

    # Reserved tenant IDs that cannot be used
    RESERVED_IDS = frozenset({"default", "public", "admin", "system", "root", "master"})

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize the tenant manager.

        Args:
            engine: SQLAlchemy async engine for database operations.
        """
        self._engine = engine
        self._is_postgres = "postgresql" in str(engine.url)

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """Validate tenant ID format.

        Args:
            tenant_id: The tenant ID to validate.

        Raises:
            ValueError: If the tenant ID is invalid.
        """
        if not tenant_id:
            raise ValueError("Tenant ID cannot be empty")

        if len(tenant_id) > 50:
            raise ValueError("Tenant ID must be 50 characters or less")

        if tenant_id in self.RESERVED_IDS:
            raise ValueError(f"Tenant ID '{tenant_id}' is reserved")

        if not self.TENANT_ID_PATTERN.match(tenant_id):
            raise ValueError(
                f"Tenant ID '{tenant_id}' is invalid. "
                "Must contain only lowercase letters, numbers, and hyphens, "
                "and cannot start or end with a hyphen."
            )

    def _get_schema_name(self, tenant_id: str) -> str:
        """Get the schema name for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Schema name for the tenant.
        """
        if tenant_id == DEFAULT_TENANT_ID:
            return DEFAULT_TENANT_SCHEMA
        return f"tenant_{tenant_id.replace('-', '_')}"

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        *,
        plan: str = "free",
        description: str | None = None,
        quotas: TenantQuotas | None = None,
        settings: TenantSettings | None = None,
        contact_email: str | None = None,
    ) -> Tenant:
        """Create a new tenant with its schema.

        Creates a new tenant record and sets up the isolated schema
        with all required tables.

        Args:
            tenant_id: Unique identifier for the tenant.
            name: Display name for the tenant.
            plan: Subscription plan (free, pro, enterprise).
            description: Optional description.
            quotas: Resource quotas, uses defaults if not provided.
            settings: Tenant settings, uses defaults if not provided.
            contact_email: Contact email for the tenant.

        Returns:
            The created Tenant object.

        Raises:
            ValueError: If the tenant ID is invalid.
            TenantExistsError: If a tenant with this ID already exists.
            TenantError: If schema creation fails.
        """
        self._validate_tenant_id(tenant_id)

        schema_name = self._get_schema_name(tenant_id)
        quotas = quotas or TenantQuotas()
        settings = settings or TenantSettings()

        # Create tenant record first
        async with AsyncSession(self._engine) as session:
            # Check if tenant already exists
            existing = await session.get(Tenant, tenant_id)
            if existing:
                raise TenantExistsError(f"Tenant '{tenant_id}' already exists")

            tenant = Tenant(
                id=tenant_id,
                name=name,
                plan=plan,
                description=description,
                quotas_json=quotas.model_dump(),
                settings_json=settings.model_dump(),
                schema_name=schema_name,
                contact_email=contact_email,
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(tenant)
            await session.commit()
            await session.refresh(tenant)

            logger.info(f"Created tenant record: {tenant_id}")

        # Create schema (PostgreSQL only)
        if self._is_postgres:
            await self._create_schema(schema_name)

        return tenant

    async def _create_schema(self, schema_name: str) -> None:
        """Create a database schema and tables for a tenant.

        Args:
            schema_name: Name of the schema to create.

        Raises:
            TenantError: If schema creation fails.
        """
        if schema_name == DEFAULT_TENANT_SCHEMA:
            # Don't create public schema, just ensure tables exist
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            return

        try:
            async with self._engine.begin() as conn:
                # Create schema
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

                # Create tables in the new schema
                # We need to temporarily set the search_path
                await conn.execute(text(f"SET search_path TO {schema_name}, public"))

                # Create all tables from the Base metadata
                await conn.run_sync(Base.metadata.create_all)

                # Reset search_path
                await conn.execute(text("SET search_path TO public"))

            logger.info(f"Created schema: {schema_name}")
        except Exception as e:
            logger.error(f"Failed to create schema {schema_name}: {e}")
            raise TenantError(f"Failed to create schema: {e}") from e

    async def delete_tenant(
        self,
        tenant_id: str,
        *,
        confirm: bool = False,
        hard_delete: bool = False,
    ) -> None:
        """Delete a tenant and optionally its schema.

        Args:
            tenant_id: ID of the tenant to delete.
            confirm: Must be True to actually delete.
            hard_delete: If True, drops the schema entirely (PostgreSQL).
                        If False, marks tenant as inactive.

        Raises:
            ValueError: If confirm is not True.
            TenantNotFoundError: If the tenant doesn't exist.
            TenantDeleteError: If deletion fails.
        """
        if tenant_id == DEFAULT_TENANT_ID:
            raise TenantDeleteError("Cannot delete the default tenant")

        if not confirm:
            raise ValueError(
                "Tenant deletion requires confirm=True. "
                "This operation cannot be undone."
            )

        async with AsyncSession(self._engine) as session:
            tenant = await session.get(Tenant, tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")

            schema_name = tenant.schema_name

            if hard_delete:
                # Drop schema (PostgreSQL only)
                if self._is_postgres and schema_name != DEFAULT_TENANT_SCHEMA:
                    await self._drop_schema(schema_name)

                # Delete tenant record
                await session.delete(tenant)
                logger.info(f"Hard deleted tenant: {tenant_id}")
            else:
                # Soft delete: mark as inactive
                tenant.is_active = False
                tenant.updated_at = datetime.now()
                logger.info(f"Soft deleted tenant: {tenant_id}")

            await session.commit()

    async def _drop_schema(self, schema_name: str) -> None:
        """Drop a tenant schema and all its data.

        Args:
            schema_name: Name of the schema to drop.

        Raises:
            TenantDeleteError: If schema drop fails.
        """
        if schema_name == DEFAULT_TENANT_SCHEMA:
            raise TenantDeleteError("Cannot drop the public schema")

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
            logger.info(f"Dropped schema: {schema_name}")
        except Exception as e:
            logger.error(f"Failed to drop schema {schema_name}: {e}")
            raise TenantDeleteError(f"Failed to drop schema: {e}") from e

    async def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Tenant object or None if not found.
        """
        async with AsyncSession(self._engine) as session:
            return await session.get(Tenant, tenant_id)

    async def list_tenants(
        self,
        *,
        active_only: bool = True,
        plan: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tenant]:
        """List tenants with optional filtering.

        Args:
            active_only: Only return active tenants.
            plan: Filter by plan type.
            limit: Maximum number of tenants to return.
            offset: Number of tenants to skip.

        Returns:
            List of Tenant objects.
        """
        from sqlalchemy import select

        async with AsyncSession(self._engine) as session:
            query = select(Tenant)

            if active_only:
                query = query.where(Tenant.is_active.is_(True))

            if plan:
                query = query.where(Tenant.plan == plan)

            query = query.order_by(Tenant.created_at.desc())
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_tenant(
        self,
        tenant_id: str,
        *,
        name: str | None = None,
        plan: str | None = None,
        description: str | None = None,
        quotas: TenantQuotas | None = None,
        settings: TenantSettings | None = None,
        contact_email: str | None = None,
        is_active: bool | None = None,
    ) -> Tenant:
        """Update a tenant's properties.

        Args:
            tenant_id: ID of the tenant to update.
            name: New display name.
            plan: New subscription plan.
            description: New description.
            quotas: New quotas.
            settings: New settings.
            contact_email: New contact email.
            is_active: New active status.

        Returns:
            Updated Tenant object.

        Raises:
            TenantNotFoundError: If tenant doesn't exist.
        """
        async with AsyncSession(self._engine) as session:
            tenant = await session.get(Tenant, tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")

            if name is not None:
                tenant.name = name
            if plan is not None:
                tenant.plan = plan
            if description is not None:
                tenant.description = description
            if quotas is not None:
                tenant.quotas_json = quotas.model_dump()
            if settings is not None:
                tenant.settings_json = settings.model_dump()
            if contact_email is not None:
                tenant.contact_email = contact_email
            if is_active is not None:
                tenant.is_active = is_active

            tenant.updated_at = datetime.now()
            await session.commit()
            await session.refresh(tenant)
            return tenant

    async def ensure_default_tenant(self) -> Tenant:
        """Ensure the default tenant exists.

        Creates the default tenant if it doesn't exist.
        This is used for single-tenant deployments and
        for migrating existing data.

        Returns:
            The default Tenant object.
        """
        async with AsyncSession(self._engine) as session:
            tenant = await session.get(Tenant, DEFAULT_TENANT_ID)
            if tenant:
                return tenant

        # Create default tenant without validation (it's a reserved ID)
        tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            name=DEFAULT_TENANT_NAME,
            plan="enterprise",  # Default tenant gets full access
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

        async with AsyncSession(self._engine) as session:
            session.add(tenant)
            await session.commit()
            await session.refresh(tenant)

        logger.info("Created default tenant")
        return tenant

    async def migrate_existing_data(self) -> int:
        """Migrate existing data to the default tenant.

        Adds tenant_id to all existing records that don't have one.
        This is used when upgrading from a single-tenant deployment.

        Returns:
            Number of records migrated.
        """
        # First ensure default tenant exists
        await self.ensure_default_tenant()

        count = 0
        tables_with_tenant_id = [
            "users",
            "agents",
            "suite_executions",
            "suite_definitions",
        ]

        async with self._engine.begin() as conn:
            for table in tables_with_tenant_id:
                # Check if tenant_id column exists
                if self._is_postgres:
                    check_sql = text(
                        "SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{table}' AND column_name = 'tenant_id'"
                    )
                else:
                    # SQLite
                    check_sql = text(f"PRAGMA table_info({table})")

                result = await conn.execute(check_sql)
                columns = result.fetchall()

                has_tenant_id = False
                if self._is_postgres:
                    has_tenant_id = len(columns) > 0
                else:
                    for col in columns:
                        if col[1] == "tenant_id":
                            has_tenant_id = True
                            break

                if has_tenant_id:
                    # Update records without tenant_id
                    update_sql = text(
                        f"UPDATE {table} SET tenant_id = :tenant_id "
                        "WHERE tenant_id IS NULL"
                    )
                    result = await conn.execute(
                        update_sql, {"tenant_id": DEFAULT_TENANT_ID}
                    )
                    count += result.rowcount

        logger.info(f"Migrated {count} records to default tenant")
        return count
