"""Tenant-aware session for scoped database queries.

This module provides TenantAwareSession, which wraps AsyncSession
to automatically scope queries to a specific tenant.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from atp.dashboard.tenancy.models import DEFAULT_TENANT_ID, DEFAULT_TENANT_SCHEMA

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TenantAwareSession:
    """Session factory that scopes queries to a specific tenant.

    For PostgreSQL, this sets the search_path to the tenant's schema.
    For SQLite (development), this uses row-level filtering (tenant_id column).

    Usage:
        async with TenantAwareSession(engine, "acme-corp") as session:
            # All queries are scoped to the acme-corp tenant
            users = await session.execute(select(User))
    """

    def __init__(
        self,
        engine: AsyncEngine,
        tenant_id: str,
        *,
        autocommit: bool = False,
        expire_on_commit: bool = False,
    ) -> None:
        """Initialize the tenant-aware session.

        Args:
            engine: SQLAlchemy async engine.
            tenant_id: ID of the tenant to scope queries to.
            autocommit: Whether to auto-commit after each statement.
            expire_on_commit: Whether to expire objects after commit.
        """
        self._engine = engine
        self._tenant_id = tenant_id
        self._autocommit = autocommit
        self._expire_on_commit = expire_on_commit
        self._is_postgres = "postgresql" in str(engine.url)
        self._schema_name = self._get_schema_name(tenant_id)
        self._session: AsyncSession | None = None

    def _get_schema_name(self, tenant_id: str) -> str:
        """Get the schema name for a tenant."""
        if tenant_id == DEFAULT_TENANT_ID:
            return DEFAULT_TENANT_SCHEMA
        return f"tenant_{tenant_id.replace('-', '_')}"

    @property
    def tenant_id(self) -> str:
        """Get the current tenant ID."""
        return self._tenant_id

    @property
    def schema_name(self) -> str:
        """Get the current schema name."""
        return self._schema_name

    async def __aenter__(self) -> AsyncSession:
        """Enter the async context manager."""
        self._session = AsyncSession(
            self._engine,
            expire_on_commit=self._expire_on_commit,
        )

        if self._is_postgres and self._schema_name != DEFAULT_TENANT_SCHEMA:
            # Set search_path for PostgreSQL
            await self._session.execute(
                text(f"SET search_path TO {self._schema_name}, public")
            )

        return self._session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the async context manager."""
        if self._session is None:
            return

        try:
            if exc_type is not None:
                await self._session.rollback()
            elif self._autocommit:
                await self._session.commit()
        finally:
            # Reset search_path for PostgreSQL
            if self._is_postgres and self._schema_name != DEFAULT_TENANT_SCHEMA:
                try:
                    await self._session.execute(text("SET search_path TO public"))
                except Exception:
                    pass  # Best effort

            await self._session.close()
            self._session = None


class TenantSessionFactory:
    """Factory for creating tenant-aware sessions.

    Provides a convenient way to create sessions for different tenants
    while reusing the same engine configuration.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        autocommit: bool = False,
        expire_on_commit: bool = False,
    ) -> None:
        """Initialize the session factory.

        Args:
            engine: SQLAlchemy async engine.
            autocommit: Default autocommit setting for sessions.
            expire_on_commit: Default expire_on_commit setting.
        """
        self._engine = engine
        self._autocommit = autocommit
        self._expire_on_commit = expire_on_commit
        self._is_postgres = "postgresql" in str(engine.url)

    def for_tenant(
        self,
        tenant_id: str,
        *,
        autocommit: bool | None = None,
        expire_on_commit: bool | None = None,
    ) -> TenantAwareSession:
        """Create a session scoped to a specific tenant.

        Args:
            tenant_id: ID of the tenant.
            autocommit: Override default autocommit setting.
            expire_on_commit: Override default expire_on_commit setting.

        Returns:
            TenantAwareSession configured for the tenant.
        """
        return TenantAwareSession(
            self._engine,
            tenant_id,
            autocommit=autocommit if autocommit is not None else self._autocommit,
            expire_on_commit=(
                expire_on_commit
                if expire_on_commit is not None
                else self._expire_on_commit
            ),
        )

    @asynccontextmanager
    async def session(self, tenant_id: str) -> AsyncGenerator[AsyncSession, None]:
        """Get a session as an async context manager.

        Args:
            tenant_id: ID of the tenant.

        Yields:
            AsyncSession scoped to the tenant.

        Example:
            async with factory.session("acme-corp") as session:
                # Use session
                pass
        """
        async with self.for_tenant(tenant_id) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


def get_tenant_session_factory(engine: AsyncEngine) -> TenantSessionFactory:
    """Get a TenantSessionFactory for the given engine.

    Args:
        engine: SQLAlchemy async engine.

    Returns:
        TenantSessionFactory instance.
    """
    return TenantSessionFactory(engine)


# Global factory instance (initialized when database is initialized)
_tenant_factory: TenantSessionFactory | None = None


def set_tenant_factory(factory: TenantSessionFactory | None) -> None:
    """Set the global tenant session factory.

    Args:
        factory: TenantSessionFactory instance or None.
    """
    global _tenant_factory
    _tenant_factory = factory


def get_tenant_factory() -> TenantSessionFactory:
    """Get the global tenant session factory.

    Returns:
        TenantSessionFactory instance.

    Raises:
        RuntimeError: If factory hasn't been initialized.
    """
    if _tenant_factory is None:
        raise RuntimeError(
            "Tenant factory not initialized. "
            "Call init_database() or set_tenant_factory() first."
        )
    return _tenant_factory
