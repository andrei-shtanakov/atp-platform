"""FastAPI dependency injection for ATP Dashboard v2.

This module provides dependency injection functions for FastAPI routes,
including database sessions, configuration, authentication, and services.
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import (
    get_current_active_user,
    get_current_admin_user,
    get_current_user,
)
from atp.dashboard.database import Database, get_database
from atp.dashboard.models import User
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.services.agent_service import AgentService
from atp.dashboard.v2.services.comparison_service import ComparisonService
from atp.dashboard.v2.services.export_service import ExportService
from atp.dashboard.v2.services.test_service import TestService


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for dependency injection.

    Yields:
        AsyncSession for database operations.

    Example:
        @app.get("/items")
        async def get_items(session: Annotated[AsyncSession, Depends(get_db_session)]):
            ...
    """
    db = get_database()
    async with db.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_dashboard_config() -> DashboardConfig:
    """Get dashboard configuration for dependency injection.

    Returns:
        DashboardConfig instance.

    Example:
        @app.get("/config")
        async def get_config_info(config: Config):
            return {"debug": config.debug}
    """
    return get_config()


def get_db() -> Database:
    """Get the database instance for dependency injection.

    Returns:
        Database instance.
    """
    return get_database()


# Type aliases for cleaner route signatures
DBSession = Annotated[AsyncSession, Depends(get_db_session)]
Config = Annotated[DashboardConfig, Depends(get_dashboard_config)]
CurrentUser = Annotated[User | None, Depends(get_current_user)]
RequiredUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Depends(get_current_admin_user)]


class PaginationParams:
    """Pagination parameters for list endpoints."""

    def __init__(
        self,
        offset: int = 0,
        limit: int = 50,
    ) -> None:
        """Initialize pagination parameters.

        Args:
            offset: Number of items to skip (default 0).
            limit: Maximum number of items to return (default 50, max 100).
        """
        self.offset = max(0, offset)
        self.limit = min(max(1, limit), 100)


Pagination = Annotated[PaginationParams, Depends()]


def require_feature(feature_name: str):
    """Dependency that checks if a feature is enabled.

    Args:
        feature_name: Name of the feature to check.

    Returns:
        Dependency function that raises 404 if feature is disabled.

    Example:
        @app.get("/beta-feature", dependencies=[Depends(require_feature("beta"))])
        async def beta_feature():
            ...
    """

    def _check_feature(config: Config) -> None:
        # For now, check if debug mode enables all features
        # In the future, this could check a feature flags config
        feature_enabled = config.debug or getattr(
            config, f"feature_{feature_name}", False
        )
        if not feature_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' is not enabled",
            )

    return _check_feature


# Service dependency injection factories
async def get_test_service(session: DBSession) -> TestService:
    """Get a TestService instance for dependency injection.

    Args:
        session: Database session.

    Returns:
        TestService instance.

    Example:
        @app.get("/tests")
        async def list_tests(service: TestServiceDep):
            return await service.list_test_executions()
    """
    return TestService(session)


async def get_agent_service(session: DBSession) -> AgentService:
    """Get an AgentService instance for dependency injection.

    Args:
        session: Database session.

    Returns:
        AgentService instance.

    Example:
        @app.get("/agents")
        async def list_agents(service: AgentServiceDep):
            return await service.list_agents()
    """
    return AgentService(session)


async def get_comparison_service(session: DBSession) -> ComparisonService:
    """Get a ComparisonService instance for dependency injection.

    Args:
        session: Database session.

    Returns:
        ComparisonService instance.

    Example:
        @app.get("/compare/agents")
        async def compare_agents(service: ComparisonServiceDep, ...):
            return await service.compare_agents(...)
    """
    return ComparisonService(session)


async def get_export_service(session: DBSession) -> ExportService:
    """Get an ExportService instance for dependency injection.

    Args:
        session: Database session.

    Returns:
        ExportService instance.

    Example:
        @app.get("/export/csv")
        async def export_csv(service: ExportServiceDep, ...):
            return await service.export_results_to_csv(...)
    """
    return ExportService(session)


# Service type aliases for cleaner route signatures
TestServiceDep = Annotated[TestService, Depends(get_test_service)]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
ComparisonServiceDep = Annotated[ComparisonService, Depends(get_comparison_service)]
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]
