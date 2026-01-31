"""FastAPI dependency injection for ATP Dashboard v2.

This module provides dependency injection functions for FastAPI routes,
including database sessions, configuration, and authentication.
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
