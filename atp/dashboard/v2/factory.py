"""App factory for ATP Dashboard v2.

This module provides a factory function for creating FastAPI application
instances with proper configuration, middleware, and routes.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atp.dashboard.api import router as api_router
from atp.dashboard.database import init_database
from atp.dashboard.v2.config import DashboardConfig, get_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Initializes database on startup and cleans up on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None during application lifetime.
    """
    # Get config from app state or use default
    config: DashboardConfig = getattr(app.state, "config", get_config())

    # Initialize database on startup
    await init_database(url=config.database_url, echo=config.database_echo)

    yield

    # Cleanup on shutdown (if needed)


def create_app(
    config: DashboardConfig | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create and configure a FastAPI application instance.

    This factory function creates a new FastAPI app with:
    - Proper configuration from environment or provided config
    - CORS middleware
    - API routes mounted at /api
    - Lifespan handler for database initialization

    Args:
        config: Optional DashboardConfig instance. If not provided,
            configuration is loaded from environment variables.
        **kwargs: Additional keyword arguments passed to FastAPI constructor.

    Returns:
        Configured FastAPI application instance.

    Example:
        # Default configuration from environment
        app = create_app()

        # Custom configuration
        config = DashboardConfig(debug=True, port=8000)
        app = create_app(config=config)

        # With additional FastAPI options
        app = create_app(docs_url="/swagger", redoc_url=None)
    """
    if config is None:
        config = get_config()

    # Merge default settings with any provided kwargs
    app_settings: dict[str, Any] = {
        "title": config.title,
        "description": config.description,
        "version": config.version,
        "lifespan": lifespan,
        "debug": config.debug,
    }
    app_settings.update(kwargs)

    # Create FastAPI application
    app = FastAPI(**app_settings)

    # Store config in app state for access in lifespan and dependencies
    app.state.config = config

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router, prefix="/api")

    return app


def create_test_app(
    database_url: str | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create a FastAPI app configured for testing.

    This is a convenience function for creating test instances with
    appropriate defaults for testing (e.g., in-memory database).

    Args:
        database_url: Optional database URL. Defaults to in-memory SQLite.
        **kwargs: Additional keyword arguments passed to create_app.

    Returns:
        FastAPI application configured for testing.

    Example:
        @pytest.fixture
        def test_app():
            return create_test_app()
    """
    config = DashboardConfig(
        database_url=database_url or "sqlite+aiosqlite:///:memory:",
        database_echo=False,
        debug=True,
        secret_key="test-secret-key",
    )
    return create_app(config=config, **kwargs)


# Default application instance for backward compatibility
# This allows `from atp.dashboard.v2.factory import app`
app = create_app()
