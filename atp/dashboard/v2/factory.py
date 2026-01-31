"""App factory for ATP Dashboard v2.

This module provides a factory function for creating FastAPI application
instances with proper configuration, middleware, and routes.

The factory supports two modes:
- v1: Uses the original monolithic api.py router (default)
- v2: Uses the new modular routes package

Set ATP_DASHBOARD_V2=true to enable v2 routes.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atp.dashboard.api import router as api_router_v1
from atp.dashboard.database import init_database
from atp.dashboard.v2.config import DashboardConfig, get_config, is_v2_enabled
from atp.dashboard.v2.routes import router as api_router_v2


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
    use_v2_routes: bool | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create and configure a FastAPI application instance.

    This factory function creates a new FastAPI app with:
    - Proper configuration from environment or provided config
    - CORS middleware
    - API routes mounted at /api (v1 or v2 based on configuration)
    - Lifespan handler for database initialization

    Args:
        config: Optional DashboardConfig instance. If not provided,
            configuration is loaded from environment variables.
        use_v2_routes: Whether to use v2 modular routes. If None, checks
            the ATP_DASHBOARD_V2 environment variable.
        **kwargs: Additional keyword arguments passed to FastAPI constructor.

    Returns:
        Configured FastAPI application instance.

    Example:
        # Default configuration from environment
        app = create_app()

        # Custom configuration
        config = DashboardConfig(debug=True, port=8000)
        app = create_app(config=config)

        # Force v2 routes
        app = create_app(use_v2_routes=True)

        # With additional FastAPI options
        app = create_app(docs_url="/swagger", redoc_url=None)
    """
    if config is None:
        config = get_config()

    # Determine which router to use
    if use_v2_routes is None:
        use_v2_routes = is_v2_enabled()

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
    app.state.use_v2_routes = use_v2_routes

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes (v1 or v2 based on configuration)
    api_router = api_router_v2 if use_v2_routes else api_router_v1
    app.include_router(api_router, prefix="/api")

    return app


def create_test_app(
    database_url: str | None = None,
    use_v2_routes: bool | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create a FastAPI app configured for testing.

    This is a convenience function for creating test instances with
    appropriate defaults for testing (e.g., in-memory database).

    Args:
        database_url: Optional database URL. Defaults to in-memory SQLite.
        use_v2_routes: Whether to use v2 modular routes. If None, uses v1.
        **kwargs: Additional keyword arguments passed to create_app.

    Returns:
        FastAPI application configured for testing.

    Example:
        @pytest.fixture
        def test_app():
            return create_test_app()

        @pytest.fixture
        def test_app_v2():
            return create_test_app(use_v2_routes=True)
    """
    config = DashboardConfig(
        database_url=database_url or "sqlite+aiosqlite:///:memory:",
        database_echo=False,
        debug=True,
        secret_key="test-secret-key",
    )
    return create_app(config=config, use_v2_routes=use_v2_routes, **kwargs)


# Default application instance for backward compatibility
# This allows `from atp.dashboard.v2.factory import app`
app = create_app()
