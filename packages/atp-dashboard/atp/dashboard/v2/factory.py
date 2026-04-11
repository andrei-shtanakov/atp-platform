"""App factory for ATP Dashboard v2.

This module provides a factory function for creating FastAPI application
instances with proper configuration, middleware, and routes.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from atp.dashboard.database import init_database
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.rate_limit import (
    JWTUserStateMiddleware,
    create_limiter,
    rate_limit_exceeded_handler,
)
from atp.dashboard.v2.routes import router as api_router

# Template and static file directories
V2_DIR = Path(__file__).parent
TEMPLATES_DIR = V2_DIR / "templates"
STATIC_DIR = V2_DIR / "static"


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
    config: DashboardConfig = getattr(app.state, "config", None) or get_config()

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
        **kwargs: Additional keyword arguments passed to FastAPI
            constructor.

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

    # Set up the FastMCP sub-app first so its lifespan can be composed
    # into the outer FastAPI lifespan. Phase 0.2 verified that Starlette
    # does NOT propagate sub-app lifespans under mount(), so we must
    # drive FastMCP's session-manager lifespan ourselves.
    from atp.dashboard.mcp import mcp_server
    from atp.dashboard.mcp import tools as _mcp_tools  # noqa: F401
    from atp.dashboard.mcp.auth import MCPAuthMiddleware

    mcp_app = mcp_server.http_app(transport="sse")

    @asynccontextmanager
    async def _combined_lifespan(app_: FastAPI) -> AsyncGenerator[None, None]:
        async with lifespan(app_):
            async with mcp_app.router.lifespan_context(app_):
                yield

    # Merge default settings with any provided kwargs
    app_settings: dict[str, Any] = {
        "title": config.title,
        "description": config.description,
        "version": config.version,
        "lifespan": _combined_lifespan,
        "debug": config.debug,
    }
    app_settings.update(kwargs)

    # Create FastAPI application
    app = FastAPI(**app_settings)

    # Store config in app state for access in lifespan and deps
    app.state.config = config

    # Set up rate limiting.
    # Middleware order matters: Starlette applies middleware LIFO, so the
    # middleware added last runs first on the request path. We want:
    #   CORS → JWTUserState → SlowAPI → routes
    # so that the rate-limit key function sees the authenticated user_id.
    limiter = create_limiter(config)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(JWTUserStateMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the MCP tournament server under /mcp.
    # MCPAuthMiddleware sits between JWTUserStateMiddleware (which
    # populates request.state.user_id) and FastMCP, rejecting
    # unauthenticated handshakes with 401.
    mcp_app.add_middleware(MCPAuthMiddleware)
    app.mount("/mcp", mcp_app)

    # Mount API routes
    app.include_router(api_router, prefix="/api")

    # Mount UI routes (HTMX + Pico CSS frontend)
    from atp.dashboard.v2.routes.ui import router as ui_router

    app.include_router(ui_router)

    # Configure Jinja2 templates
    if TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        app.state.templates = templates

        # Mount static files for v2
        if STATIC_DIR.exists():
            app.mount(
                "/static/v2",
                StaticFiles(directory=str(STATIC_DIR)),
                name="static_v2",
            )

        # Redirect root to the new UI
        @app.get("/")
        async def home() -> RedirectResponse:
            """Redirect root to /ui/ dashboard."""
            return RedirectResponse(url="/ui/", status_code=302)

        # Legacy v1 routes — redirect to new /ui/ equivalents
        @app.get("/login")
        async def legacy_login() -> RedirectResponse:
            return RedirectResponse(url="/ui/login", status_code=302)

        @app.get("/register")
        async def legacy_register() -> RedirectResponse:
            return RedirectResponse(url="/ui/register", status_code=302)

        @app.get("/games")
        async def legacy_games() -> RedirectResponse:
            return RedirectResponse(url="/ui/games", status_code=302)

        @app.get("/analytics")
        async def legacy_analytics() -> RedirectResponse:
            return RedirectResponse(url="/ui/analytics", status_code=302)

    return app


def create_test_app(
    database_url: str | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create a FastAPI app configured for testing.

    This is a convenience function for creating test instances with
    appropriate defaults for testing (e.g., in-memory database).

    Args:
        database_url: Optional database URL. Defaults to in-memory
            SQLite.
        **kwargs: Additional keyword arguments passed to create_app.

    Returns:
        FastAPI application configured for testing.

    Example:
        @pytest.fixture
        def test_app():
            return create_test_app()
    """
    import os

    # Set env vars so get_config() (called by require_permission etc.) works
    _env_overrides = {
        "ATP_SECRET_KEY": "test-secret-key",
        "ATP_DEBUG": "true",
        "ATP_RATE_LIMIT_ENABLED": "false",
    }
    _saved = {k: os.environ.get(k) for k in _env_overrides}
    for k, v in _env_overrides.items():
        if k not in os.environ:
            os.environ[k] = v
    get_config.cache_clear()

    config = DashboardConfig(
        database_url=(database_url or "sqlite+aiosqlite:///:memory:"),
        database_echo=False,
        debug=True,
        secret_key="test-secret-key",
        disable_auth=True,
        rate_limit_enabled=False,
    )
    return create_app(config=config, **kwargs)


# Default application instance
# This allows `from atp.dashboard.v2.factory import app`
# Uses test-safe defaults when no environment is configured.
try:
    app = create_app()
except ValueError:
    # Fallback for environments without ATP_SECRET_KEY configured.
    # Production deployments must set ATP_SECRET_KEY.
    import logging as _logging

    _logging.getLogger("atp.dashboard").warning(
        "Failed to create default app instance: ATP_SECRET_KEY not set. "
        "Using debug fallback. Set ATP_SECRET_KEY for production."
    )
    app = create_app(config=DashboardConfig(debug=True))
