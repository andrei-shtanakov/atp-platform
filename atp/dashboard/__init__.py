"""ATP Web Dashboard package.

This package provides a web interface for viewing ATP test results,
including historical trends, agent comparison, and test details.

Usage:
    # Start the dashboard server
    from atp.dashboard import run_server
    run_server(host="0.0.0.0", port=8080)

    # Or from command line
    python -m atp.dashboard

Environment Variables:
    ATP_DATABASE_URL: Database connection URL (default: SQLite at ~/.atp/dashboard.db)
    ATP_SECRET_KEY: JWT secret key for authentication
    ATP_TOKEN_EXPIRE_MINUTES: JWT token expiration time (default: 60)
    ATP_CORS_ORIGINS: Comma-separated list of allowed CORS origins
    ATP_DEBUG: Enable debug mode with SQL echo
    ATP_DASHBOARD_V2: Set to 'true' to enable v2 modular dashboard (RECOMMENDED)

Version Switching:
    The dashboard supports two versions:

    - v1 (default): Original monolithic implementation in app.py
      **DEPRECATED** - Will be removed in ATP 1.0.0

    - v2 (recommended): New modular implementation with app factory pattern
      Enable with: ATP_DASHBOARD_V2=true

    **Recommended**: Set ATP_DASHBOARD_V2=true to use the v2 implementation.
    See docs/reference/dashboard-migration.md for migration guide.

Example (v2 recommended):
    >>> import os
    >>> os.environ["ATP_DASHBOARD_V2"] = "true"
    >>> from atp.dashboard import create_app
    >>> app = create_app()

    Or directly use v2:
    >>> from atp.dashboard.v2 import create_app, DashboardConfig
    >>> config = DashboardConfig(debug=True)
    >>> app = create_app(config=config)
"""

import os
from typing import TYPE_CHECKING

from atp.dashboard.database import Database, get_database, init_database, set_database
from atp.dashboard.models import (
    Agent,
    Artifact,
    Base,
    EvaluationResult,
    RunResult,
    ScoreComponent,
    SuiteExecution,
    TestExecution,
    User,
)
from atp.dashboard.storage import ResultStorage

if TYPE_CHECKING:
    from fastapi import FastAPI


def _is_v2_enabled() -> bool:
    """Check if Dashboard v2 is enabled via feature flag."""
    return os.getenv("ATP_DASHBOARD_V2", "false").lower() == "true"


def _get_v1_app() -> "FastAPI":
    """Get the v1 app instance."""
    from atp.dashboard.app import app as v1_app

    return v1_app


def _get_v2_app() -> "FastAPI":
    """Get the v2 app instance."""
    from atp.dashboard.v2 import app as v2_app

    return v2_app


def _get_app() -> "FastAPI":
    """Get the appropriate app instance based on feature flag."""
    if _is_v2_enabled():
        return _get_v2_app()
    return _get_v1_app()


# Lazy app property - defers import until accessed
class _AppProxy:
    """Proxy for lazy loading the app based on feature flag."""

    _app: "FastAPI | None" = None

    def __getattr__(self, name: str):
        if self._app is None:
            self._app = _get_app()
        return getattr(self._app, name)

    def __call__(self, *args, **kwargs):
        if self._app is None:
            self._app = _get_app()
        return self._app(*args, **kwargs)


# Export the app proxy - it will resolve to the correct version when accessed
app = _AppProxy()


def create_app() -> "FastAPI":
    """Factory function to create the FastAPI application.

    Returns the v1 or v2 app based on ATP_DASHBOARD_V2 environment variable.

    Returns:
        FastAPI application instance.
    """
    if _is_v2_enabled():
        from atp.dashboard.v2 import create_app as create_v2_app

        return create_v2_app()
    else:
        from atp.dashboard.app import create_app as create_v1_app

        return create_v1_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
) -> None:  # pragma: no cover
    """Run the dashboard server.

    Uses v1 or v2 based on ATP_DASHBOARD_V2 environment variable.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Whether to enable auto-reload.
    """
    import uvicorn

    if _is_v2_enabled():
        uvicorn.run(
            "atp.dashboard.v2.factory:app",
            host=host,
            port=port,
            reload=reload,
        )
    else:
        uvicorn.run(
            "atp.dashboard.app:app",
            host=host,
            port=port,
            reload=reload,
        )


__all__ = [
    # Application
    "app",
    "create_app",
    "run_server",
    # Database
    "Base",
    "Database",
    "get_database",
    "init_database",
    "set_database",
    # Models
    "Agent",
    "Artifact",
    "EvaluationResult",
    "RunResult",
    "ScoreComponent",
    "SuiteExecution",
    "TestExecution",
    "User",
    # Storage
    "ResultStorage",
]
