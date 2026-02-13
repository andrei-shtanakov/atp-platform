"""ATP Web Dashboard package.

This package provides a web interface for viewing ATP test results,
including historical trends, agent comparison, and test details.

Usage:
    # Start the dashboard server
    from atp.dashboard import run_server
    run_server(host="0.0.0.0", port=8080)

    # Or from command line
    python -m atp.dashboard

    # Create app programmatically
    from atp.dashboard import create_app
    app = create_app()

    # With custom config
    from atp.dashboard.v2 import create_app, DashboardConfig
    config = DashboardConfig(debug=True)
    app = create_app(config=config)

Environment Variables:
    ATP_DATABASE_URL: Database connection URL
        (default: SQLite at ~/.atp/dashboard.db)
    ATP_SECRET_KEY: JWT secret key for authentication
    ATP_TOKEN_EXPIRE_MINUTES: JWT token expiration time
        (default: 60)
    ATP_CORS_ORIGINS: Comma-separated list of allowed
        CORS origins
    ATP_DEBUG: Enable debug mode with SQL echo
"""

from typing import TYPE_CHECKING

from atp.dashboard.database import (
    Database,
    get_database,
    init_database,
    set_database,
)
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


def create_app() -> "FastAPI":
    """Factory function to create the FastAPI application.

    Returns:
        FastAPI application instance.
    """
    from atp.dashboard.v2 import create_app as _create_app

    return _create_app()


def _get_app() -> "FastAPI":
    """Get the app instance."""
    from atp.dashboard.v2 import app as v2_app

    return v2_app


# Lazy app property - defers import until accessed
class _AppProxy:
    """Proxy for lazy loading the app."""

    _app: "FastAPI | None" = None

    def __getattr__(self, name: str):  # pyrefly: ignore
        if self._app is None:
            self._app = _get_app()
        return getattr(self._app, name)

    def __call__(self, *args, **kwargs):  # pyrefly: ignore
        if self._app is None:
            self._app = _get_app()
        return self._app(*args, **kwargs)


# Export the app proxy - resolves to v2 when accessed
app = _AppProxy()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
) -> None:  # pragma: no cover
    """Run the dashboard server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Whether to enable auto-reload.
    """
    import uvicorn

    uvicorn.run(
        "atp.dashboard.v2.factory:app",
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
