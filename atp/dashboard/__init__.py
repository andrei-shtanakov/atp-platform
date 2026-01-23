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
"""

from atp.dashboard.app import app, create_app, run_server
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
