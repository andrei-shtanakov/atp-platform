"""ATP Dashboard v2 - Modular dashboard architecture.

This package provides a modular implementation of the ATP Dashboard
with improved organization, testability, and maintainability.

Usage:
    # Create app with default config
    from atp.dashboard.v2 import create_app
    app = create_app()

    # Create app with custom config
    from atp.dashboard.v2 import create_app, DashboardConfig
    config = DashboardConfig(debug=True)
    app = create_app(config=config)

    # Use pre-configured app instance
    from atp.dashboard.v2 import app

Environment Variables:
    ATP_DATABASE_URL: Database connection URL
    ATP_SECRET_KEY: JWT secret key for authentication
    ATP_TOKEN_EXPIRE_MINUTES: JWT token expiration time
    ATP_CORS_ORIGINS: Comma-separated list of allowed CORS origins
    ATP_DEBUG: Enable debug mode
"""

from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.dependencies import (
    AdminUser,
    Config,
    CurrentUser,
    DBSession,
    Pagination,
    PaginationParams,
    RequiredUser,
    get_dashboard_config,
    get_db,
    get_db_session,
    require_feature,
)
from atp.dashboard.v2.factory import app, create_app, create_test_app
from atp.dashboard.v2.routes import router

__all__ = [
    # App factory
    "app",
    "create_app",
    "create_test_app",
    # Configuration
    "DashboardConfig",
    "get_config",
    # Dependencies
    "get_db_session",
    "get_dashboard_config",
    "get_db",
    "require_feature",
    # Type aliases for dependencies
    "DBSession",
    "Config",
    "CurrentUser",
    "RequiredUser",
    "AdminUser",
    "Pagination",
    "PaginationParams",
    # Routes
    "router",
]
