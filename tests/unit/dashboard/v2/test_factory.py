"""Tests for ATP Dashboard v2 app factory module."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.factory import app, create_app, create_test_app, lifespan


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI instance."""
        test_app = create_app()
        assert isinstance(test_app, FastAPI)

    def test_app_has_correct_title(self) -> None:
        """Test that app has correct title from config."""
        test_app = create_app()
        assert test_app.title == "ATP Dashboard"

    def test_app_has_correct_version(self) -> None:
        """Test that app has correct version from config."""
        test_app = create_app()
        assert test_app.version == "0.2.0"

    def test_app_with_custom_config(self) -> None:
        """Test create_app with custom configuration."""
        config = DashboardConfig(
            title="Custom Dashboard",
            version="1.0.0",
            debug=True,
        )
        test_app = create_app(config=config)
        assert test_app.title == "Custom Dashboard"
        assert test_app.version == "1.0.0"
        assert test_app.debug is True

    def test_app_with_kwargs(self) -> None:
        """Test create_app with additional kwargs."""
        test_app = create_app(docs_url="/swagger")
        assert test_app.docs_url == "/swagger"

    def test_app_stores_config_in_state(self) -> None:
        """Test that config is stored in app state."""
        config = DashboardConfig(debug=True)
        test_app = create_app(config=config)
        assert test_app.state.config is config

    def test_app_has_cors_middleware(self) -> None:
        """Test that app has CORS middleware configured."""
        test_app = create_app()
        middleware_classes = [m.cls.__name__ for m in test_app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_app_has_api_routes(self) -> None:
        """Test that app has API routes mounted."""
        test_app = create_app()
        route_paths = [str(r.path) for r in test_app.routes if hasattr(r, "path")]
        assert any("/api" in p for p in route_paths)

    def test_app_has_lifespan(self) -> None:
        """Test that app has lifespan handler."""
        test_app = create_app()
        assert test_app.router.lifespan_context is not None


class TestCreateTestApp:
    """Tests for create_test_app function."""

    def test_returns_fastapi_instance(self) -> None:
        """Test that create_test_app returns a FastAPI instance."""
        test_app = create_test_app()
        assert isinstance(test_app, FastAPI)

    def test_default_in_memory_database(self) -> None:
        """Test that create_test_app uses in-memory database by default."""
        test_app = create_test_app()
        config = test_app.state.config
        assert config.database_url == "sqlite+aiosqlite:///:memory:"

    def test_custom_database_url(self) -> None:
        """Test create_test_app with custom database URL."""
        test_app = create_test_app(database_url="sqlite+aiosqlite:///test.db")
        config = test_app.state.config
        assert config.database_url == "sqlite+aiosqlite:///test.db"

    def test_debug_enabled(self) -> None:
        """Test that create_test_app enables debug mode."""
        test_app = create_test_app()
        assert test_app.debug is True

    def test_test_secret_key(self) -> None:
        """Test that create_test_app uses test secret key."""
        test_app = create_test_app()
        config = test_app.state.config
        assert config.secret_key == "test-secret-key"


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.anyio
    async def test_lifespan_initializes_database(self) -> None:
        """Test that lifespan initializes database on startup."""
        test_app = create_test_app()

        with patch(
            "atp.dashboard.v2.factory.init_database", new_callable=AsyncMock
        ) as mock_init:
            async with lifespan(test_app):
                mock_init.assert_called_once()

    @pytest.mark.anyio
    async def test_lifespan_uses_config_from_app_state(self) -> None:
        """Test that lifespan uses config from app state."""
        config = DashboardConfig(
            database_url="postgresql+asyncpg://localhost/test",
            database_echo=True,
        )
        test_app = create_app(config=config)

        with patch(
            "atp.dashboard.v2.factory.init_database", new_callable=AsyncMock
        ) as mock_init:
            async with lifespan(test_app):
                mock_init.assert_called_once_with(
                    url="postgresql+asyncpg://localhost/test",
                    echo=True,
                )


class TestDefaultApp:
    """Tests for default app instance."""

    def test_app_is_fastapi_instance(self) -> None:
        """Test that default app is a FastAPI instance."""
        assert isinstance(app, FastAPI)

    def test_app_has_expected_title(self) -> None:
        """Test that default app has expected title."""
        assert app.title == "ATP Dashboard"


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_cors_with_wildcard_origin(self) -> None:
        """Test CORS configuration with wildcard origin."""
        config = DashboardConfig(cors_origins="*")
        test_app = create_app(config=config)
        middleware_classes = [m.cls.__name__ for m in test_app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_cors_with_multiple_origins(self) -> None:
        """Test CORS configuration with multiple origins."""
        config = DashboardConfig(
            cors_origins="http://localhost:3000,http://localhost:8080"
        )
        test_app = create_app(config=config)
        # CORS middleware should be present
        middleware_classes = [m.cls.__name__ for m in test_app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestAppRoutes:
    """Tests for app route configuration."""

    def test_api_router_mounted(self) -> None:
        """Test that API router is mounted at /api."""
        test_app = create_app()
        route_paths = [str(r.path) for r in test_app.routes if hasattr(r, "path")]
        # Check that there are routes under /api
        api_routes = [p for p in route_paths if p.startswith("/api")]
        assert len(api_routes) > 0

    def test_dashboard_routes_exist(self) -> None:
        """Test that dashboard API routes exist."""
        test_app = create_app()
        route_paths = [str(r.path) for r in test_app.routes if hasattr(r, "path")]
        assert any("/api/dashboard" in p for p in route_paths)

    def test_agents_routes_exist(self) -> None:
        """Test that agents API routes exist."""
        test_app = create_app()
        route_paths = [str(r.path) for r in test_app.routes if hasattr(r, "path")]
        assert any("/api/agents" in p for p in route_paths)

    def test_suites_routes_exist(self) -> None:
        """Test that suites API routes exist."""
        test_app = create_app()
        route_paths = [str(r.path) for r in test_app.routes if hasattr(r, "path")]
        assert any("/api/suites" in p for p in route_paths)
