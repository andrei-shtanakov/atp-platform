"""Tests for ATP Dashboard application configuration."""

from atp.dashboard.v2.factory import create_app


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_is_fastapi_instance(self) -> None:
        """Test that app is a FastAPI instance."""
        app = create_app()
        assert app is not None
        assert hasattr(app, "router")

    def test_app_has_title(self) -> None:
        """Test that app has correct title."""
        app = create_app()
        assert app.title == "ATP Dashboard"

    def test_app_includes_api_router(self) -> None:
        """Test that app includes API router."""
        app = create_app()
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert any("/api" in str(p) for p in route_paths)

    def test_app_has_cors_middleware(self) -> None:
        """Test that app has CORS middleware."""
        app = create_app()
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestAppRoutes:
    """Tests for app routes."""

    def test_root_route_exists(self) -> None:
        """Test that root route exists."""
        app = create_app()
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/" in route_paths

    def test_api_routes_exist(self) -> None:
        """Test that API routes exist."""
        app = create_app()
        route_paths = [str(r.path) for r in app.routes if hasattr(r, "path")]
        assert any("/api/dashboard" in p for p in route_paths)
        assert any("/api/agents" in p for p in route_paths)
        assert any("/api/suites" in p for p in route_paths)
