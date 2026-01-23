"""Tests for ATP Dashboard FastAPI application."""

from atp.dashboard.app import app, create_index_html


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_is_fastapi_instance(self) -> None:
        """Test that app is a FastAPI instance."""
        assert app is not None
        assert hasattr(app, "router")

    def test_app_has_title(self) -> None:
        """Test that app has correct title."""
        assert app.title == "ATP Dashboard"

    def test_app_includes_api_router(self) -> None:
        """Test that app includes API router."""
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert any("/api" in str(p) for p in route_paths)

    def test_app_has_cors_middleware(self) -> None:
        """Test that app has CORS middleware."""
        # Check middleware stack includes CORS
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestHTMLContent:
    """Tests for HTML content generation."""

    def test_create_index_html_returns_string(self) -> None:
        """Test that create_index_html returns HTML string."""
        html = create_index_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_content_has_doctype(self) -> None:
        """Test that HTML has DOCTYPE."""
        html = create_index_html()
        assert "<!DOCTYPE html>" in html

    def test_html_content_has_html_tag(self) -> None:
        """Test that HTML has html tag."""
        html = create_index_html()
        assert "<html" in html
        assert "</html>" in html

    def test_html_content_has_head(self) -> None:
        """Test that HTML has head section."""
        html = create_index_html()
        assert "<head>" in html
        assert "</head>" in html

    def test_html_content_has_body(self) -> None:
        """Test that HTML has body section."""
        html = create_index_html()
        assert "<body" in html
        assert "</body>" in html

    def test_html_content_has_title(self) -> None:
        """Test that HTML has title."""
        html = create_index_html()
        assert "<title>" in html
        assert "ATP Dashboard" in html

    def test_html_content_has_react(self) -> None:
        """Test that HTML includes React."""
        html = create_index_html()
        assert "react" in html.lower()

    def test_html_content_has_root_div(self) -> None:
        """Test that HTML has root div for React."""
        html = create_index_html()
        assert 'id="root"' in html

    def test_html_content_has_script(self) -> None:
        """Test that HTML has script tags."""
        html = create_index_html()
        assert "<script" in html
        assert "</script>" in html

    def test_html_content_has_style(self) -> None:
        """Test that HTML has style tags."""
        html = create_index_html()
        assert "<style>" in html

    def test_html_content_has_tailwind(self) -> None:
        """Test that HTML includes Tailwind CSS."""
        html = create_index_html()
        assert "tailwindcss" in html.lower()

    def test_html_content_has_chartjs(self) -> None:
        """Test that HTML includes Chart.js."""
        html = create_index_html()
        assert "chart.js" in html.lower()


class TestAppRoutes:
    """Tests for app routes."""

    def test_root_route_exists(self) -> None:
        """Test that root route exists."""
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/" in route_paths

    def test_api_routes_exist(self) -> None:
        """Test that API routes exist."""
        route_paths = [str(r.path) for r in app.routes if hasattr(r, "path")]
        # Check some API endpoints exist
        assert any("/api/dashboard" in p for p in route_paths)
        assert any("/api/agents" in p for p in route_paths)
        assert any("/api/suites" in p for p in route_paths)
