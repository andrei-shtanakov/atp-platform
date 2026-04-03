"""Tests for dashboard UI routes."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def app():
    """Create test app with UI routes."""
    return create_test_app()


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUIRoutes:
    """Tests for UI page routes."""

    @pytest.mark.anyio
    async def test_home_returns_html(self, client) -> None:
        """GET /ui/ returns HTML page."""
        resp = await client.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "ATP Platform" in resp.text

    @pytest.mark.anyio
    async def test_benchmarks_returns_html(self, client) -> None:
        """GET /ui/benchmarks returns HTML page."""
        resp = await client.get("/ui/benchmarks")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Benchmarks" in resp.text

    @pytest.mark.anyio
    async def test_placeholder_games(self, client) -> None:
        """GET /ui/games returns placeholder page."""
        resp = await client.get("/ui/games")
        assert resp.status_code == 200
        assert "Coming soon" in resp.text

    @pytest.mark.anyio
    async def test_placeholder_runs(self, client) -> None:
        """GET /ui/runs returns placeholder page."""
        resp = await client.get("/ui/runs")
        assert resp.status_code == 200
        assert "Coming soon" in resp.text

    @pytest.mark.anyio
    async def test_placeholder_leaderboard(self, client) -> None:
        """GET /ui/leaderboard returns placeholder page."""
        resp = await client.get("/ui/leaderboard")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_placeholder_suites(self, client) -> None:
        """GET /ui/suites returns placeholder page."""
        resp = await client.get("/ui/suites")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_placeholder_analytics(self, client) -> None:
        """GET /ui/analytics returns placeholder page."""
        resp = await client.get("/ui/analytics")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_login_page(self, client) -> None:
        """GET /ui/login returns login page."""
        resp = await client.get("/ui/login")
        assert resp.status_code == 200
        assert "Login" in resp.text or "login" in resp.text

    @pytest.mark.anyio
    async def test_sidebar_has_all_nav_links(self, client) -> None:
        """Home page sidebar contains all navigation links."""
        resp = await client.get("/ui/")
        html = resp.text
        assert "/ui/benchmarks" in html
        assert "/ui/games" in html
        assert "/ui/runs" in html
        assert "/ui/leaderboard" in html
        assert "/ui/suites" in html
        assert "/ui/analytics" in html

    @pytest.mark.anyio
    async def test_htmx_loaded(self, client) -> None:
        """Pages include HTMX CDN script."""
        resp = await client.get("/ui/")
        assert "htmx.org" in resp.text

    @pytest.mark.anyio
    async def test_pico_css_loaded(self, client) -> None:
        """Pages include Pico CSS CDN."""
        resp = await client.get("/ui/")
        assert "pico" in resp.text.lower()


class TestBenchmarkDetail:
    """Tests for benchmark detail page."""

    @pytest.mark.anyio
    async def test_benchmark_not_found(self, client) -> None:
        """GET /ui/benchmarks/999 returns 404 error page."""
        resp = await client.get("/ui/benchmarks/999")
        assert resp.status_code == 404
        assert "Not Found" in resp.text
