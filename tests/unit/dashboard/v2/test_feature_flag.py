"""Tests for dashboard app routing."""

from fastapi import FastAPI


class TestAppRouting:
    """Tests for app routing (v2 is now the only version)."""

    def test_create_app_returns_v2(self) -> None:
        """Test that create_app returns v2 app."""
        from atp.dashboard import create_app

        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.version == "0.2.0"

    def test_v2_create_app_returns_fastapi(self) -> None:
        """Test that v2 create_app returns FastAPI instance."""
        from atp.dashboard.v2 import create_app

        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "ATP Dashboard"


class TestAppProxy:
    """Tests for the _AppProxy class."""

    def test_proxy_getattr_resolves_app(self) -> None:
        """Test that proxy resolves app on attribute access."""
        from atp.dashboard import _get_app

        app = _get_app()
        title = app.title
        assert title == "ATP Dashboard"

    def test_proxy_returns_consistent_app(self) -> None:
        """Test that _get_app returns consistent app."""
        from atp.dashboard import _get_app

        app1 = _get_app()
        app2 = _get_app()
        assert app1.title == app2.title
