"""Tests for feature flag routing in ATP Dashboard."""

import os
from unittest.mock import patch

from fastapi import FastAPI


class TestFeatureFlagRouting:
    """Tests for feature flag based app routing."""

    def test_v1_app_by_default(self) -> None:
        """Test that v1 app is used by default."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ATP_DASHBOARD_V2", None)
            # Import fresh to test routing
            from atp.dashboard import _is_v2_enabled

            assert _is_v2_enabled() is False

    def test_v2_app_when_enabled(self) -> None:
        """Test that v2 app is used when feature flag is enabled."""
        with patch.dict(os.environ, {"ATP_DASHBOARD_V2": "true"}):
            from atp.dashboard import _is_v2_enabled

            assert _is_v2_enabled() is True

    def test_create_app_returns_v1_by_default(self) -> None:
        """Test that create_app returns v1 app by default."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ATP_DASHBOARD_V2", None)
            from atp.dashboard import create_app

            app = create_app()
            assert isinstance(app, FastAPI)
            # v1 app has specific version
            assert app.version == "0.1.0"

    def test_create_app_returns_v2_when_enabled(self) -> None:
        """Test that create_app returns v2 app when feature flag is enabled."""
        with patch.dict(os.environ, {"ATP_DASHBOARD_V2": "true"}):
            from atp.dashboard import create_app

            app = create_app()
            assert isinstance(app, FastAPI)
            # v2 app has specific version
            assert app.version == "0.2.0"


class TestAppProxy:
    """Tests for the _AppProxy class."""

    def test_proxy_getattr_resolves_app(self) -> None:
        """Test that proxy resolves app on attribute access."""
        # Import the actual get_app function to test
        from atp.dashboard import _get_app

        app = _get_app()
        # Access an attribute to trigger resolution
        title = app.title
        assert title == "ATP Dashboard"

    def test_proxy_returns_consistent_app(self) -> None:
        """Test that _get_app returns consistent app instance."""
        from atp.dashboard import _get_app

        # Multiple accesses should return same underlying app
        app1 = _get_app()
        app2 = _get_app()
        # Both should have the same title
        assert app1.title == app2.title
