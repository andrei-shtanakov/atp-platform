"""Tests for rate limiting module."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.rate_limit import (
    create_limiter,
    get_rate_limit_key,
    rate_limit_exceeded_handler,
)


class TestGetRateLimitKey:
    """Tests for rate limit key function."""

    def test_returns_user_id_when_jwt_present(self) -> None:
        request = MagicMock()
        request.state.user_id = "42"
        key = get_rate_limit_key(request)
        assert key == "user:42"

    def test_returns_ip_when_no_jwt(self) -> None:
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "192.168.1.1"
        key = get_rate_limit_key(request)
        assert key == "ip:192.168.1.1"

    def test_returns_ip_when_user_id_is_none(self) -> None:
        request = MagicMock()
        request.state.user_id = None
        request.headers = {}
        request.client.host = "10.0.0.1"
        key = get_rate_limit_key(request)
        assert key == "ip:10.0.0.1"

    def test_uses_x_forwarded_for_header(self) -> None:
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18"}
        key = get_rate_limit_key(request)
        assert key == "ip:203.0.113.50"

    def test_uses_single_x_forwarded_for(self) -> None:
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {"x-forwarded-for": "198.51.100.1"}
        key = get_rate_limit_key(request)
        assert key == "ip:198.51.100.1"


class TestCreateLimiter:
    """Tests for limiter factory."""

    def test_creates_limiter_with_default_config(self) -> None:
        config = DashboardConfig(debug=True)
        result = create_limiter(config)
        assert result is not None

    def test_disabled_limiter(self) -> None:
        config = DashboardConfig(debug=True, rate_limit_enabled=False)
        result = create_limiter(config)
        assert result is not None

    def test_sets_module_level_limiter(self) -> None:
        import atp.dashboard.v2.rate_limit as rl_mod

        config = DashboardConfig(debug=True)
        result = create_limiter(config)
        assert rl_mod.limiter is result

    def test_default_limits_are_limit_group_instances(self) -> None:
        """Regression: _default_limits must hold LimitGroup objects, not raw
        strings. Previously create_limiter() stored the raw rate_limit_default
        string, which caused itertools.chain(*_default_limits) inside slowapi's
        middleware to iterate the string character-by-character and crash with
        an AttributeError on any non-decorated route — masking itself behind
        slowapi's RateLimitExceeded handler (500 on /, /docs, /openapi.json).
        """
        from slowapi.wrappers import LimitGroup

        config = DashboardConfig(debug=True, rate_limit_default="60/minute")
        result = create_limiter(config)
        defaults = result._default_limits  # type: ignore[attr-defined]
        assert defaults, "default limits should not be empty"
        for group in defaults:
            assert isinstance(group, LimitGroup), (
                f"expected LimitGroup, got {type(group).__name__}"
            )
        # And iterating yields real Limit instances (what slowapi does
        # internally via itertools.chain).
        import itertools

        from slowapi.wrappers import Limit

        flat = list(itertools.chain(*defaults))
        assert flat, "chained defaults should yield at least one Limit"
        for limit in flat:
            assert isinstance(limit, Limit)


class TestRateLimitEnabledNonDecoratedRoute:
    """Regression: non-decorated routes must not 500 when rate limiting is
    enabled. This reproduces the prod bug where /, /docs, /openapi.json all
    returned 500 because create_limiter() stored raw strings in _default_limits.
    """

    def test_non_decorated_route_does_not_500(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware

        from atp.dashboard.v2.rate_limit import (
            create_limiter,
            rate_limit_exceeded_handler,
        )

        config = DashboardConfig(
            debug=True,
            rate_limit_enabled=True,
            rate_limit_default="60/minute",
        )
        limiter = create_limiter(config)

        app = FastAPI()
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

        @app.get("/plain")
        async def plain() -> dict[str, bool]:
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/plain")
        assert resp.status_code == 200, (
            f"non-decorated route returned {resp.status_code}: {resp.text}"
        )


class TestRateLimitExceededHandler:
    """Tests for 429 response handler."""

    @staticmethod
    def _make_exc(detail: str) -> RateLimitExceeded:
        """Create a RateLimitExceeded with a given detail string."""
        exc = MagicMock(spec=RateLimitExceeded)
        exc.detail = detail
        exc.retry_after = 60
        return exc

    @pytest.mark.anyio
    async def test_returns_429_with_json(self) -> None:
        request = MagicMock()
        exc = self._make_exc("5 per 1 minute")
        response = await rate_limit_exceeded_handler(request, exc)
        assert response.status_code == 429
        body = json.loads(response.body)
        assert body["error"] == "rate_limit_exceeded"
        assert "5 per 1 minute" in body["detail"]
        assert "retry_after" in body

    @pytest.mark.anyio
    async def test_includes_retry_after_header(self) -> None:
        request = MagicMock()
        exc = self._make_exc("10 per 1 hour")
        response = await rate_limit_exceeded_handler(request, exc)
        assert "Retry-After" in response.headers


class TestRateLimitIntegration:
    """Integration tests with FastAPI test client."""

    def test_rate_limit_returns_429(self) -> None:
        from slowapi import Limiter
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address

        app = FastAPI()
        test_limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["2/minute"],
            storage_uri="memory://",
        )
        app.state.limiter = test_limiter
        app.add_middleware(SlowAPIMiddleware)
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

        @app.get("/test")
        @test_limiter.limit("2/minute")
        async def test_endpoint(request: Request) -> dict[str, bool]:
            return {"ok": True}

        client = TestClient(app)
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 200
        resp = client.get("/test")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        body = resp.json()
        assert body["error"] == "rate_limit_exceeded"


class TestRateLimitWithApp:
    """Test rate limiting with the real app factory."""

    def test_app_has_limiter_in_state(self) -> None:
        """App factory sets up limiter in app.state."""
        from atp.dashboard.v2.factory import create_test_app

        app = create_test_app()
        assert hasattr(app.state, "limiter")
        assert app.state.limiter is not None

    def test_rate_limit_disabled(self) -> None:
        """Rate limiting can be disabled via config."""
        from atp.dashboard.v2.factory import create_app

        config = DashboardConfig(
            debug=True,
            rate_limit_enabled=False,
        )
        app = create_app(config=config)
        client = TestClient(app)
        # Should never get 429 even with many requests
        for _ in range(20):
            resp = client.get("/ui/login")
            assert resp.status_code != 429
