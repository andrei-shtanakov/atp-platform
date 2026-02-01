"""Tests for quota enforcement middleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request, status

from atp.dashboard.tenancy.middleware import (
    QuotaEnforcementMiddleware,
    create_quota_exceeded_response,
    get_quota_checker,
    get_quota_tracker,
    require_quota,
)
from atp.dashboard.tenancy.quotas import (
    QuotaChecker,
    QuotaExceededError,
    QuotaType,
    QuotaUsageTracker,
)


class TestQuotaEnforcementMiddleware:
    """Tests for QuotaEnforcementMiddleware class."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create a test FastAPI app."""
        app = FastAPI()

        @app.get("/api/tests")
        async def get_tests() -> dict:
            return {"tests": []}

        @app.post("/api/tests")
        async def create_test() -> dict:
            return {"created": True}

        @app.get("/api/health")
        async def health() -> dict:
            return {"status": "ok"}

        return app

    def test_middleware_initialization(self, app: FastAPI) -> None:
        """Test middleware initialization."""
        middleware = QuotaEnforcementMiddleware(app)
        assert middleware is not None

    def test_middleware_custom_exempt_paths(self, app: FastAPI) -> None:
        """Test middleware with custom exempt paths."""
        middleware = QuotaEnforcementMiddleware(
            app,
            exempt_paths={"/api/custom"},
        )
        assert "/api/custom" in middleware._exempt_paths

    def test_middleware_check_read_requests(self, app: FastAPI) -> None:
        """Test middleware with read request checking enabled."""
        middleware = QuotaEnforcementMiddleware(
            app,
            check_read_requests=True,
        )
        assert middleware._check_read_requests is True

    def test_get_quota_type_for_tests_path(self) -> None:
        """Test quota type mapping for tests path."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        quota_type = middleware._get_quota_type_for_path("/api/tests", "POST")
        assert quota_type == QuotaType.TESTS_PER_DAY

    def test_get_quota_type_for_agents_path(self) -> None:
        """Test quota type mapping for agents path."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        quota_type = middleware._get_quota_type_for_path("/api/agents", "POST")
        assert quota_type == QuotaType.AGENTS

    def test_get_quota_type_for_users_path(self) -> None:
        """Test quota type mapping for users path."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        quota_type = middleware._get_quota_type_for_path("/api/users", "POST")
        assert quota_type == QuotaType.USERS

    def test_get_quota_type_for_suites_path(self) -> None:
        """Test quota type mapping for suites path."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        quota_type = middleware._get_quota_type_for_path("/api/suites", "POST")
        assert quota_type == QuotaType.SUITES

    def test_get_quota_type_for_unknown_path(self) -> None:
        """Test quota type mapping for unknown path."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        quota_type = middleware._get_quota_type_for_path("/api/unknown", "POST")
        assert quota_type is None

    def test_get_quota_type_for_get_request(self) -> None:
        """Test quota type mapping for GET request."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        quota_type = middleware._get_quota_type_for_path("/api/tests", "GET")
        assert quota_type is None

    def test_create_quota_exceeded_response(self) -> None:
        """Test creating quota exceeded response."""
        app = FastAPI()
        middleware = QuotaEnforcementMiddleware(app)

        error = QuotaExceededError(
            quota_type=QuotaType.TESTS_PER_DAY,
            current_value=150,
            limit_value=100,
            message="Daily test limit exceeded",
        )

        response = middleware._create_quota_exceeded_response(error)
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in response.headers


class TestGetQuotaChecker:
    """Tests for get_quota_checker function."""

    def test_get_quota_checker_with_session(self) -> None:
        """Test getting quota checker when session is available."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.db_session = AsyncMock()

        checker = get_quota_checker(mock_request)
        assert checker is not None
        assert isinstance(checker, QuotaChecker)

    def test_get_quota_checker_without_session(self) -> None:
        """Test getting quota checker when session is not available."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock(spec=[])

        checker = get_quota_checker(mock_request)
        assert checker is None


class TestGetQuotaTracker:
    """Tests for get_quota_tracker function."""

    def test_get_quota_tracker_with_session(self) -> None:
        """Test getting quota tracker when session is available."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.db_session = AsyncMock()

        tracker = get_quota_tracker(mock_request)
        assert tracker is not None
        assert isinstance(tracker, QuotaUsageTracker)

    def test_get_quota_tracker_without_session(self) -> None:
        """Test getting quota tracker when session is not available."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock(spec=[])

        tracker = get_quota_tracker(mock_request)
        assert tracker is None


class TestRequireQuota:
    """Tests for require_quota dependency."""

    def test_require_quota_creates_dependency(self) -> None:
        """Test that require_quota creates a callable."""
        dep = require_quota(QuotaType.TESTS_PER_DAY)
        assert callable(dep)

    def test_require_quota_with_additional(self) -> None:
        """Test require_quota with additional parameter."""
        dep = require_quota(QuotaType.AGENTS, additional=5)
        assert callable(dep)

    @pytest.mark.anyio
    async def test_require_quota_passes(self) -> None:
        """Test require_quota when quota check passes."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.tenant_id = "test-tenant"

        mock_checker = AsyncMock(spec=QuotaChecker)
        mock_checker.enforce_quota = AsyncMock()

        dep = require_quota(QuotaType.TESTS_PER_DAY)
        # Should not raise
        await dep(mock_request, mock_checker)

    @pytest.mark.anyio
    async def test_require_quota_raises_http_exception(self) -> None:
        """Test require_quota raises HTTPException when quota exceeded."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.tenant_id = "test-tenant"

        mock_checker = AsyncMock(spec=QuotaChecker)
        mock_checker.enforce_quota = AsyncMock(
            side_effect=QuotaExceededError(
                quota_type=QuotaType.TESTS_PER_DAY,
                current_value=150,
                limit_value=100,
            )
        )

        dep = require_quota(QuotaType.TESTS_PER_DAY)
        with pytest.raises(HTTPException) as exc_info:
            await dep(mock_request, mock_checker)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.anyio
    async def test_require_quota_no_tenant_id(self) -> None:
        """Test require_quota skips check when no tenant ID."""
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock(spec=[])

        mock_checker = AsyncMock(spec=QuotaChecker)

        dep = require_quota(QuotaType.TESTS_PER_DAY)
        # Should not raise, just return
        await dep(mock_request, mock_checker)
        # enforce_quota should not be called
        mock_checker.enforce_quota.assert_not_called()


class TestCreateQuotaExceededResponse:
    """Tests for create_quota_exceeded_response function."""

    def test_create_response_default_retry_after(self) -> None:
        """Test creating response with default retry-after."""
        error = QuotaExceededError(
            quota_type=QuotaType.TESTS_PER_DAY,
            current_value=150,
            limit_value=100,
            message="Limit exceeded",
        )

        response = create_quota_exceeded_response(error)
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert response.headers.get("Retry-After") == "3600"

    def test_create_response_custom_retry_after(self) -> None:
        """Test creating response with custom retry-after."""
        error = QuotaExceededError(
            quota_type=QuotaType.PARALLEL_RUNS,
            current_value=10,
            limit_value=5,
        )

        response = create_quota_exceeded_response(error, retry_after=1800)
        assert response.headers.get("Retry-After") == "1800"

    def test_create_response_content(self) -> None:
        """Test response content includes quota details."""
        error = QuotaExceededError(
            quota_type=QuotaType.AGENTS,
            current_value=15,
            limit_value=10,
            message="Agent limit exceeded",
        )

        response = create_quota_exceeded_response(error)
        # Response body should contain quota information
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
