"""Detailed tests for ATP Dashboard API endpoints with mocking."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.api import (
    get_session,
    router,
)
from atp.dashboard.schemas import (
    AgentCreate,
    AgentResponse,
    AgentUpdate,
    SuiteExecutionSummary,
    Token,
    UserCreate,
)


class TestGetSession:
    """Tests for get_session dependency."""

    @pytest.mark.anyio
    async def test_get_session_yields_session(self) -> None:
        """Test that get_session yields a session."""
        mock_session = MagicMock(spec=AsyncSession)
        mock_db = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_db.session.return_value = mock_context

        with patch("atp.dashboard.api.get_database", return_value=mock_db):
            async for session in get_session():
                assert session is mock_session


class TestAgentSchemas:
    """Tests for agent schema validation."""

    def test_agent_create_valid(self) -> None:
        """Test valid agent create schema."""
        agent = AgentCreate(
            name="test-agent",
            agent_type="http",
            config={"url": "http://localhost"},
        )
        assert agent.name == "test-agent"
        assert agent.agent_type == "http"

    def test_agent_create_defaults(self) -> None:
        """Test agent create with defaults."""
        agent = AgentCreate(name="test", agent_type="cli")
        assert agent.config == {}
        assert agent.description is None

    def test_agent_update_partial(self) -> None:
        """Test partial agent update."""
        update = AgentUpdate(description="New description")
        assert update.description == "New description"
        assert update.agent_type is None
        assert update.config is None


class TestAuthSchemas:
    """Tests for auth schema validation."""

    def test_user_create_valid(self) -> None:
        """Test valid user create schema."""
        user = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_token_default_type(self) -> None:
        """Test token default type."""
        token = Token(access_token="abc123")
        assert token.token_type == "bearer"


class TestSuiteExecutionSchemas:
    """Tests for suite execution schemas."""

    def test_suite_execution_summary_from_model(self) -> None:
        """Test creating summary from model attributes."""
        now = datetime.now()
        summary = SuiteExecutionSummary(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            agent_name="test-agent",
            started_at=now,
            completed_at=now,
            duration_seconds=10.5,
            runs_per_test=3,
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            success_rate=0.8,
            status="completed",
            error=None,
        )
        assert summary.success_rate == 0.8
        assert summary.total_tests == 5


class TestAgentResponseSchema:
    """Tests for agent response schema."""

    def test_agent_response_from_attributes(self) -> None:
        """Test creating response from attributes."""
        now = datetime.now()
        response = AgentResponse(
            id=1,
            name="test-agent",
            agent_type="http",
            config={},
            description=None,
            created_at=now,
            updated_at=now,
        )
        assert response.id == 1
        assert response.name == "test-agent"


class TestRouterRoutes:
    """Tests for router routes configuration."""

    def test_auth_routes_exist(self) -> None:
        """Test that auth routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/auth/token" in paths
        assert "/auth/register" in paths
        assert "/auth/me" in paths

    def test_agent_routes_exist(self) -> None:
        """Test that agent routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/agents" in paths
        assert "/agents/{agent_id}" in paths

    def test_suite_routes_exist(self) -> None:
        """Test that suite routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/suites" in paths
        assert "/suites/{execution_id}" in paths

    def test_test_routes_exist(self) -> None:
        """Test that test routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/tests" in paths
        assert "/tests/{execution_id}" in paths

    def test_trend_routes_exist(self) -> None:
        """Test that trend routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/trends/suite" in paths
        assert "/trends/test" in paths

    def test_comparison_routes_exist(self) -> None:
        """Test that comparison routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/compare/agents" in paths

    def test_dashboard_routes_exist(self) -> None:
        """Test that dashboard routes exist."""
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/dashboard/summary" in paths
