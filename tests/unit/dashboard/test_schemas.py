"""Tests for ATP Dashboard API schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from atp.dashboard.schemas import (
    AgentComparisonMetrics,
    AgentComparisonResponse,
    AgentCreate,
    AgentResponse,
    AgentUpdate,
    DashboardSummary,
    EvaluationResultResponse,
    RunResultSummary,
    ScoreComponentResponse,
    SuiteExecutionDetail,
    SuiteExecutionList,
    SuiteExecutionSummary,
    SuiteTrend,
    TestComparisonMetrics,
    TestExecutionDetail,
    TestExecutionSummary,
    TestTrend,
    Token,
    TokenData,
    TrendDataPoint,
    TrendResponse,
    UserCreate,
    UserResponse,
)


class TestAgentSchemas:
    """Tests for agent-related schemas."""

    def test_agent_create_valid(self) -> None:
        """Test valid agent creation schema."""
        agent = AgentCreate(
            name="test-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
            description="Test agent",
        )
        assert agent.name == "test-agent"
        assert agent.agent_type == "http"
        assert agent.description == "Test agent"

    def test_agent_create_minimal(self) -> None:
        """Test minimal agent creation schema."""
        agent = AgentCreate(
            name="test-agent",
            agent_type="http",
        )
        assert agent.config == {}
        assert agent.description is None

    def test_agent_create_empty_name_fails(self) -> None:
        """Test that empty name fails validation."""
        with pytest.raises(ValidationError):
            AgentCreate(name="", agent_type="http")

    def test_agent_update(self) -> None:
        """Test agent update schema."""
        update = AgentUpdate(
            agent_type="cli",
            config={"command": "python agent.py"},
        )
        assert update.agent_type == "cli"

    def test_agent_update_partial(self) -> None:
        """Test partial agent update."""
        update = AgentUpdate(description="Updated description")
        assert update.agent_type is None
        assert update.config is None
        assert update.description == "Updated description"

    def test_agent_response(self) -> None:
        """Test agent response schema."""
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


class TestSuiteExecutionSchemas:
    """Tests for suite execution schemas."""

    def test_suite_execution_summary(self) -> None:
        """Test suite execution summary schema."""
        now = datetime.now()
        summary = SuiteExecutionSummary(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            completed_at=now,
            duration_seconds=10.5,
            runs_per_test=5,
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            success_rate=0.8,
            status="completed",
            error=None,
        )
        assert summary.success_rate == 0.8
        assert summary.total_tests == 10

    def test_suite_execution_detail(self) -> None:
        """Test suite execution detail schema."""
        now = datetime.now()
        detail = SuiteExecutionDetail(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            completed_at=now,
            duration_seconds=10.5,
            runs_per_test=5,
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            success_rate=0.8,
            status="completed",
            error=None,
            tests=[],
        )
        assert detail.tests == []

    def test_suite_execution_list(self) -> None:
        """Test suite execution list schema."""
        listing = SuiteExecutionList(
            total=100,
            items=[],
            limit=50,
            offset=0,
        )
        assert listing.total == 100
        assert listing.limit == 50


class TestTestExecutionSchemas:
    """Tests for test execution schemas."""

    def test_test_execution_summary(self) -> None:
        """Test test execution summary schema."""
        now = datetime.now()
        summary = TestExecutionSummary(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            tags=["smoke"],
            started_at=now,
            completed_at=now,
            duration_seconds=5.5,
            total_runs=5,
            successful_runs=4,
            success=True,
            score=85.0,
            status="completed",
            error=None,
        )
        assert summary.test_id == "test-001"
        assert summary.score == 85.0

    def test_run_result_summary(self) -> None:
        """Test run result summary schema."""
        now = datetime.now()
        summary = RunResultSummary(
            id=1,
            run_number=1,
            started_at=now,
            completed_at=now,
            duration_seconds=2.5,
            response_status="completed",
            success=True,
            error=None,
            total_tokens=1000,
            input_tokens=800,
            output_tokens=200,
            total_steps=5,
            tool_calls=3,
            llm_calls=2,
            cost_usd=0.01,
        )
        assert summary.total_tokens == 1000

    def test_evaluation_result_response(self) -> None:
        """Test evaluation result response schema."""
        response = EvaluationResultResponse(
            id=1,
            evaluator_name="artifact",
            passed=True,
            score=0.95,
            total_checks=5,
            passed_checks=5,
            failed_checks=0,
            checks_json=[{"name": "check1", "passed": True}],
        )
        assert response.evaluator_name == "artifact"
        assert response.passed is True

    def test_score_component_response(self) -> None:
        """Test score component response schema."""
        response = ScoreComponentResponse(
            id=1,
            component_name="quality",
            raw_value=0.9,
            normalized_value=0.9,
            weight=0.4,
            weighted_value=0.36,
            details_json=None,
        )
        assert response.component_name == "quality"

    def test_test_execution_detail(self) -> None:
        """Test test execution detail schema."""
        now = datetime.now()
        detail = TestExecutionDetail(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            tags=["smoke"],
            started_at=now,
            completed_at=now,
            duration_seconds=5.5,
            total_runs=5,
            successful_runs=4,
            success=True,
            score=85.0,
            status="completed",
            error=None,
            runs=[],
            evaluations=[],
            score_components=[],
        )
        assert detail.runs == []
        assert detail.evaluations == []


class TestTrendSchemas:
    """Tests for trend-related schemas."""

    def test_trend_data_point(self) -> None:
        """Test trend data point schema."""
        now = datetime.now()
        point = TrendDataPoint(
            timestamp=now,
            value=0.85,
            execution_id=1,
        )
        assert point.value == 0.85

    def test_test_trend(self) -> None:
        """Test test trend schema."""
        now = datetime.now()
        trend = TestTrend(
            test_id="test-001",
            test_name="Test One",
            data_points=[
                TrendDataPoint(timestamp=now, value=0.8, execution_id=1),
            ],
            metric="score",
        )
        assert len(trend.data_points) == 1

    def test_suite_trend(self) -> None:
        """Test suite trend schema."""
        now = datetime.now()
        trend = SuiteTrend(
            suite_name="test-suite",
            agent_name="test-agent",
            data_points=[
                TrendDataPoint(timestamp=now, value=0.85, execution_id=1),
            ],
            metric="success_rate",
        )
        assert trend.suite_name == "test-suite"

    def test_trend_response(self) -> None:
        """Test trend response schema."""
        response = TrendResponse(
            suite_trends=[],
            test_trends=[],
        )
        assert response.suite_trends == []


class TestComparisonSchemas:
    """Tests for comparison-related schemas."""

    def test_agent_comparison_metrics(self) -> None:
        """Test agent comparison metrics schema."""
        metrics = AgentComparisonMetrics(
            agent_name="test-agent",
            total_executions=10,
            avg_success_rate=0.85,
            avg_score=82.5,
            avg_duration_seconds=5.5,
            latest_success_rate=0.9,
            latest_score=85.0,
        )
        assert metrics.avg_success_rate == 0.85

    def test_test_comparison_metrics(self) -> None:
        """Test test comparison metrics schema."""
        metrics = TestComparisonMetrics(
            test_id="test-001",
            test_name="Test One",
            metrics_by_agent={
                "agent-1": AgentComparisonMetrics(
                    agent_name="agent-1",
                    total_executions=5,
                    avg_success_rate=0.8,
                    avg_score=None,
                    avg_duration_seconds=None,
                    latest_success_rate=None,
                    latest_score=None,
                ),
            },
        )
        assert "agent-1" in metrics.metrics_by_agent

    def test_agent_comparison_response(self) -> None:
        """Test agent comparison response schema."""
        response = AgentComparisonResponse(
            suite_name="test-suite",
            agents=[],
            tests=[],
        )
        assert response.suite_name == "test-suite"


class TestAuthSchemas:
    """Tests for authentication schemas."""

    def test_user_create_valid(self) -> None:
        """Test valid user creation schema."""
        user = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_user_create_short_username_fails(self) -> None:
        """Test that short username fails validation."""
        with pytest.raises(ValidationError):
            UserCreate(
                username="ab",  # Too short
                email="test@example.com",
                password="password123",
            )

    def test_user_create_short_password_fails(self) -> None:
        """Test that short password fails validation."""
        with pytest.raises(ValidationError):
            UserCreate(
                username="testuser",
                email="test@example.com",
                password="short",  # Too short
            )

    def test_user_response(self) -> None:
        """Test user response schema."""
        now = datetime.now()
        response = UserResponse(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_admin=False,
            created_at=now,
        )
        assert response.id == 1
        assert response.is_active is True

    def test_token(self) -> None:
        """Test token schema."""
        token = Token(
            access_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        )
        assert token.token_type == "bearer"

    def test_token_data(self) -> None:
        """Test token data schema."""
        data = TokenData(
            username="testuser",
            user_id=1,
        )
        assert data.username == "testuser"


class TestDashboardSummary:
    """Tests for dashboard summary schema."""

    def test_dashboard_summary(self) -> None:
        """Test dashboard summary schema."""
        summary = DashboardSummary(
            total_agents=5,
            total_suites=10,
            total_executions=100,
            recent_success_rate=0.85,
            recent_avg_score=82.5,
            recent_executions=[],
        )
        assert summary.total_agents == 5
        assert summary.recent_success_rate == 0.85

    def test_dashboard_summary_no_score(self) -> None:
        """Test dashboard summary with no score."""
        summary = DashboardSummary(
            total_agents=0,
            total_suites=0,
            total_executions=0,
            recent_success_rate=0.0,
            recent_avg_score=None,
            recent_executions=[],
        )
        assert summary.recent_avg_score is None
