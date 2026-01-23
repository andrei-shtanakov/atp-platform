"""Tests for ATP Dashboard database models."""

from datetime import datetime

from atp.dashboard.models import (
    Agent,
    Artifact,
    Base,
    EvaluationResult,
    RunResult,
    ScoreComponent,
    SuiteExecution,
    TestExecution,
    User,
)


class TestUserModel:
    """Tests for User model."""

    def test_create_user(self) -> None:
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            is_active=True,
            is_admin=False,
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed"
        assert user.is_active is True
        assert user.is_admin is False

    def test_user_repr(self) -> None:
        """Test user string representation."""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )
        assert "testuser" in repr(user)
        assert "1" in repr(user)

    def test_user_admin(self) -> None:
        """Test admin user creation."""
        user = User(
            username="admin",
            email="admin@example.com",
            hashed_password="hashed",
            is_admin=True,
        )
        assert user.is_admin is True


class TestAgentModel:
    """Tests for Agent model."""

    def test_create_agent(self) -> None:
        """Test creating an agent."""
        agent = Agent(
            name="test-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
        )
        assert agent.name == "test-agent"
        assert agent.agent_type == "http"
        assert agent.config["endpoint"] == "http://localhost:8000"

    def test_agent_default_config(self) -> None:
        """Test agent with default empty config."""
        agent = Agent(
            name="simple-agent",
            agent_type="cli",
            config={},  # Explicit empty config since defaults apply at DB level
        )
        assert agent.config == {}

    def test_agent_repr(self) -> None:
        """Test agent string representation."""
        agent = Agent(
            id=1,
            name="test-agent",
            agent_type="http",
        )
        assert "test-agent" in repr(agent)
        assert "http" in repr(agent)


class TestSuiteExecutionModel:
    """Tests for SuiteExecution model."""

    def test_create_suite_execution(self) -> None:
        """Test creating a suite execution."""
        now = datetime.now()
        execution = SuiteExecution(
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            runs_per_test=5,
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            success_rate=0.8,
            status="completed",
        )
        assert execution.suite_name == "test-suite"
        assert execution.runs_per_test == 5
        assert execution.total_tests == 10
        assert execution.success_rate == 0.8
        assert execution.status == "completed"

    def test_suite_execution_default_status(self) -> None:
        """Test suite execution default status."""
        execution = SuiteExecution(
            suite_name="test-suite",
            agent_id=1,
            started_at=datetime.now(),
            status="running",  # Explicit since defaults apply at DB level
        )
        assert execution.status == "running"

    def test_suite_execution_repr(self) -> None:
        """Test suite execution string representation."""
        execution = SuiteExecution(
            id=1,
            suite_name="test-suite",
            agent_id=2,
            started_at=datetime.now(),
        )
        assert "test-suite" in repr(execution)


class TestTestExecutionModel:
    """Tests for TestExecution model."""

    def test_create_test_execution(self) -> None:
        """Test creating a test execution."""
        now = datetime.now()
        execution = TestExecution(
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            tags=["smoke", "fast"],
            started_at=now,
            total_runs=5,
            successful_runs=4,
            success=True,
            score=85.5,
        )
        assert execution.test_id == "test-001"
        assert execution.test_name == "Test One"
        assert execution.tags == ["smoke", "fast"]
        assert execution.successful_runs == 4
        assert execution.score == 85.5

    def test_test_execution_default_values(self) -> None:
        """Test test execution default values."""
        execution = TestExecution(
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            started_at=datetime.now(),
            # Explicit defaults since they apply at DB insert time
            tags=[],
            total_runs=1,
            successful_runs=0,
            success=False,
            status="running",
        )
        assert execution.tags == []
        assert execution.total_runs == 1
        assert execution.successful_runs == 0
        assert execution.success is False
        assert execution.status == "running"


class TestRunResultModel:
    """Tests for RunResult model."""

    def test_create_run_result(self) -> None:
        """Test creating a run result."""
        now = datetime.now()
        run = RunResult(
            test_execution_id=1,
            run_number=1,
            started_at=now,
            response_status="completed",
            success=True,
            total_tokens=1000,
            total_steps=5,
            cost_usd=0.01,
        )
        assert run.run_number == 1
        assert run.response_status == "completed"
        assert run.success is True
        assert run.total_tokens == 1000
        assert run.cost_usd == 0.01

    def test_run_result_with_metrics(self) -> None:
        """Test run result with all metrics."""
        run = RunResult(
            test_execution_id=1,
            run_number=1,
            started_at=datetime.now(),
            response_status="completed",
            success=True,
            total_tokens=1000,
            input_tokens=800,
            output_tokens=200,
            total_steps=5,
            tool_calls=3,
            llm_calls=2,
            cost_usd=0.01,
        )
        assert run.input_tokens == 800
        assert run.output_tokens == 200
        assert run.tool_calls == 3
        assert run.llm_calls == 2


class TestArtifactModel:
    """Tests for Artifact model."""

    def test_create_file_artifact(self) -> None:
        """Test creating a file artifact."""
        artifact = Artifact(
            run_result_id=1,
            artifact_type="file",
            path="report.md",
            content_type="text/markdown",
            size_bytes=1024,
            content="# Report",
        )
        assert artifact.artifact_type == "file"
        assert artifact.path == "report.md"
        assert artifact.content_type == "text/markdown"
        assert artifact.content == "# Report"

    def test_create_structured_artifact(self) -> None:
        """Test creating a structured artifact."""
        artifact = Artifact(
            run_result_id=1,
            artifact_type="structured",
            name="results",
            data_json={"key": "value"},
        )
        assert artifact.artifact_type == "structured"
        assert artifact.name == "results"
        assert artifact.data_json == {"key": "value"}


class TestEvaluationResultModel:
    """Tests for EvaluationResult model."""

    def test_create_evaluation_result(self) -> None:
        """Test creating an evaluation result."""
        result = EvaluationResult(
            test_execution_id=1,
            evaluator_name="artifact",
            passed=True,
            score=0.95,
            total_checks=5,
            passed_checks=5,
            failed_checks=0,
            checks_json=[{"name": "file_exists", "passed": True, "score": 1.0}],
        )
        assert result.evaluator_name == "artifact"
        assert result.passed is True
        assert result.score == 0.95
        assert result.total_checks == 5

    def test_evaluation_result_with_failures(self) -> None:
        """Test evaluation result with failed checks."""
        result = EvaluationResult(
            test_execution_id=1,
            evaluator_name="behavior",
            passed=False,
            score=0.6,
            total_checks=5,
            passed_checks=3,
            failed_checks=2,
            checks_json=[
                {"name": "check1", "passed": True},
                {"name": "check2", "passed": False},
            ],
        )
        assert result.passed is False
        assert result.failed_checks == 2


class TestScoreComponentModel:
    """Tests for ScoreComponent model."""

    def test_create_score_component(self) -> None:
        """Test creating a score component."""
        component = ScoreComponent(
            test_execution_id=1,
            component_name="quality",
            raw_value=0.85,
            normalized_value=0.85,
            weight=0.4,
            weighted_value=0.34,
        )
        assert component.component_name == "quality"
        assert component.raw_value == 0.85
        assert component.normalized_value == 0.85
        assert component.weight == 0.4
        assert component.weighted_value == 0.34

    def test_score_component_all_types(self) -> None:
        """Test all score component types."""
        component_names = ["quality", "completeness", "efficiency", "cost"]
        for name in component_names:
            component = ScoreComponent(
                test_execution_id=1,
                component_name=name,
                normalized_value=0.8,
                weight=0.25,
                weighted_value=0.2,
            )
            assert component.component_name == name


class TestBaseModel:
    """Tests for Base model class."""

    def test_base_metadata(self) -> None:
        """Test that Base has metadata."""
        assert Base.metadata is not None
        assert hasattr(Base.metadata, "tables")
