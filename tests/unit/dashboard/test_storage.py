"""Tests for ATP Dashboard storage module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.models import (
    Agent,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.storage import ResultStorage
from atp.evaluators.base import EvalCheck, EvalResult
from atp.protocol import ATPResponse, Metrics, ResponseStatus
from atp.scoring.models import ComponentScore, ScoreBreakdown, ScoredTestResult


class TestResultStorageAgent:
    """Tests for agent operations in ResultStorage."""

    @pytest.mark.anyio
    async def test_get_or_create_agent_new(self) -> None:
        """Test creating a new agent."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        agent = await storage.get_or_create_agent(
            name="test-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
        )

        assert agent.name == "test-agent"
        assert agent.agent_type == "http"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_get_or_create_agent_existing(self) -> None:
        """Test getting an existing agent."""
        existing_agent = Agent(
            id=1,
            name="existing-agent",
            agent_type="http",
            config={},
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_agent
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        agent = await storage.get_or_create_agent(
            name="existing-agent",
            agent_type="http",
        )

        assert agent is existing_agent
        mock_session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_get_agent_by_name(self) -> None:
        """Test getting agent by name."""
        existing_agent = Agent(
            id=1,
            name="test-agent",
            agent_type="http",
            config={},
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_agent
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        agent = await storage.get_agent_by_name("test-agent")

        assert agent is not None
        assert agent.name == "test-agent"

    @pytest.mark.anyio
    async def test_list_agents(self) -> None:
        """Test listing all agents."""
        agents = [
            Agent(id=1, name="agent-1", agent_type="http", config={}),
            Agent(id=2, name="agent-2", agent_type="cli", config={}),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = agents
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        result = await storage.list_agents()

        assert len(result) == 2
        assert result[0].name == "agent-1"


class TestResultStorageSuiteExecution:
    """Tests for suite execution operations."""

    @pytest.mark.anyio
    async def test_create_suite_execution(self) -> None:
        """Test creating a suite execution."""
        agent = Agent(id=1, name="test-agent", agent_type="http", config={})
        now = datetime.now()

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        execution = await storage.create_suite_execution(
            suite_name="test-suite",
            agent=agent,
            runs_per_test=5,
            started_at=now,
        )

        assert execution.suite_name == "test-suite"
        assert execution.agent_id == 1
        assert execution.runs_per_test == 5
        assert execution.status == "running"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_update_suite_execution(self) -> None:
        """Test updating a suite execution."""
        now = datetime.now()
        execution = SuiteExecution(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            status="running",
        )

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        updated = await storage.update_suite_execution(
            execution,
            completed_at=now,
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            success_rate=0.8,
            status="completed",
        )

        assert updated.total_tests == 10
        assert updated.passed_tests == 8
        assert updated.success_rate == 0.8
        assert updated.status == "completed"

    @pytest.mark.anyio
    async def test_list_suite_executions(self) -> None:
        """Test listing suite executions."""
        executions = [
            SuiteExecution(
                id=1,
                suite_name="suite-1",
                agent_id=1,
                started_at=datetime.now(),
            ),
            SuiteExecution(
                id=2,
                suite_name="suite-2",
                agent_id=1,
                started_at=datetime.now(),
            ),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = executions
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        result = await storage.list_suite_executions()

        assert len(result) == 2


class TestResultStorageTestExecution:
    """Tests for test execution operations."""

    @pytest.mark.anyio
    async def test_create_test_execution(self) -> None:
        """Test creating a test execution."""
        suite_exec = SuiteExecution(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=datetime.now(),
        )
        now = datetime.now()

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        execution = await storage.create_test_execution(
            suite_execution=suite_exec,
            test_id="test-001",
            test_name="Test One",
            tags=["smoke", "fast"],
            started_at=now,
            total_runs=5,
        )

        assert execution.test_id == "test-001"
        assert execution.test_name == "Test One"
        assert execution.tags == ["smoke", "fast"]
        assert execution.total_runs == 5
        mock_session.add.assert_called_once()

    @pytest.mark.anyio
    async def test_update_test_execution(self) -> None:
        """Test updating a test execution."""
        now = datetime.now()
        execution = TestExecution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            started_at=now,
            status="running",
        )

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        updated = await storage.update_test_execution(
            execution,
            completed_at=now,
            successful_runs=4,
            success=True,
            score=85.5,
            status="completed",
        )

        assert updated.successful_runs == 4
        assert updated.success is True
        assert updated.score == 85.5
        assert updated.status == "completed"


class TestResultStorageRunResult:
    """Tests for run result operations."""

    @pytest.mark.anyio
    async def test_create_run_result(self) -> None:
        """Test creating a run result."""
        test_exec = TestExecution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            started_at=datetime.now(),
        )

        response = ATPResponse(
            version="1.0",
            task_id="task-001",
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(
                total_tokens=1000,
                input_tokens=800,
                output_tokens=200,
                total_steps=5,
                tool_calls=3,
                llm_calls=2,
                cost_usd=0.01,
            ),
        )

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        run = await storage.create_run_result(
            test_execution=test_exec,
            run_number=1,
            response=response,
        )

        assert run.run_number == 1
        assert run.response_status == "completed"
        assert run.success is True
        assert run.total_tokens == 1000


class TestResultStorageEvaluation:
    """Tests for evaluation result operations."""

    @pytest.mark.anyio
    async def test_store_evaluation_results(self) -> None:
        """Test storing evaluation results."""
        test_exec = TestExecution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            started_at=datetime.now(),
        )

        eval_results = [
            EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(
                        name="file_exists",
                        passed=True,
                        score=1.0,
                        message="File exists",
                    ),
                ],
            ),
        ]

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        records = await storage.store_evaluation_results(test_exec, eval_results)

        assert len(records) == 1
        assert records[0].evaluator_name == "artifact"
        assert records[0].passed is True


class TestResultStorageScoring:
    """Tests for score component operations."""

    @pytest.mark.anyio
    async def test_store_scored_result(self) -> None:
        """Test storing scored result."""
        test_exec = TestExecution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            started_at=datetime.now(),
        )

        scored = ScoredTestResult(
            test_id="test-001",
            score=85.0,
            passed=True,
            breakdown=ScoreBreakdown(
                quality=ComponentScore(
                    name="quality",
                    normalized_value=0.9,
                    weight=0.4,
                    weighted_value=0.36,
                ),
                completeness=ComponentScore(
                    name="completeness",
                    normalized_value=0.8,
                    weight=0.3,
                    weighted_value=0.24,
                ),
                efficiency=ComponentScore(
                    name="efficiency",
                    normalized_value=0.7,
                    weight=0.2,
                    weighted_value=0.14,
                ),
                cost=ComponentScore(
                    name="cost",
                    normalized_value=0.6,
                    weight=0.1,
                    weighted_value=0.06,
                ),
            ),
        )

        mock_session = AsyncMock()
        storage = ResultStorage(mock_session)

        components = await storage.store_scored_result(test_exec, scored)

        assert len(components) == 4
        assert test_exec.score == 85.0


class TestResultStorageQueries:
    """Tests for query operations."""

    @pytest.mark.anyio
    async def test_get_historical_executions(self) -> None:
        """Test getting historical executions."""
        executions = [
            SuiteExecution(
                id=1,
                suite_name="test-suite",
                agent_id=1,
                started_at=datetime.now(),
            ),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = executions
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        result = await storage.get_historical_executions("test-suite")

        assert len(result) == 1
        assert result[0].suite_name == "test-suite"

    @pytest.mark.anyio
    async def test_compare_agents(self) -> None:
        """Test comparing agents."""
        executions = [
            SuiteExecution(
                id=1,
                suite_name="test-suite",
                agent_id=1,
                started_at=datetime.now(),
            ),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = executions
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        storage = ResultStorage(mock_session)
        result = await storage.compare_agents(
            suite_name="test-suite",
            agent_names=["agent-1", "agent-2"],
        )

        assert "agent-1" in result
        assert "agent-2" in result
