"""Unit tests for Dashboard v2 services.

This module contains tests for the service layer classes in the
ATP Dashboard v2 module.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.models import (
    Agent,
    SuiteDefinition,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.schemas import (
    AgentCreate,
    AgentUpdate,
)
from atp.dashboard.v2.services.agent_service import AgentService
from atp.dashboard.v2.services.comparison_service import ComparisonService
from atp.dashboard.v2.services.export_service import ExportService
from atp.dashboard.v2.services.test_service import TestService


def _make_agent(
    id: int = 1,
    name: str = "test-agent",
    agent_type: str = "http",
    config: dict | None = None,
    description: str | None = None,
) -> Agent:
    """Helper to create an Agent with required fields."""
    now = datetime.now()
    return Agent(
        id=id,
        name=name,
        agent_type=agent_type,
        config=config or {},
        description=description,
        created_at=now,
        updated_at=now,
    )


def _make_suite_execution(
    id: int = 1,
    suite_name: str = "test-suite",
    agent_id: int = 1,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    duration_seconds: float | None = None,
    runs_per_test: int = 1,
    total_tests: int = 1,
    passed_tests: int = 1,
    failed_tests: int = 0,
    success_rate: float = 1.0,
    status: str = "completed",
) -> SuiteExecution:
    """Helper to create a SuiteExecution with required fields."""
    now = datetime.now()
    return SuiteExecution(
        id=id,
        suite_name=suite_name,
        agent_id=agent_id,
        started_at=started_at or now,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        runs_per_test=runs_per_test,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        success_rate=success_rate,
        status=status,
    )


def _make_test_execution(
    id: int = 1,
    suite_execution_id: int = 1,
    test_id: str = "test-001",
    test_name: str = "Test One",
    tags: list[str] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    duration_seconds: float | None = None,
    total_runs: int = 1,
    successful_runs: int = 1,
    success: bool = True,
    score: float | None = None,
    status: str = "completed",
) -> TestExecution:
    """Helper to create a TestExecution with required fields."""
    now = datetime.now()
    return TestExecution(
        id=id,
        suite_execution_id=suite_execution_id,
        test_id=test_id,
        test_name=test_name,
        tags=tags or [],
        started_at=started_at or now,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        total_runs=total_runs,
        successful_runs=successful_runs,
        success=success,
        score=score,
        status=status,
    )


class TestAgentService:
    """Tests for AgentService."""

    @pytest.mark.anyio
    async def test_list_agents(self) -> None:
        """Test listing all agents."""
        agents = [
            _make_agent(id=1, name="agent-1", agent_type="http"),
            _make_agent(id=2, name="agent-2", agent_type="cli"),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = agents
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        service = AgentService(mock_session)
        result = await service.list_agents()

        assert len(result) == 2
        assert result[0].name == "agent-1"
        assert result[1].name == "agent-2"

    @pytest.mark.anyio
    async def test_get_agent(self) -> None:
        """Test getting agent by ID."""
        agent = _make_agent(
            id=1,
            name="test-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
            description="Test agent",
        )

        mock_session = AsyncMock()
        mock_session.get.return_value = agent

        service = AgentService(mock_session)
        result = await service.get_agent(1)

        assert result is not None
        assert result.id == 1
        assert result.name == "test-agent"
        mock_session.get.assert_called_once_with(Agent, 1)

    @pytest.mark.anyio
    async def test_get_agent_not_found(self) -> None:
        """Test getting agent that doesn't exist."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        service = AgentService(mock_session)
        result = await service.get_agent(999)

        assert result is None

    @pytest.mark.anyio
    async def test_get_agent_by_name(self) -> None:
        """Test getting agent by name."""
        agent = _make_agent(id=1, name="test-agent", agent_type="http")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = agent
        mock_session.execute.return_value = mock_result

        service = AgentService(mock_session)
        result = await service.get_agent_by_name("test-agent")

        assert result is not None
        assert result.name == "test-agent"

    @pytest.mark.anyio
    async def test_create_agent(self) -> None:
        """Test creating a new agent."""
        now = datetime.now()
        mock_session = AsyncMock()
        # Return None for existence check
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Mock refresh to set the datetime fields on the agent
        async def mock_refresh(agent: Agent) -> None:
            agent.id = 1
            agent.created_at = now
            agent.updated_at = now

        mock_session.refresh = mock_refresh

        agent_data = AgentCreate(
            name="new-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
            description="New test agent",
        )

        service = AgentService(mock_session)
        result = await service.create_agent(agent_data)

        assert result is not None
        assert result.name == "new-agent"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_create_agent_duplicate_name(self) -> None:
        """Test creating agent with duplicate name returns None."""
        existing_agent = _make_agent(id=1, name="existing-agent", agent_type="http")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_agent
        mock_session.execute.return_value = mock_result

        agent_data = AgentCreate(
            name="existing-agent",
            agent_type="http",
            config={},
        )

        service = AgentService(mock_session)
        result = await service.create_agent(agent_data)

        assert result is None
        mock_session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_update_agent(self) -> None:
        """Test updating an agent."""
        agent = _make_agent(
            id=1,
            name="test-agent",
            agent_type="http",
            config={"endpoint": "http://old"},
            description="Old description",
        )

        mock_session = AsyncMock()
        mock_session.get.return_value = agent

        update_data = AgentUpdate(
            agent_type="cli",
            config={"command": "python agent.py"},
            description="New description",
        )

        service = AgentService(mock_session)
        result = await service.update_agent(1, update_data)

        assert result is not None
        assert result.agent_type == "cli"
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_update_agent_not_found(self) -> None:
        """Test updating non-existent agent."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        update_data = AgentUpdate(agent_type="cli")

        service = AgentService(mock_session)
        result = await service.update_agent(999, update_data)

        assert result is None

    @pytest.mark.anyio
    async def test_delete_agent(self) -> None:
        """Test deleting an agent."""
        agent = _make_agent(id=1, name="test-agent", agent_type="http")

        mock_session = AsyncMock()
        mock_session.get.return_value = agent

        service = AgentService(mock_session)
        result = await service.delete_agent(1)

        assert result is True
        mock_session.delete.assert_called_once_with(agent)
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_delete_agent_not_found(self) -> None:
        """Test deleting non-existent agent."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        service = AgentService(mock_session)
        result = await service.delete_agent(999)

        assert result is False
        mock_session.delete.assert_not_called()

    @pytest.mark.anyio
    async def test_agent_exists(self) -> None:
        """Test checking if agent exists."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 1  # Agent ID
        mock_session.execute.return_value = mock_result

        service = AgentService(mock_session)
        result = await service.agent_exists("test-agent")

        assert result is True

    @pytest.mark.anyio
    async def test_agent_not_exists(self) -> None:
        """Test checking if agent doesn't exist."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        service = AgentService(mock_session)
        result = await service.agent_exists("nonexistent")

        assert result is False


class TestTestService:
    """Tests for TestService."""

    @pytest.mark.anyio
    async def test_list_suite_executions(self) -> None:
        """Test listing suite executions."""
        now = datetime.now()
        agent = _make_agent(id=1, name="test-agent", agent_type="http")
        executions = [
            _make_suite_execution(
                id=1,
                suite_name="test-suite",
                agent_id=1,
                started_at=now,
                completed_at=now + timedelta(hours=1),
                success_rate=0.9,
                status="completed",
            ),
        ]
        executions[0].agent = agent

        mock_session = AsyncMock()

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        # Mock list query
        mock_list_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = executions
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        service = TestService(mock_session)
        result = await service.list_suite_executions(limit=50, offset=0)

        assert result.total == 1
        assert len(result.items) == 1
        assert result.items[0].suite_name == "test-suite"
        assert result.items[0].agent_name == "test-agent"

    @pytest.mark.anyio
    async def test_list_suite_names(self) -> None:
        """Test listing unique suite names."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = ["suite-1", "suite-2", "suite-3"]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        service = TestService(mock_session)
        result = await service.list_suite_names()

        assert len(result) == 3
        assert "suite-1" in result

    @pytest.mark.anyio
    async def test_get_suite_execution(self) -> None:
        """Test getting suite execution details."""
        now = datetime.now()
        agent = _make_agent(id=1, name="test-agent", agent_type="http")
        test_exec = _make_test_execution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="First Test",
            started_at=now,
            success=True,
        )

        execution = _make_suite_execution(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            status="completed",
        )
        execution.agent = agent
        execution.test_executions = [test_exec]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = execution
        mock_session.execute.return_value = mock_result

        service = TestService(mock_session)
        result = await service.get_suite_execution(1)

        assert result is not None
        assert result.suite_name == "test-suite"
        assert result.agent_name == "test-agent"
        assert len(result.tests) == 1

    @pytest.mark.anyio
    async def test_get_suite_execution_not_found(self) -> None:
        """Test getting non-existent suite execution."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        service = TestService(mock_session)
        result = await service.get_suite_execution(999)

        assert result is None

    @pytest.mark.anyio
    async def test_list_test_executions(self) -> None:
        """Test listing test executions."""
        now = datetime.now()
        executions = [
            _make_test_execution(
                id=1,
                suite_execution_id=1,
                test_id="test-001",
                test_name="Test One",
                started_at=now,
                success=True,
                score=85.0,
            ),
        ]

        mock_session = AsyncMock()

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        # Mock list query
        mock_list_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = executions
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        service = TestService(mock_session)
        result = await service.list_test_executions(limit=50, offset=0)

        assert result.total == 1
        assert len(result.items) == 1
        assert result.items[0].test_id == "test-001"

    @pytest.mark.anyio
    async def test_get_test_execution(self) -> None:
        """Test getting test execution details."""
        now = datetime.now()
        execution = _make_test_execution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            started_at=now,
            success=True,
            score=85.0,
        )
        execution.run_results = []
        execution.evaluation_results = []
        execution.score_components = []

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = execution
        mock_session.execute.return_value = mock_result

        service = TestService(mock_session)
        result = await service.get_test_execution(1)

        assert result is not None
        assert result.test_id == "test-001"
        assert result.score == 85.0

    @pytest.mark.anyio
    async def test_get_test_execution_not_found(self) -> None:
        """Test getting non-existent test execution."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        service = TestService(mock_session)
        result = await service.get_test_execution(999)

        assert result is None


class TestComparisonService:
    """Tests for ComparisonService."""

    @pytest.mark.anyio
    async def test_format_event_summary_tool_call(self) -> None:
        """Test formatting tool_call event."""
        mock_session = AsyncMock()
        service = ComparisonService(mock_session)

        event = {
            "sequence": 1,
            "timestamp": "2024-01-01T10:00:00Z",
            "event_type": "tool_call",
            "payload": {"tool": "read_file", "status": "success"},
        }

        result = service._format_event_summary(event)

        assert result.event_type == "tool_call"
        assert "read_file" in result.summary
        assert result.sequence == 1

    @pytest.mark.anyio
    async def test_format_event_summary_llm_request(self) -> None:
        """Test formatting llm_request event."""
        mock_session = AsyncMock()
        service = ComparisonService(mock_session)

        event = {
            "sequence": 2,
            "timestamp": "2024-01-01T10:00:01Z",
            "event_type": "llm_request",
            "payload": {"model": "gpt-4", "input_tokens": 100, "output_tokens": 50},
        }

        result = service._format_event_summary(event)

        assert result.event_type == "llm_request"
        assert "gpt-4" in result.summary
        assert "150" in result.summary  # Total tokens

    @pytest.mark.anyio
    async def test_format_event_summary_error(self) -> None:
        """Test formatting error event."""
        mock_session = AsyncMock()
        service = ComparisonService(mock_session)

        event = {
            "sequence": 3,
            "timestamp": "2024-01-01T10:00:02Z",
            "event_type": "error",
            "payload": {"error_type": "ValidationError", "message": "Invalid input"},
        }

        result = service._format_event_summary(event)

        assert result.event_type == "error"
        assert "ValidationError" in result.summary

    @pytest.mark.anyio
    async def test_format_timeline_event(self) -> None:
        """Test formatting timeline event with relative time."""

        mock_session = AsyncMock()
        service = ComparisonService(mock_session)

        # Use timezone-aware datetime to match the parsed timestamp
        first_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        event = {
            "sequence": 2,
            "timestamp": "2024-01-01T10:00:05Z",
            "event_type": "tool_call",
            "payload": {"tool": "write_file", "duration_ms": 100},
        }

        result = service._format_timeline_event(event, first_time)

        assert result.sequence == 2
        assert result.relative_time_ms == 5000.0  # 5 seconds = 5000 ms
        assert result.duration_ms == 100

    @pytest.mark.anyio
    async def test_get_side_by_side_comparison_no_executions(self) -> None:
        """Test side-by-side comparison with no executions found."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        service = ComparisonService(mock_session)
        result = await service.get_side_by_side_comparison(
            suite_name="test-suite",
            test_id="test-001",
            agents=["agent-1", "agent-2"],
        )

        assert result is None


class TestExportService:
    """Tests for ExportService."""

    @pytest.mark.anyio
    async def test_export_results_to_csv(self) -> None:
        """Test exporting results to CSV."""
        now = datetime.now()
        agent = _make_agent(id=1, name="test-agent", agent_type="http")
        execution = _make_suite_execution(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            completed_at=now + timedelta(hours=1),
            duration_seconds=3600.0,
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            success_rate=0.8,
            status="completed",
        )
        execution.agent = agent
        execution.test_executions = []

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [execution]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        service = ExportService(mock_session)
        result = await service.export_results_to_csv()

        assert "suite_name" in result
        assert "test-suite" in result
        assert "test-agent" in result
        assert "0.8" in result

    @pytest.mark.anyio
    async def test_export_test_results_to_csv(self) -> None:
        """Test exporting test results to CSV."""
        now = datetime.now()
        execution = _make_test_execution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            tags=["integration", "api"],
            started_at=now,
            completed_at=now + timedelta(minutes=10),
            duration_seconds=600.0,
            total_runs=3,
            successful_runs=3,
            success=True,
            score=95.0,
            status="completed",
        )
        execution.suite_execution = None

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [execution]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        service = ExportService(mock_session)
        result = await service.export_test_results_to_csv()

        assert "test_id" in result
        assert "test-001" in result
        assert "Test One" in result
        assert "integration,api" in result
        assert "95.0" in result

    @pytest.mark.anyio
    async def test_export_results_to_json(self) -> None:
        """Test exporting results to JSON."""
        now = datetime.now()
        agent = _make_agent(id=1, name="test-agent", agent_type="http")
        test_exec = _make_test_execution(
            id=1,
            suite_execution_id=1,
            test_id="test-001",
            test_name="Test One",
            tags=[],
            started_at=now,
            success=True,
            score=90.0,
            status="completed",
        )
        execution = _make_suite_execution(
            id=1,
            suite_name="test-suite",
            agent_id=1,
            started_at=now,
            status="completed",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            success_rate=1.0,
        )
        execution.agent = agent
        execution.test_executions = [test_exec]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [execution]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        service = ExportService(mock_session)
        result = await service.export_results_to_json(include_tests=True)

        import json

        data = json.loads(result)

        assert "executions" in data
        assert len(data["executions"]) == 1
        assert data["executions"][0]["suite_name"] == "test-suite"
        assert "tests" in data["executions"][0]
        assert len(data["executions"][0]["tests"]) == 1

    @pytest.mark.anyio
    async def test_export_suite_to_yaml_not_found(self) -> None:
        """Test exporting non-existent suite to YAML."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        service = ExportService(mock_session)
        result = await service.export_suite_to_yaml(999)

        assert result is None

    @pytest.mark.anyio
    async def test_export_suite_to_yaml_no_tests(self) -> None:
        """Test exporting suite with no tests to YAML."""
        suite_def = SuiteDefinition(
            id=1,
            name="empty-suite",
            version="1.0",
            defaults_json={},
            agents_json=[],
            tests_json=[],
        )

        mock_session = AsyncMock()
        mock_session.get.return_value = suite_def

        service = ExportService(mock_session)
        result = await service.export_suite_to_yaml(1)

        assert result is None

    @pytest.mark.anyio
    async def test_get_suite_definition(self) -> None:
        """Test getting suite definition."""
        now = datetime.now()
        suite_def = SuiteDefinition(
            id=1,
            name="test-suite",
            version="1.0",
            description="Test suite description",
            defaults_json={
                "runs_per_test": 1,
                "timeout_seconds": 300,
                "scoring": {},
            },
            agents_json=[{"name": "test-agent", "type": "http", "config": {}}],
            tests_json=[
                {
                    "id": "test-001",
                    "name": "Test One",
                    "task": {"description": "Do something"},
                    "constraints": {},
                    "assertions": [],
                }
            ],
            created_at=now,
            updated_at=now,
        )

        mock_session = AsyncMock()
        mock_session.get.return_value = suite_def

        service = ExportService(mock_session)
        result = await service.get_suite_definition(1)

        assert result is not None
        assert result.name == "test-suite"
        assert len(result.agents) == 1
        assert len(result.tests) == 1

    @pytest.mark.anyio
    async def test_get_suite_definition_not_found(self) -> None:
        """Test getting non-existent suite definition."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        service = ExportService(mock_session)
        result = await service.get_suite_definition(999)

        assert result is None

    @pytest.mark.anyio
    async def test_get_suite_definition_by_name(self) -> None:
        """Test getting suite definition by name."""
        now = datetime.now()
        suite_def = SuiteDefinition(
            id=1,
            name="named-suite",
            version="1.0",
            defaults_json={
                "runs_per_test": 1,
                "timeout_seconds": 300,
                "scoring": {},
            },
            agents_json=[],
            tests_json=[
                {
                    "id": "test-001",
                    "name": "Test",
                    "task": {"description": "Task"},
                    "constraints": {},
                    "assertions": [],
                }
            ],
            created_at=now,
            updated_at=now,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = suite_def
        mock_session.execute.return_value = mock_result

        service = ExportService(mock_session)
        result = await service.get_suite_definition_by_name("named-suite")

        assert result is not None
        assert result.name == "named-suite"


class TestDependencyInjection:
    """Tests for service dependency injection functions."""

    @pytest.mark.anyio
    async def test_get_test_service(self) -> None:
        """Test getting TestService via dependency injection."""
        from atp.dashboard.v2.dependencies import get_test_service

        mock_session = AsyncMock()
        service = await get_test_service(mock_session)

        assert isinstance(service, TestService)

    @pytest.mark.anyio
    async def test_get_agent_service(self) -> None:
        """Test getting AgentService via dependency injection."""
        from atp.dashboard.v2.dependencies import get_agent_service

        mock_session = AsyncMock()
        service = await get_agent_service(mock_session)

        assert isinstance(service, AgentService)

    @pytest.mark.anyio
    async def test_get_comparison_service(self) -> None:
        """Test getting ComparisonService via dependency injection."""
        from atp.dashboard.v2.dependencies import get_comparison_service

        mock_session = AsyncMock()
        service = await get_comparison_service(mock_session)

        assert isinstance(service, ComparisonService)

    @pytest.mark.anyio
    async def test_get_export_service(self) -> None:
        """Test getting ExportService via dependency injection."""
        from atp.dashboard.v2.dependencies import get_export_service

        mock_session = AsyncMock()
        service = await get_export_service(mock_session)

        assert isinstance(service, ExportService)
