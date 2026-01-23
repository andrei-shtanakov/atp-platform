"""Unit tests for CrewAIAdapter."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterTimeoutError,
    CrewAIAdapter,
    CrewAIAdapterConfig,
)
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    ResponseStatus,
    Task,
)


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request for testing."""
    return ATPRequest(
        task_id="test-task-123",
        task=Task(description="Research AI trends"),
        constraints={"max_steps": 10, "timeout_seconds": 60},
    )


@pytest.fixture
def crewai_config() -> CrewAIAdapterConfig:
    """Create CrewAI adapter config."""
    return CrewAIAdapterConfig(
        module="tests.fixtures.mock_crewai",
        crew="mock_crew",
        timeout_seconds=30.0,
    )


class TestCrewAIAdapterConfig:
    """Tests for CrewAIAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = CrewAIAdapterConfig(
            module="my_module",
            crew="my_crew",
        )
        assert config.module == "my_module"
        assert config.crew == "my_crew"
        assert config.is_factory is False
        assert config.factory_args == {}
        assert config.verbose is False
        assert config.memory is False

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = CrewAIAdapterConfig(
            module="agents.research",
            crew="create_crew",
            is_factory=True,
            factory_args={"model": "gpt-4"},
            verbose=True,
            memory=True,
            timeout_seconds=120.0,
        )
        assert config.module == "agents.research"
        assert config.crew == "create_crew"
        assert config.is_factory is True
        assert config.factory_args == {"model": "gpt-4"}
        assert config.verbose is True
        assert config.memory is True
        assert config.timeout_seconds == 120.0


class TestCrewAIAdapter:
    """Tests for CrewAIAdapter."""

    def test_adapter_type(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = CrewAIAdapter(crewai_config)
        assert adapter.adapter_type == "crewai"

    def test_build_inputs(
        self, crewai_config: CrewAIAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test building inputs from request."""
        adapter = CrewAIAdapter(crewai_config)
        inputs = adapter._build_inputs(sample_request)

        assert "task" in inputs
        assert inputs["task"] == "Research AI trends"
        assert "description" in inputs
        assert inputs["description"] == "Research AI trends"

    def test_build_inputs_with_input_data(
        self, crewai_config: CrewAIAdapterConfig
    ) -> None:
        """Test building inputs with additional input data."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(
                description="Research AI",
                input_data={"topic": "transformers", "depth": "deep"},
            ),
        )
        adapter = CrewAIAdapter(crewai_config)
        inputs = adapter._build_inputs(request)

        assert inputs["task"] == "Research AI"
        assert inputs["topic"] == "transformers"
        assert inputs["depth"] == "deep"

    def test_extract_output_with_raw(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test extracting output from result with raw attribute."""
        adapter = CrewAIAdapter(crewai_config)

        # Mock CrewOutput-like object
        mock_result = MagicMock()
        mock_result.raw = "The final research report..."
        mock_result.tasks_output = []

        output, data = adapter._extract_output(mock_result)

        assert output == "The final research report..."
        assert data["raw"] == "The final research report..."

    def test_extract_output_with_tasks_output(
        self, crewai_config: CrewAIAdapterConfig
    ) -> None:
        """Test extracting output with tasks_output."""
        adapter = CrewAIAdapter(crewai_config)

        # Mock CrewOutput with tasks
        mock_task_output = MagicMock()
        mock_task_output.description = "Research task"
        mock_task_output.raw = "Task result"
        mock_task_output.agent = "researcher"

        mock_result = MagicMock()
        mock_result.raw = "Final output"
        mock_result.tasks_output = [mock_task_output]

        output, data = adapter._extract_output(mock_result)

        assert output == "Final output"
        assert len(data["tasks_output"]) == 1
        assert data["tasks_output"][0]["description"] == "Research task"

    def test_extract_output_dict(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test extracting output from dict result."""
        adapter = CrewAIAdapter(crewai_config)

        result = {"raw": "Output text", "other": "data"}
        output, data = adapter._extract_output(result)

        assert output == "Output text"
        assert data["raw"] == "Output text"

    def test_extract_output_string(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test extracting output from string result."""
        adapter = CrewAIAdapter(crewai_config)

        result = "Plain string output"
        output, data = adapter._extract_output(result)

        assert output == "Plain string output"
        assert data["raw"] == "Plain string output"

    def test_create_event(
        self, crewai_config: CrewAIAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test creating ATP event."""
        adapter = CrewAIAdapter(crewai_config)

        event = adapter._create_event(
            sample_request,
            EventType.PROGRESS,
            {"message": "Processing..."},
            sequence=0,
        )

        assert isinstance(event, ATPEvent)
        assert event.task_id == "test-task-123"
        assert event.event_type == EventType.PROGRESS
        assert event.sequence == 0
        assert event.payload["message"] == "Processing..."

    @pytest.mark.anyio
    async def test_execute_success(
        self, crewai_config: CrewAIAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test successful execute call."""
        adapter = CrewAIAdapter(crewai_config)

        # Mock the crew
        mock_result = MagicMock()
        mock_result.raw = "Research completed successfully"
        mock_result.tasks_output = []

        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value=mock_result)
        mock_crew.tasks = []
        mock_crew.agents = []

        with patch.object(adapter, "_load_crew", return_value=mock_crew):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert isinstance(response, ATPResponse)
        assert response.task_id == "test-task-123"
        assert response.status == ResponseStatus.COMPLETED
        assert response.metrics is not None

    @pytest.mark.anyio
    async def test_execute_sync_kickoff(
        self, crewai_config: CrewAIAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute with synchronous kickoff."""
        adapter = CrewAIAdapter(crewai_config)

        mock_result = MagicMock()
        mock_result.raw = "Result"
        mock_result.tasks_output = []

        mock_crew = MagicMock()
        mock_crew.kickoff_async = None
        del mock_crew.kickoff_async
        mock_crew.kickoff = MagicMock(return_value=mock_result)
        mock_crew.tasks = []
        mock_crew.agents = []

        with patch.object(adapter, "_load_crew", return_value=mock_crew):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_import_error(self, sample_request: ATPRequest) -> None:
        """Test execute with module import error."""
        config = CrewAIAdapterConfig(
            module="nonexistent.module",
            crew="crew",
        )
        adapter = CrewAIAdapter(config)

        with pytest.raises(AdapterConnectionError) as exc_info:
            async with adapter:
                await adapter.execute(sample_request)

        assert "Failed to import module" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_crew_not_found(self, sample_request: ATPRequest) -> None:
        """Test execute with crew not found in module."""
        config = CrewAIAdapterConfig(
            module="atp.adapters.base",  # Existing module
            crew="nonexistent_crew",
        )
        adapter = CrewAIAdapter(config)

        with pytest.raises(AdapterConnectionError) as exc_info:
            async with adapter:
                await adapter.execute(sample_request)

        assert "not found in module" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_factory(self, sample_request: ATPRequest) -> None:
        """Test execute with crew factory."""
        config = CrewAIAdapterConfig(
            module="tests.fixtures.mock_crewai",
            crew="create_crew",
            is_factory=True,
            factory_args={"model": "gpt-4"},
        )
        adapter = CrewAIAdapter(config)

        mock_result = MagicMock()
        mock_result.raw = "Factory result"
        mock_result.tasks_output = []

        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value=mock_result)
        mock_crew.tasks = []
        mock_crew.agents = []

        mock_factory = MagicMock(return_value=mock_crew)

        with patch("atp.adapters.crewai.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.create_crew = mock_factory
            mock_import.return_value = mock_module

            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED
        mock_factory.assert_called_once_with(model="gpt-4")

    @pytest.mark.anyio
    async def test_execute_timeout(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test execute with timeout."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(description="Test"),
            constraints={"timeout_seconds": 0.001},
        )

        adapter = CrewAIAdapter(crewai_config)

        async def slow_kickoff(*args: Any, **kwargs: Any) -> MagicMock:
            import asyncio

            await asyncio.sleep(10)
            return MagicMock()

        mock_crew = MagicMock()
        mock_crew.kickoff_async = slow_kickoff
        mock_crew.tasks = []
        mock_crew.agents = []

        with patch.object(adapter, "_load_crew", return_value=mock_crew):
            async with adapter:
                with pytest.raises(AdapterTimeoutError) as exc_info:
                    await adapter.execute(request)

                assert exc_info.value.timeout_seconds == 0.001

    @pytest.mark.anyio
    async def test_execute_failure(
        self, crewai_config: CrewAIAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute with crew error returns failed response."""
        adapter = CrewAIAdapter(crewai_config)

        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(
            side_effect=RuntimeError("Crew execution failed")
        )
        mock_crew.tasks = []
        mock_crew.agents = []

        with patch.object(adapter, "_load_crew", return_value=mock_crew):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.FAILED
        assert "Crew execution failed" in str(response.error)

    def test_extract_metrics(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test extracting metrics from crew execution."""
        adapter = CrewAIAdapter(crewai_config)

        mock_task_output1 = MagicMock()
        mock_task_output2 = MagicMock()

        mock_result = MagicMock()
        mock_result.tasks_output = [mock_task_output1, mock_task_output2]
        mock_result.token_usage = MagicMock()
        mock_result.token_usage.total_tokens = 1500

        mock_agent = MagicMock()
        mock_agent.tools = [MagicMock(), MagicMock()]

        mock_crew = MagicMock()
        mock_crew.tasks = [MagicMock(), MagicMock()]
        mock_crew.agents = [mock_agent]

        metrics = adapter._extract_metrics(mock_crew, mock_result, 5.0)

        assert metrics.total_tokens == 1500
        assert metrics.total_steps == 2
        assert metrics.llm_calls == 2
        assert metrics.tool_calls == 2
        assert metrics.wall_time_seconds == 5.0

    @pytest.mark.anyio
    async def test_stream_events_success(
        self, crewai_config: CrewAIAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test streaming events successfully."""
        adapter = CrewAIAdapter(crewai_config)

        mock_task = MagicMock()
        mock_task.description = "Research task"

        mock_agent = MagicMock()
        mock_agent.role = "Researcher"

        mock_task_output = MagicMock()
        mock_task_output.description = "Research task"
        mock_task_output.agent = "Researcher"

        mock_result = MagicMock()
        mock_result.raw = "Final output"
        mock_result.tasks_output = [mock_task_output]

        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value=mock_result)
        mock_crew.tasks = [mock_task]
        mock_crew.agents = [mock_agent]

        with patch.object(adapter, "_load_crew", return_value=mock_crew):
            events: list[ATPEvent | ATPResponse] = []
            async with adapter:
                async for item in adapter.stream_events(sample_request):
                    events.append(item)

        # Should have progress events plus final response
        assert len(events) >= 3  # Start + task + agent + result + response
        assert isinstance(events[-1], ATPResponse)
        assert events[-1].status == ResponseStatus.COMPLETED

        # Check for expected event types
        event_types = [e.event_type for e in events if isinstance(e, ATPEvent)]
        assert EventType.PROGRESS in event_types

    @pytest.mark.anyio
    async def test_health_check_success(
        self, crewai_config: CrewAIAdapterConfig
    ) -> None:
        """Test health check success."""
        adapter = CrewAIAdapter(crewai_config)

        mock_crew = MagicMock()
        with patch.object(adapter, "_load_crew", return_value=mock_crew):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(self) -> None:
        """Test health check failure."""
        config = CrewAIAdapterConfig(
            module="nonexistent.module",
            crew="crew",
        )
        adapter = CrewAIAdapter(config)

        result = await adapter.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_cleanup(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test cleanup releases resources."""
        adapter = CrewAIAdapter(crewai_config)

        # Simulate loaded crew
        adapter._crew = MagicMock()
        adapter._module = MagicMock()

        await adapter.cleanup()

        assert adapter._crew is None
        assert adapter._module is None

    @pytest.mark.anyio
    async def test_context_manager(self, crewai_config: CrewAIAdapterConfig) -> None:
        """Test adapter as async context manager."""
        adapter = CrewAIAdapter(crewai_config)
        adapter._crew = MagicMock()
        adapter._module = MagicMock()

        async with adapter as ctx_adapter:
            assert ctx_adapter is adapter
            assert adapter._crew is not None

        # Resources should be cleaned up
        assert adapter._crew is None
