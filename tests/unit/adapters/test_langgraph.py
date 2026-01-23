"""Unit tests for LangGraphAdapter."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterTimeoutError,
    LangGraphAdapter,
    LangGraphAdapterConfig,
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
def langgraph_config() -> LangGraphAdapterConfig:
    """Create LangGraph adapter config."""
    return LangGraphAdapterConfig(
        module="tests.fixtures.mock_langgraph",
        graph="mock_graph",
        timeout_seconds=30.0,
    )


class TestLangGraphAdapterConfig:
    """Tests for LangGraphAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = LangGraphAdapterConfig(
            module="my_module",
            graph="my_graph",
        )
        assert config.module == "my_module"
        assert config.graph == "my_graph"
        assert config.input_key == "messages"
        assert config.output_key is None
        assert config.config == {}

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = LangGraphAdapterConfig(
            module="agents.research",
            graph="agent_graph",
            config={"recursion_limit": 50},
            input_key="query",
            output_key="result",
            timeout_seconds=120.0,
        )
        assert config.module == "agents.research"
        assert config.graph == "agent_graph"
        assert config.config == {"recursion_limit": 50}
        assert config.input_key == "query"
        assert config.output_key == "result"
        assert config.timeout_seconds == 120.0


class TestLangGraphAdapter:
    """Tests for LangGraphAdapter."""

    def test_adapter_type(self, langgraph_config: LangGraphAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = LangGraphAdapter(langgraph_config)
        assert adapter.adapter_type == "langgraph"

    def test_build_input_state_messages(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test building input state with messages key."""
        adapter = LangGraphAdapter(langgraph_config)
        state = adapter._build_input_state(sample_request)

        assert "messages" in state
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][0]["content"] == "Research AI trends"

    def test_build_input_state_custom_key(self, sample_request: ATPRequest) -> None:
        """Test building input state with custom input key."""
        config = LangGraphAdapterConfig(
            module="test_module",
            graph="test_graph",
            input_key="query",
        )
        adapter = LangGraphAdapter(config)
        state = adapter._build_input_state(sample_request)

        assert "query" in state
        assert state["query"] == "Research AI trends"

    def test_build_input_state_with_input_data(
        self, langgraph_config: LangGraphAdapterConfig
    ) -> None:
        """Test building input state with additional input data."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(
                description="Research AI",
                input_data={"topic": "transformers", "depth": "deep"},
            ),
        )
        adapter = LangGraphAdapter(langgraph_config)
        state = adapter._build_input_state(request)

        assert "messages" in state
        assert "topic" in state
        assert state["topic"] == "transformers"
        assert state["depth"] == "deep"

    def test_extract_output_from_messages(
        self, langgraph_config: LangGraphAdapterConfig
    ) -> None:
        """Test extracting output from messages in state."""
        adapter = LangGraphAdapter(langgraph_config)

        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        output = adapter._extract_output(state)
        assert output == "Hi there!"

    def test_extract_output_with_output_key(self) -> None:
        """Test extracting output with configured output key."""
        config = LangGraphAdapterConfig(
            module="test_module",
            graph="test_graph",
            output_key="result",
        )
        adapter = LangGraphAdapter(config)

        state = {"result": "The final answer", "other": "data"}
        output = adapter._extract_output(state)
        assert output == "The final answer"

    def test_create_event(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test creating ATP event."""
        adapter = LangGraphAdapter(langgraph_config)

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
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test successful execute call."""
        adapter = LangGraphAdapter(langgraph_config)

        # Mock the graph
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    {"role": "user", "content": "Research AI trends"},
                    {"role": "assistant", "content": "AI is evolving rapidly..."},
                ]
            }
        )

        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert isinstance(response, ATPResponse)
        assert response.task_id == "test-task-123"
        assert response.status == ResponseStatus.COMPLETED
        assert response.metrics is not None
        assert response.metrics.total_steps == 2

    @pytest.mark.anyio
    async def test_execute_sync_graph(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute with synchronous graph.invoke."""
        adapter = LangGraphAdapter(langgraph_config)

        # Mock graph without ainvoke
        mock_graph = MagicMock()
        mock_graph.ainvoke = None
        del mock_graph.ainvoke
        mock_graph.invoke = MagicMock(
            return_value={"messages": [{"role": "assistant", "content": "Result"}]}
        )

        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_import_error(self, sample_request: ATPRequest) -> None:
        """Test execute with module import error."""
        config = LangGraphAdapterConfig(
            module="nonexistent.module",
            graph="graph",
        )
        adapter = LangGraphAdapter(config)

        with pytest.raises(AdapterConnectionError) as exc_info:
            async with adapter:
                await adapter.execute(sample_request)

        assert "Failed to import module" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_graph_not_found(self, sample_request: ATPRequest) -> None:
        """Test execute with graph not found in module."""
        config = LangGraphAdapterConfig(
            module="atp.adapters.base",  # Existing module
            graph="nonexistent_graph",
        )
        adapter = LangGraphAdapter(config)

        with pytest.raises(AdapterConnectionError) as exc_info:
            async with adapter:
                await adapter.execute(sample_request)

        assert "not found in module" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_timeout(
        self, langgraph_config: LangGraphAdapterConfig
    ) -> None:
        """Test execute with timeout."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(description="Test"),
            constraints={"timeout_seconds": 0.001},
        )

        adapter = LangGraphAdapter(langgraph_config)

        # Mock a slow graph
        async def slow_invoke(*args: Any, **kwargs: Any) -> dict:
            import asyncio

            await asyncio.sleep(10)
            return {}

        mock_graph = MagicMock()
        mock_graph.ainvoke = slow_invoke

        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            async with adapter:
                with pytest.raises(AdapterTimeoutError) as exc_info:
                    await adapter.execute(request)

                assert exc_info.value.timeout_seconds == 0.001

    @pytest.mark.anyio
    async def test_execute_failure(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute with graph error returns failed response."""
        adapter = LangGraphAdapter(langgraph_config)

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crashed"))

        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.FAILED
        assert "Graph crashed" in str(response.error)

    @pytest.mark.anyio
    async def test_stream_events_success(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test streaming events successfully."""
        adapter = LangGraphAdapter(langgraph_config)

        # Mock async streaming
        async def mock_stream(*args: Any, **kwargs: Any):
            yield {"agent": {"messages": [{"role": "assistant", "content": "Step 1"}]}}
            yield {"agent": {"messages": [{"role": "assistant", "content": "Step 2"}]}}

        mock_graph = MagicMock()
        mock_graph.astream = mock_stream

        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            events: list[ATPEvent | ATPResponse] = []
            async with adapter:
                async for item in adapter.stream_events(sample_request):
                    events.append(item)

        # Should have events plus final response
        assert len(events) >= 1
        assert isinstance(events[-1], ATPResponse)
        assert events[-1].status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_stream_events_sync_fallback(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test streaming with sync stream fallback."""
        adapter = LangGraphAdapter(langgraph_config)

        # Mock sync streaming
        mock_graph = MagicMock()
        mock_graph.astream = None
        del mock_graph.astream
        mock_graph.stream = MagicMock(
            return_value=[
                {"node": {"data": "step1"}},
                {"node": {"data": "step2"}},
            ]
        )

        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            events: list[ATPEvent | ATPResponse] = []
            async with adapter:
                async for item in adapter.stream_events(sample_request):
                    events.append(item)

        assert len(events) >= 1
        assert isinstance(events[-1], ATPResponse)

    @pytest.mark.anyio
    async def test_health_check_success(
        self, langgraph_config: LangGraphAdapterConfig
    ) -> None:
        """Test health check success."""
        adapter = LangGraphAdapter(langgraph_config)

        mock_graph = MagicMock()
        with patch.object(adapter, "_load_graph", return_value=mock_graph):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(self) -> None:
        """Test health check failure."""
        config = LangGraphAdapterConfig(
            module="nonexistent.module",
            graph="graph",
        )
        adapter = LangGraphAdapter(config)

        result = await adapter.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_cleanup(self, langgraph_config: LangGraphAdapterConfig) -> None:
        """Test cleanup releases resources."""
        adapter = LangGraphAdapter(langgraph_config)

        # Simulate loaded graph
        adapter._graph = MagicMock()
        adapter._module = MagicMock()

        await adapter.cleanup()

        assert adapter._graph is None
        assert adapter._module is None

    @pytest.mark.anyio
    async def test_context_manager(
        self, langgraph_config: LangGraphAdapterConfig
    ) -> None:
        """Test adapter as async context manager."""
        adapter = LangGraphAdapter(langgraph_config)
        adapter._graph = MagicMock()
        adapter._module = MagicMock()

        async with adapter as ctx_adapter:
            assert ctx_adapter is adapter
            assert adapter._graph is not None

        # Resources should be cleaned up
        assert adapter._graph is None

    def test_convert_langgraph_event_dict(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test converting LangGraph dict event to ATP event."""
        adapter = LangGraphAdapter(langgraph_config)

        event = {"agent": {"messages": [{"role": "assistant", "content": "Hello"}]}}

        atp_event = adapter._convert_langgraph_event(sample_request, event, 0)

        assert atp_event is not None
        assert atp_event.event_type == EventType.LLM_REQUEST
        assert atp_event.payload["node"] == "agent"

    def test_message_to_event_assistant(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test converting assistant message to LLM_REQUEST event."""
        adapter = LangGraphAdapter(langgraph_config)

        message = {"role": "assistant", "content": "AI response"}
        event = adapter._message_to_event(sample_request, message, "agent", 0)

        assert event.event_type == EventType.LLM_REQUEST
        assert event.payload["content"] == "AI response"

    def test_message_to_event_tool(
        self, langgraph_config: LangGraphAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test converting tool message to TOOL_CALL event."""
        adapter = LangGraphAdapter(langgraph_config)

        message = {"role": "tool", "name": "search", "content": "Results"}
        event = adapter._message_to_event(sample_request, message, "tools", 0)

        assert event.event_type == EventType.TOOL_CALL
        assert event.payload["tool"] == "search"
