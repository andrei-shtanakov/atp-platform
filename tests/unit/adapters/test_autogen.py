"""Unit tests for AutoGenAdapter."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.adapters import (
    AdapterConnectionError,
    AdapterTimeoutError,
    AutoGenAdapter,
    AutoGenAdapterConfig,
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
def autogen_config() -> AutoGenAdapterConfig:
    """Create AutoGen adapter config."""
    return AutoGenAdapterConfig(
        module="tests.fixtures.mock_autogen",
        agent="mock_agent",
        timeout_seconds=30.0,
    )


class TestAutoGenAdapterConfig:
    """Tests for AutoGenAdapterConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = AutoGenAdapterConfig(
            module="my_module",
            agent="my_agent",
        )
        assert config.module == "my_module"
        assert config.agent == "my_agent"
        assert config.is_factory is False
        assert config.factory_args == {}
        assert config.user_proxy is None
        assert config.max_consecutive_auto_reply == 10
        assert config.human_input_mode == "NEVER"

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = AutoGenAdapterConfig(
            module="agents.chat",
            agent="create_agent",
            is_factory=True,
            factory_args={"model": "gpt-4"},
            user_proxy="user_proxy_agent",
            max_consecutive_auto_reply=5,
            human_input_mode="TERMINATE",
            timeout_seconds=120.0,
        )
        assert config.module == "agents.chat"
        assert config.agent == "create_agent"
        assert config.is_factory is True
        assert config.factory_args == {"model": "gpt-4"}
        assert config.user_proxy == "user_proxy_agent"
        assert config.max_consecutive_auto_reply == 5
        assert config.human_input_mode == "TERMINATE"
        assert config.timeout_seconds == 120.0


class TestAutoGenAdapter:
    """Tests for AutoGenAdapter."""

    def test_adapter_type(self, autogen_config: AutoGenAdapterConfig) -> None:
        """Test adapter type property."""
        adapter = AutoGenAdapter(autogen_config)
        assert adapter.adapter_type == "autogen"

    def test_build_message(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test building message from request."""
        adapter = AutoGenAdapter(autogen_config)
        message = adapter._build_message(sample_request)

        assert message == "Research AI trends"

    def test_build_message_with_input_data(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test building message with additional input data."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(
                description="Research AI",
                input_data={"topic": "transformers", "depth": "deep"},
            ),
        )
        adapter = AutoGenAdapter(autogen_config)
        message = adapter._build_message(request)

        assert "Research AI" in message
        assert "topic: transformers" in message
        assert "depth: deep" in message

    def test_create_event(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test creating ATP event."""
        adapter = AutoGenAdapter(autogen_config)

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

    def test_extract_chat_history(self, autogen_config: AutoGenAdapterConfig) -> None:
        """Test extracting chat history from agent."""
        adapter = AutoGenAdapter(autogen_config)

        mock_agent = MagicMock()
        mock_agent.chat_messages = {
            "assistant": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        history = adapter._extract_chat_history(mock_agent)

        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi there!"

    def test_extract_chat_history_oai_messages(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test extracting chat history from _oai_messages."""
        adapter = AutoGenAdapter(autogen_config)

        mock_agent = MagicMock()
        mock_agent.chat_messages = None
        del mock_agent.chat_messages
        mock_agent._oai_messages = {
            "assistant": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }

        history = adapter._extract_chat_history(mock_agent)

        assert len(history) == 2

    @pytest.mark.anyio
    async def test_execute_success(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test successful execute call."""
        adapter = AutoGenAdapter(autogen_config)

        # Mock chat result
        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Task completed successfully"
        mock_chat_result.chat_history = [
            {"role": "user", "content": "Research AI trends"},
            {"role": "assistant", "content": "Here are the trends..."},
        ]
        mock_chat_result.cost = {}

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.a_initiate_chat = AsyncMock(return_value=mock_chat_result)
        mock_user_proxy.chat_messages = {}

        with patch.object(
            adapter, "_load_agent", return_value=(mock_agent, mock_user_proxy)
        ):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert isinstance(response, ATPResponse)
        assert response.task_id == "test-task-123"
        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_sync_initiate_chat(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute with synchronous initiate_chat."""
        adapter = AutoGenAdapter(autogen_config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Result"
        mock_chat_result.chat_history = []
        mock_chat_result.cost = {}

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.a_initiate_chat = None
        del mock_user_proxy.a_initiate_chat
        mock_user_proxy.initiate_chat = MagicMock(return_value=mock_chat_result)
        mock_user_proxy.chat_messages = {}

        with patch.object(
            adapter, "_load_agent", return_value=(mock_agent, mock_user_proxy)
        ):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_with_default_user_proxy(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute creates default user proxy when not provided."""
        adapter = AutoGenAdapter(autogen_config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Result"
        mock_chat_result.chat_history = []
        mock_chat_result.cost = {}

        mock_agent = MagicMock()

        mock_user_proxy_class = MagicMock()
        mock_user_proxy_instance = MagicMock()
        mock_user_proxy_instance.a_initiate_chat = AsyncMock(
            return_value=mock_chat_result
        )
        mock_user_proxy_instance.chat_messages = {}
        mock_user_proxy_class.return_value = mock_user_proxy_instance

        with patch.object(adapter, "_load_agent", return_value=(mock_agent, None)):
            with patch.object(
                adapter, "_create_user_proxy", return_value=mock_user_proxy_instance
            ):
                async with adapter:
                    response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_execute_import_error(self, sample_request: ATPRequest) -> None:
        """Test execute with module import error."""
        config = AutoGenAdapterConfig(
            module="nonexistent.module",
            agent="agent",
        )
        adapter = AutoGenAdapter(config)

        with pytest.raises(AdapterConnectionError) as exc_info:
            async with adapter:
                await adapter.execute(sample_request)

        assert "Failed to import module" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_agent_not_found(self, sample_request: ATPRequest) -> None:
        """Test execute with agent not found in module."""
        config = AutoGenAdapterConfig(
            module="atp.adapters.base",  # Existing module
            agent="nonexistent_agent",
        )
        adapter = AutoGenAdapter(config)

        with pytest.raises(AdapterConnectionError) as exc_info:
            async with adapter:
                await adapter.execute(sample_request)

        assert "not found in module" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_factory(self, sample_request: ATPRequest) -> None:
        """Test execute with agent factory."""
        config = AutoGenAdapterConfig(
            module="tests.fixtures.mock_autogen",
            agent="create_agent",
            is_factory=True,
            factory_args={"model": "gpt-4"},
        )
        adapter = AutoGenAdapter(config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Factory result"
        mock_chat_result.chat_history = []
        mock_chat_result.cost = {}

        mock_agent = MagicMock()
        mock_factory = MagicMock(return_value=mock_agent)

        mock_user_proxy = MagicMock()
        mock_user_proxy.a_initiate_chat = AsyncMock(return_value=mock_chat_result)
        mock_user_proxy.chat_messages = {}

        with patch("atp.adapters.autogen.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.create_agent = mock_factory
            mock_import.return_value = mock_module

            with patch.object(
                adapter, "_create_user_proxy", return_value=mock_user_proxy
            ):
                async with adapter:
                    response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.COMPLETED
        mock_factory.assert_called_once_with(model="gpt-4")

    @pytest.mark.anyio
    async def test_execute_timeout(self, autogen_config: AutoGenAdapterConfig) -> None:
        """Test execute with timeout."""
        request = ATPRequest(
            task_id="test-123",
            task=Task(description="Test"),
            constraints={"timeout_seconds": 0.001},
        )

        adapter = AutoGenAdapter(autogen_config)

        async def slow_chat(*args: Any, **kwargs: Any) -> MagicMock:
            import asyncio

            await asyncio.sleep(10)
            return MagicMock()

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.a_initiate_chat = slow_chat
        mock_user_proxy.chat_messages = {}

        with patch.object(
            adapter, "_load_agent", return_value=(mock_agent, mock_user_proxy)
        ):
            async with adapter:
                with pytest.raises(AdapterTimeoutError) as exc_info:
                    await adapter.execute(request)

                assert exc_info.value.timeout_seconds == 0.001

    @pytest.mark.anyio
    async def test_execute_failure(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test execute with error returns failed response."""
        adapter = AutoGenAdapter(autogen_config)

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.a_initiate_chat = AsyncMock(
            side_effect=RuntimeError("Chat failed")
        )
        mock_user_proxy.chat_messages = {}

        with patch.object(
            adapter, "_load_agent", return_value=(mock_agent, mock_user_proxy)
        ):
            async with adapter:
                response = await adapter.execute(sample_request)

        assert response.status == ResponseStatus.FAILED
        assert "Chat failed" in str(response.error)

    def test_extract_result_with_summary(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test extracting result with summary."""
        adapter = AutoGenAdapter(autogen_config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Final summary"
        mock_chat_result.chat_history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        mock_chat_result.cost = {}

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.chat_messages = {}

        output, metrics = adapter._extract_result(
            mock_chat_result, mock_agent, mock_user_proxy, 5.0
        )

        assert output == "Final summary"
        assert metrics.total_steps == 0  # No messages in mock_user_proxy
        assert metrics.wall_time_seconds == 5.0

    def test_extract_result_from_chat_history(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test extracting result from chat history."""
        adapter = AutoGenAdapter(autogen_config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = None
        del mock_chat_result.summary
        mock_chat_result.chat_history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "The final answer"},
        ]
        mock_chat_result.cost = {}

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        # Chat messages are stored in user_proxy for metrics extraction
        mock_user_proxy.chat_messages = {
            "assistant": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "The final answer"},
            ]
        }

        output, metrics = adapter._extract_result(
            mock_chat_result, mock_agent, mock_user_proxy, 3.0
        )

        assert output == "The final answer"
        assert metrics.total_steps == 2

    def test_extract_result_with_token_cost(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test extracting result with token cost."""
        adapter = AutoGenAdapter(autogen_config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Summary"
        mock_chat_result.chat_history = []
        mock_chat_result.cost = {
            "gpt-4": {"total_tokens": 1000},
            "gpt-3.5-turbo": {"total_tokens": 500},
        }

        mock_agent = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.chat_messages = {}

        output, metrics = adapter._extract_result(
            mock_chat_result, mock_agent, mock_user_proxy, 2.0
        )

        assert metrics.total_tokens == 1500

    @pytest.mark.anyio
    async def test_stream_events_success(
        self, autogen_config: AutoGenAdapterConfig, sample_request: ATPRequest
    ) -> None:
        """Test streaming events successfully."""
        adapter = AutoGenAdapter(autogen_config)

        mock_chat_result = MagicMock()
        mock_chat_result.summary = "Final result"
        mock_chat_result.chat_history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        mock_chat_result.cost = {}

        mock_agent = MagicMock()
        mock_agent.register_reply = MagicMock()

        mock_user_proxy = MagicMock()
        mock_user_proxy.a_initiate_chat = AsyncMock(return_value=mock_chat_result)
        mock_user_proxy.chat_messages = {}

        with patch.object(
            adapter, "_load_agent", return_value=(mock_agent, mock_user_proxy)
        ):
            events: list[ATPEvent | ATPResponse] = []
            async with adapter:
                async for item in adapter.stream_events(sample_request):
                    events.append(item)

        # Should have start event plus final response
        assert len(events) >= 2
        assert isinstance(events[-1], ATPResponse)
        assert events[-1].status == ResponseStatus.COMPLETED

        # Check for progress event
        event_types = [e.event_type for e in events if isinstance(e, ATPEvent)]
        assert EventType.PROGRESS in event_types

    @pytest.mark.anyio
    async def test_health_check_success(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test health check success."""
        adapter = AutoGenAdapter(autogen_config)

        mock_agent = MagicMock()
        with patch.object(adapter, "_load_agent", return_value=(mock_agent, None)):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.anyio
    async def test_health_check_failure(self) -> None:
        """Test health check failure."""
        config = AutoGenAdapterConfig(
            module="nonexistent.module",
            agent="agent",
        )
        adapter = AutoGenAdapter(config)

        result = await adapter.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_cleanup(self, autogen_config: AutoGenAdapterConfig) -> None:
        """Test cleanup releases resources."""
        adapter = AutoGenAdapter(autogen_config)

        # Simulate loaded agent
        mock_agent = MagicMock()
        mock_agent.reset = MagicMock()

        mock_user_proxy = MagicMock()
        mock_user_proxy.reset = MagicMock()

        adapter._agent = mock_agent
        adapter._user_proxy = mock_user_proxy
        adapter._module = MagicMock()

        await adapter.cleanup()

        assert adapter._agent is None
        assert adapter._user_proxy is None
        assert adapter._module is None
        mock_agent.reset.assert_called_once()
        mock_user_proxy.reset.assert_called_once()

    @pytest.mark.anyio
    async def test_cleanup_with_clear_history(
        self, autogen_config: AutoGenAdapterConfig
    ) -> None:
        """Test cleanup with clear_history method."""
        adapter = AutoGenAdapter(autogen_config)

        mock_agent = MagicMock()
        mock_agent.reset = None
        del mock_agent.reset
        mock_agent.clear_history = MagicMock()

        adapter._agent = mock_agent
        adapter._user_proxy = None
        adapter._module = MagicMock()

        await adapter.cleanup()

        mock_agent.clear_history.assert_called_once()

    @pytest.mark.anyio
    async def test_context_manager(self, autogen_config: AutoGenAdapterConfig) -> None:
        """Test adapter as async context manager."""
        adapter = AutoGenAdapter(autogen_config)
        adapter._agent = MagicMock()
        adapter._module = MagicMock()

        async with adapter as ctx_adapter:
            assert ctx_adapter is adapter
            assert adapter._agent is not None

        # Resources should be cleaned up
        assert adapter._agent is None
