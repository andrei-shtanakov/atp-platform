"""Integration tests for framework adapters (LangGraph, CrewAI, AutoGen).

These tests verify that framework adapters work correctly with real
(mock) agents implemented using the respective frameworks.
"""

import pytest

from atp.adapters import (
    AutoGenAdapter,
    AutoGenAdapterConfig,
    CrewAIAdapter,
    CrewAIAdapterConfig,
    LangGraphAdapter,
    LangGraphAdapterConfig,
)
from atp.protocol import (
    ATPRequest,
    Task,
)


@pytest.fixture
def sample_request() -> ATPRequest:
    """Create a sample ATP request for testing."""
    return ATPRequest(
        task_id="integration-test-123",
        task=Task(
            description="Test task for framework adapter integration",
            input_data={"topic": "AI testing"},
        ),
        constraints={"max_steps": 10, "timeout_seconds": 30},
    )


class TestLangGraphAdapterIntegration:
    """Integration tests for LangGraphAdapter.

    These tests verify the adapter works correctly with mock LangGraph graphs.
    Real LangGraph integration requires the langgraph package.
    """

    @pytest.mark.anyio
    async def test_adapter_initialization(self) -> None:
        """Test adapter can be initialized with valid config."""
        config = LangGraphAdapterConfig(
            module="atp.protocol.models",  # Use existing module
            graph="ATPRequest",  # Use existing class (will fail on invoke)
        )
        adapter = LangGraphAdapter(config)

        assert adapter.adapter_type == "langgraph"
        assert adapter._config.module == "atp.protocol.models"
        assert adapter._config.graph == "ATPRequest"

    @pytest.mark.anyio
    async def test_health_check_with_loadable_module(self) -> None:
        """Test health check returns False when graph doesn't exist."""
        config = LangGraphAdapterConfig(
            module="atp.protocol.models",
            graph="nonexistent_graph",
        )
        adapter = LangGraphAdapter(config)

        result = await adapter.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_input_state_building(self, sample_request: ATPRequest) -> None:
        """Test input state is correctly built from request."""
        config = LangGraphAdapterConfig(
            module="atp.protocol.models",
            graph="ATPRequest",
            input_key="messages",
        )
        adapter = LangGraphAdapter(config)

        state = adapter._build_input_state(sample_request)

        assert "messages" in state
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"
        assert "Test task" in state["messages"][0]["content"]
        assert state["topic"] == "AI testing"

    @pytest.mark.anyio
    async def test_custom_input_key(self, sample_request: ATPRequest) -> None:
        """Test custom input key is used correctly."""
        config = LangGraphAdapterConfig(
            module="atp.protocol.models",
            graph="ATPRequest",
            input_key="query",
        )
        adapter = LangGraphAdapter(config)

        state = adapter._build_input_state(sample_request)

        assert "query" in state
        assert "Test task" in state["query"]

    @pytest.mark.anyio
    async def test_output_extraction(self) -> None:
        """Test output is correctly extracted from state."""
        config = LangGraphAdapterConfig(
            module="atp.protocol.models",
            graph="ATPRequest",
        )
        adapter = LangGraphAdapter(config)

        # Test with messages format
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Final response"},
            ]
        }
        output = adapter._extract_output(state)
        assert output == "Final response"

        # Test with custom output key
        adapter._config.output_key = "result"
        state = {"result": "Custom output", "other": "data"}
        output = adapter._extract_output(state)
        assert output == "Custom output"


class TestCrewAIAdapterIntegration:
    """Integration tests for CrewAIAdapter.

    These tests verify the adapter works correctly with mock CrewAI crews.
    Real CrewAI integration requires the crewai package.
    """

    @pytest.mark.anyio
    async def test_adapter_initialization(self) -> None:
        """Test adapter can be initialized with valid config."""
        config = CrewAIAdapterConfig(
            module="atp.protocol.models",  # Use existing module
            crew="ATPResponse",  # Use existing class (will fail on kickoff)
        )
        adapter = CrewAIAdapter(config)

        assert adapter.adapter_type == "crewai"
        assert adapter._config.module == "atp.protocol.models"
        assert adapter._config.crew == "ATPResponse"

    @pytest.mark.anyio
    async def test_health_check_with_loadable_module(self) -> None:
        """Test health check returns False when crew doesn't exist."""
        config = CrewAIAdapterConfig(
            module="atp.protocol.models",
            crew="nonexistent_crew",
        )
        adapter = CrewAIAdapter(config)

        result = await adapter.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_inputs_building(self, sample_request: ATPRequest) -> None:
        """Test inputs are correctly built from request."""
        config = CrewAIAdapterConfig(
            module="atp.protocol.models",
            crew="ATPResponse",
        )
        adapter = CrewAIAdapter(config)

        inputs = adapter._build_inputs(sample_request)

        assert "task" in inputs
        assert "description" in inputs
        assert "topic" in inputs
        assert inputs["topic"] == "AI testing"

    @pytest.mark.anyio
    async def test_factory_config(self) -> None:
        """Test factory configuration is correctly handled."""
        config = CrewAIAdapterConfig(
            module="atp.protocol.models",
            crew="create_crew",
            is_factory=True,
            factory_args={"model": "gpt-4", "verbose": True},
        )
        adapter = CrewAIAdapter(config)

        assert adapter._config.is_factory is True
        assert adapter._config.factory_args == {"model": "gpt-4", "verbose": True}


class TestAutoGenAdapterIntegration:
    """Integration tests for AutoGenAdapter.

    These tests verify the adapter works correctly with mock AutoGen agents.
    Real AutoGen integration requires the autogen or autogen-agentchat package.
    """

    @pytest.mark.anyio
    async def test_adapter_initialization(self) -> None:
        """Test adapter can be initialized with valid config."""
        config = AutoGenAdapterConfig(
            module="atp.protocol.models",  # Use existing module
            agent="ATPEvent",  # Use existing class (will fail on initiate_chat)
        )
        adapter = AutoGenAdapter(config)

        assert adapter.adapter_type == "autogen"
        assert adapter._config.module == "atp.protocol.models"
        assert adapter._config.agent == "ATPEvent"

    @pytest.mark.anyio
    async def test_health_check_with_loadable_module(self) -> None:
        """Test health check returns False when agent doesn't exist."""
        config = AutoGenAdapterConfig(
            module="atp.protocol.models",
            agent="nonexistent_agent",
        )
        adapter = AutoGenAdapter(config)

        result = await adapter.health_check()
        assert result is False

    @pytest.mark.anyio
    async def test_message_building(self, sample_request: ATPRequest) -> None:
        """Test message is correctly built from request."""
        config = AutoGenAdapterConfig(
            module="atp.protocol.models",
            agent="ATPEvent",
        )
        adapter = AutoGenAdapter(config)

        message = adapter._build_message(sample_request)

        assert "Test task" in message
        assert "topic: AI testing" in message

    @pytest.mark.anyio
    async def test_user_proxy_config(self) -> None:
        """Test user proxy configuration is correctly handled."""
        config = AutoGenAdapterConfig(
            module="atp.protocol.models",
            agent="ATPEvent",
            user_proxy="user_proxy_agent",
            max_consecutive_auto_reply=5,
            human_input_mode="TERMINATE",
        )
        adapter = AutoGenAdapter(config)

        assert adapter._config.user_proxy == "user_proxy_agent"
        assert adapter._config.max_consecutive_auto_reply == 5
        assert adapter._config.human_input_mode == "TERMINATE"

    @pytest.mark.anyio
    async def test_chat_history_extraction(self) -> None:
        """Test chat history is correctly extracted."""
        config = AutoGenAdapterConfig(
            module="atp.protocol.models",
            agent="ATPEvent",
        )
        adapter = AutoGenAdapter(config)

        # Mock agent with chat_messages
        class MockAgent:
            chat_messages = {
                "assistant": [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Answer!"},
                ]
            }

        history = adapter._extract_chat_history(MockAgent())

        assert len(history) == 2
        assert history[0]["content"] == "Question?"
        assert history[1]["content"] == "Answer!"


class TestAdapterCommonBehavior:
    """Test common behavior across all framework adapters."""

    @pytest.mark.anyio
    async def test_all_adapters_have_adapter_type(self) -> None:
        """Test all adapters return correct adapter_type."""
        langgraph = LangGraphAdapter(LangGraphAdapterConfig(module="m", graph="g"))
        crewai = CrewAIAdapter(CrewAIAdapterConfig(module="m", crew="c"))
        autogen = AutoGenAdapter(AutoGenAdapterConfig(module="m", agent="a"))

        assert langgraph.adapter_type == "langgraph"
        assert crewai.adapter_type == "crewai"
        assert autogen.adapter_type == "autogen"

    @pytest.mark.anyio
    async def test_all_adapters_support_context_manager(self) -> None:
        """Test all adapters can be used as async context managers."""
        configs = [
            LangGraphAdapterConfig(module="m", graph="g"),
            CrewAIAdapterConfig(module="m", crew="c"),
            AutoGenAdapterConfig(module="m", agent="a"),
        ]
        adapters = [
            LangGraphAdapter(configs[0]),
            CrewAIAdapter(configs[1]),
            AutoGenAdapter(configs[2]),
        ]

        for adapter in adapters:
            async with adapter as ctx:
                assert ctx is adapter

    @pytest.mark.anyio
    async def test_all_adapters_cleanup_resources(self) -> None:
        """Test all adapters cleanup resources on cleanup()."""
        langgraph = LangGraphAdapter(LangGraphAdapterConfig(module="m", graph="g"))
        crewai = CrewAIAdapter(CrewAIAdapterConfig(module="m", crew="c"))
        autogen = AutoGenAdapter(AutoGenAdapterConfig(module="m", agent="a"))

        # Set some state
        langgraph._graph = object()
        crewai._crew = object()
        autogen._agent = object()

        # Cleanup
        await langgraph.cleanup()
        await crewai.cleanup()
        await autogen.cleanup()

        # Verify cleanup
        assert langgraph._graph is None
        assert crewai._crew is None
        assert autogen._agent is None

    @pytest.mark.anyio
    async def test_health_check_fails_for_invalid_module(self) -> None:
        """Test health check returns False for invalid module."""
        adapters = [
            LangGraphAdapter(
                LangGraphAdapterConfig(module="invalid.module", graph="g")
            ),
            CrewAIAdapter(CrewAIAdapterConfig(module="invalid.module", crew="c")),
            AutoGenAdapter(AutoGenAdapterConfig(module="invalid.module", agent="a")),
        ]

        for adapter in adapters:
            result = await adapter.health_check()
            assert result is False, f"{adapter.adapter_type} should return False"
