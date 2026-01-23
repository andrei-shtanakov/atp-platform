"""Unit tests for AdapterRegistry."""

import pytest

from atp.adapters import (
    AdapterConfig,
    AdapterNotFoundError,
    AdapterRegistry,
    AgentAdapter,
    AutoGenAdapter,
    AutoGenAdapterConfig,
    CLIAdapter,
    CLIAdapterConfig,
    ContainerAdapter,
    ContainerAdapterConfig,
    CrewAIAdapter,
    CrewAIAdapterConfig,
    HTTPAdapter,
    HTTPAdapterConfig,
    LangGraphAdapter,
    LangGraphAdapterConfig,
    create_adapter,
    get_registry,
)
from atp.protocol import ATPRequest, ATPResponse


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_builtin_adapters_registered(self) -> None:
        """Test that built-in adapters are registered."""
        registry = AdapterRegistry()

        assert registry.is_registered("http")
        assert registry.is_registered("container")
        assert registry.is_registered("cli")
        assert registry.is_registered("langgraph")
        assert registry.is_registered("crewai")
        assert registry.is_registered("autogen")

    def test_list_adapters(self) -> None:
        """Test listing all registered adapters."""
        registry = AdapterRegistry()
        adapters = registry.list_adapters()

        assert "http" in adapters
        assert "container" in adapters
        assert "cli" in adapters
        assert "langgraph" in adapters
        assert "crewai" in adapters
        assert "autogen" in adapters

    def test_get_adapter_class(self) -> None:
        """Test getting adapter class."""
        registry = AdapterRegistry()

        assert registry.get_adapter_class("http") is HTTPAdapter
        assert registry.get_adapter_class("container") is ContainerAdapter
        assert registry.get_adapter_class("cli") is CLIAdapter
        assert registry.get_adapter_class("langgraph") is LangGraphAdapter
        assert registry.get_adapter_class("crewai") is CrewAIAdapter
        assert registry.get_adapter_class("autogen") is AutoGenAdapter

    def test_get_adapter_class_not_found(self) -> None:
        """Test getting non-existent adapter class."""
        registry = AdapterRegistry()

        with pytest.raises(AdapterNotFoundError) as exc_info:
            registry.get_adapter_class("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_config_class(self) -> None:
        """Test getting config class."""
        registry = AdapterRegistry()

        assert registry.get_config_class("http") is HTTPAdapterConfig
        assert registry.get_config_class("container") is ContainerAdapterConfig
        assert registry.get_config_class("cli") is CLIAdapterConfig
        assert registry.get_config_class("langgraph") is LangGraphAdapterConfig
        assert registry.get_config_class("crewai") is CrewAIAdapterConfig
        assert registry.get_config_class("autogen") is AutoGenAdapterConfig

    def test_get_config_class_not_found(self) -> None:
        """Test getting non-existent config class."""
        registry = AdapterRegistry()

        with pytest.raises(AdapterNotFoundError):
            registry.get_config_class("nonexistent")

    def test_create_http_adapter(self) -> None:
        """Test creating HTTP adapter from dict config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "http",
            {"endpoint": "http://localhost:8000"},
        )

        assert isinstance(adapter, HTTPAdapter)
        assert adapter.adapter_type == "http"

    def test_create_container_adapter(self) -> None:
        """Test creating container adapter from dict config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "container",
            {"image": "my-agent:latest"},
        )

        assert isinstance(adapter, ContainerAdapter)
        assert adapter.adapter_type == "container"

    def test_create_cli_adapter(self) -> None:
        """Test creating CLI adapter from dict config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "cli",
            {"command": "python agent.py"},
        )

        assert isinstance(adapter, CLIAdapter)
        assert adapter.adapter_type == "cli"

    def test_create_with_config_object(self) -> None:
        """Test creating adapter from config object."""
        registry = AdapterRegistry()
        config = HTTPAdapterConfig(endpoint="http://localhost:8000")

        adapter = registry.create("http", config)

        assert isinstance(adapter, HTTPAdapter)

    def test_create_not_found(self) -> None:
        """Test creating non-existent adapter."""
        registry = AdapterRegistry()

        with pytest.raises(AdapterNotFoundError):
            registry.create("nonexistent", {})

    def test_create_invalid_config(self) -> None:
        """Test creating adapter with invalid config."""
        registry = AdapterRegistry()

        with pytest.raises(ValueError):
            registry.create("http", {})  # Missing required 'endpoint'

    def test_register_custom_adapter(self) -> None:
        """Test registering a custom adapter."""
        registry = AdapterRegistry()

        class CustomConfig(AdapterConfig):
            custom_field: str

        class CustomAdapter(AgentAdapter):
            def __init__(self, config: CustomConfig) -> None:
                super().__init__(config)

            @property
            def adapter_type(self) -> str:
                return "custom"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                raise NotImplementedError

            async def stream_events(self, request):
                yield  # type: ignore

        registry.register("custom", CustomAdapter, CustomConfig)

        assert registry.is_registered("custom")
        assert registry.get_adapter_class("custom") is CustomAdapter
        assert registry.get_config_class("custom") is CustomConfig

    def test_unregister_adapter(self) -> None:
        """Test unregistering an adapter."""
        registry = AdapterRegistry()

        # First verify it exists
        assert registry.is_registered("cli")

        # Unregister
        result = registry.unregister("cli")
        assert result is True
        assert not registry.is_registered("cli")

    def test_unregister_nonexistent_adapter(self) -> None:
        """Test unregistering non-existent adapter."""
        registry = AdapterRegistry()

        result = registry.unregister("nonexistent")
        assert result is False

    def test_is_registered(self) -> None:
        """Test checking if adapter is registered."""
        registry = AdapterRegistry()

        assert registry.is_registered("http") is True
        assert registry.is_registered("nonexistent") is False


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self) -> None:
        """Test that get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_create_adapter_function(self) -> None:
        """Test create_adapter convenience function."""
        adapter = create_adapter(
            "http",
            {"endpoint": "http://localhost:8000"},
        )

        assert isinstance(adapter, HTTPAdapter)

    def test_create_adapter_not_found(self) -> None:
        """Test create_adapter with non-existent type."""
        with pytest.raises(AdapterNotFoundError):
            create_adapter("nonexistent", {})


class TestFrameworkAdapters:
    """Tests for framework adapter registration."""

    def test_create_langgraph_adapter(self) -> None:
        """Test creating LangGraph adapter from dict config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "langgraph",
            {"module": "my.module", "graph": "my_graph"},
        )

        assert isinstance(adapter, LangGraphAdapter)
        assert adapter.adapter_type == "langgraph"

    def test_create_crewai_adapter(self) -> None:
        """Test creating CrewAI adapter from dict config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "crewai",
            {"module": "my.module", "crew": "my_crew"},
        )

        assert isinstance(adapter, CrewAIAdapter)
        assert adapter.adapter_type == "crewai"

    def test_create_autogen_adapter(self) -> None:
        """Test creating AutoGen adapter from dict config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "autogen",
            {"module": "my.module", "agent": "my_agent"},
        )

        assert isinstance(adapter, AutoGenAdapter)
        assert adapter.adapter_type == "autogen"

    def test_create_langgraph_with_config_object(self) -> None:
        """Test creating LangGraph adapter from config object."""
        registry = AdapterRegistry()
        config = LangGraphAdapterConfig(
            module="my.module",
            graph="my_graph",
            config={"recursion_limit": 50},
        )

        adapter = registry.create("langgraph", config)

        assert isinstance(adapter, LangGraphAdapter)

    def test_create_crewai_with_factory(self) -> None:
        """Test creating CrewAI adapter with factory config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "crewai",
            {
                "module": "my.module",
                "crew": "create_crew",
                "is_factory": True,
                "factory_args": {"model": "gpt-4"},
            },
        )

        assert isinstance(adapter, CrewAIAdapter)

    def test_create_autogen_with_user_proxy(self) -> None:
        """Test creating AutoGen adapter with user proxy config."""
        registry = AdapterRegistry()

        adapter = registry.create(
            "autogen",
            {
                "module": "my.module",
                "agent": "my_agent",
                "user_proxy": "my_user_proxy",
                "max_consecutive_auto_reply": 5,
            },
        )

        assert isinstance(adapter, AutoGenAdapter)
