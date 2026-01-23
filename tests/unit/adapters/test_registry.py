"""Unit tests for AdapterRegistry."""

import pytest

from atp.adapters import (
    AdapterConfig,
    AdapterNotFoundError,
    AdapterRegistry,
    AgentAdapter,
    CLIAdapter,
    CLIAdapterConfig,
    ContainerAdapter,
    ContainerAdapterConfig,
    HTTPAdapter,
    HTTPAdapterConfig,
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

    def test_list_adapters(self) -> None:
        """Test listing all registered adapters."""
        registry = AdapterRegistry()
        adapters = registry.list_adapters()

        assert "http" in adapters
        assert "container" in adapters
        assert "cli" in adapters

    def test_get_adapter_class(self) -> None:
        """Test getting adapter class."""
        registry = AdapterRegistry()

        assert registry.get_adapter_class("http") is HTTPAdapter
        assert registry.get_adapter_class("container") is ContainerAdapter
        assert registry.get_adapter_class("cli") is CLIAdapter

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
