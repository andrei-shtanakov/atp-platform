"""Adapter registry for managing and creating adapters."""

from typing import Any

from .autogen import AutoGenAdapter, AutoGenAdapterConfig
from .base import AdapterConfig, AgentAdapter
from .cli import CLIAdapter, CLIAdapterConfig
from .container import ContainerAdapter, ContainerAdapterConfig
from .crewai import CrewAIAdapter, CrewAIAdapterConfig
from .exceptions import AdapterNotFoundError
from .http import HTTPAdapter, HTTPAdapterConfig
from .langgraph import LangGraphAdapter, LangGraphAdapterConfig


class AdapterRegistry:
    """
    Registry for adapter types.

    Provides factory methods for creating adapters from configuration.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in adapters."""
        self._adapters: dict[str, type[AgentAdapter]] = {}
        self._configs: dict[str, type[AdapterConfig]] = {}

        # Register built-in adapters
        self.register("http", HTTPAdapter, HTTPAdapterConfig)
        self.register("container", ContainerAdapter, ContainerAdapterConfig)
        self.register("cli", CLIAdapter, CLIAdapterConfig)
        self.register("langgraph", LangGraphAdapter, LangGraphAdapterConfig)
        self.register("crewai", CrewAIAdapter, CrewAIAdapterConfig)
        self.register("autogen", AutoGenAdapter, AutoGenAdapterConfig)

    def register(
        self,
        adapter_type: str,
        adapter_class: type[AgentAdapter],
        config_class: type[AdapterConfig],
    ) -> None:
        """
        Register an adapter type.

        Args:
            adapter_type: Unique identifier for the adapter type.
            adapter_class: Adapter class to instantiate.
            config_class: Configuration class for the adapter.
        """
        self._adapters[adapter_type] = adapter_class
        self._configs[adapter_type] = config_class

    def unregister(self, adapter_type: str) -> bool:
        """
        Unregister an adapter type.

        Args:
            adapter_type: Identifier of the adapter to remove.

        Returns:
            True if adapter was removed, False if it didn't exist.
        """
        if adapter_type in self._adapters:
            del self._adapters[adapter_type]
            del self._configs[adapter_type]
            return True
        return False

    def get_adapter_class(self, adapter_type: str) -> type[AgentAdapter]:
        """
        Get the adapter class for a type.

        Args:
            adapter_type: Adapter type identifier.

        Returns:
            Adapter class.

        Raises:
            AdapterNotFoundError: If adapter type is not registered.
        """
        if adapter_type not in self._adapters:
            raise AdapterNotFoundError(adapter_type)
        return self._adapters[adapter_type]

    def get_config_class(self, adapter_type: str) -> type[AdapterConfig]:
        """
        Get the configuration class for an adapter type.

        Args:
            adapter_type: Adapter type identifier.

        Returns:
            Configuration class.

        Raises:
            AdapterNotFoundError: If adapter type is not registered.
        """
        if adapter_type not in self._configs:
            raise AdapterNotFoundError(adapter_type)
        return self._configs[adapter_type]

    def create(
        self, adapter_type: str, config: dict[str, Any] | AdapterConfig
    ) -> AgentAdapter:
        """
        Create an adapter instance.

        Args:
            adapter_type: Adapter type identifier.
            config: Configuration dict or object for the adapter.

        Returns:
            Configured adapter instance.

        Raises:
            AdapterNotFoundError: If adapter type is not registered.
            ValueError: If configuration is invalid.
        """
        adapter_class = self.get_adapter_class(adapter_type)
        config_class = self.get_config_class(adapter_type)

        if isinstance(config, dict):
            config_obj = config_class.model_validate(config)
        else:
            config_obj = config

        return adapter_class(config_obj)

    def list_adapters(self) -> list[str]:
        """
        List all registered adapter types.

        Returns:
            List of adapter type identifiers.
        """
        return list(self._adapters.keys())

    def is_registered(self, adapter_type: str) -> bool:
        """
        Check if an adapter type is registered.

        Args:
            adapter_type: Adapter type identifier.

        Returns:
            True if adapter is registered, False otherwise.
        """
        return adapter_type in self._adapters


# Global registry instance
_default_registry: AdapterRegistry | None = None


def get_registry() -> AdapterRegistry:
    """
    Get the global adapter registry.

    Returns:
        Global AdapterRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = AdapterRegistry()
    return _default_registry


def create_adapter(
    adapter_type: str, config: dict[str, Any] | AdapterConfig
) -> AgentAdapter:
    """
    Create an adapter using the global registry.

    Args:
        adapter_type: Adapter type identifier.
        config: Configuration for the adapter.

    Returns:
        Configured adapter instance.
    """
    return get_registry().create(adapter_type, config)
