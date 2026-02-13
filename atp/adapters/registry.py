"""Adapter registry for managing and creating adapters."""

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from .base import AdapterConfig, AgentAdapter
from .exceptions import AdapterNotFoundError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LazyEntry:
    """Deferred import reference for an adapter and its config."""

    module: str
    adapter_class_name: str
    config_class_name: str


# Map adapter types to the extras group needed for installation.
_ADAPTER_EXTRAS: dict[str, str] = {
    "bedrock": "bedrock",
    "vertex": "vertex",
    "azure_openai": "azure-openai",
}

# Built-in adapters: imported lazily on first access.
_BUILTIN_ADAPTERS: dict[str, _LazyEntry] = {
    "http": _LazyEntry("atp.adapters.http", "HTTPAdapter", "HTTPAdapterConfig"),
    "container": _LazyEntry(
        "atp.adapters.container",
        "ContainerAdapter",
        "ContainerAdapterConfig",
    ),
    "cli": _LazyEntry("atp.adapters.cli", "CLIAdapter", "CLIAdapterConfig"),
    "langgraph": _LazyEntry(
        "atp.adapters.langgraph",
        "LangGraphAdapter",
        "LangGraphAdapterConfig",
    ),
    "crewai": _LazyEntry(
        "atp.adapters.crewai",
        "CrewAIAdapter",
        "CrewAIAdapterConfig",
    ),
    "autogen": _LazyEntry(
        "atp.adapters.autogen",
        "AutoGenAdapter",
        "AutoGenAdapterConfig",
    ),
    "mcp": _LazyEntry("atp.adapters.mcp", "MCPAdapter", "MCPAdapterConfig"),
    "bedrock": _LazyEntry(
        "atp.adapters.bedrock",
        "BedrockAdapter",
        "BedrockAdapterConfig",
    ),
    "vertex": _LazyEntry(
        "atp.adapters.vertex",
        "VertexAdapter",
        "VertexAdapterConfig",
    ),
    "azure_openai": _LazyEntry(
        "atp.adapters.azure_openai",
        "AzureOpenAIAdapter",
        "AzureOpenAIAdapterConfig",
    ),
}


def _resolve_entry(
    entry: _LazyEntry,
) -> tuple[type[AgentAdapter], type[AdapterConfig]]:
    """Import and return (adapter_class, config_class) from a lazy entry."""
    mod = importlib.import_module(entry.module)
    adapter_cls: type[AgentAdapter] = getattr(mod, entry.adapter_class_name)
    config_cls: type[AdapterConfig] = getattr(mod, entry.config_class_name)
    return adapter_cls, config_cls


class AdapterRegistry:
    """
    Registry for adapter types.

    Provides factory methods for creating adapters from configuration.
    Adapter modules are imported lazily on first access.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in adapters."""
        self._adapters: dict[str, type[AgentAdapter]] = {}
        self._configs: dict[str, type[AdapterConfig]] = {}
        self._lazy: dict[str, _LazyEntry] = dict(_BUILTIN_ADAPTERS)

    def _resolve(self, adapter_type: str) -> None:
        """Resolve a lazy entry, importing the module on demand."""
        if adapter_type in self._adapters:
            return
        entry = self._lazy.get(adapter_type)
        if entry is None:
            return
        try:
            adapter_cls, config_cls = _resolve_entry(entry)
        except ImportError as exc:
            extras_hint = _ADAPTER_EXTRAS.get(adapter_type, adapter_type)
            raise ImportError(
                f"Adapter '{adapter_type}' requires additional "
                f"dependencies. Install them with: "
                f"uv add 'atp-platform[{extras_hint}]'"
            ) from exc
        self._adapters[adapter_type] = adapter_cls
        self._configs[adapter_type] = config_cls

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
        # Remove any lazy entry so _resolve won't overwrite
        self._lazy.pop(adapter_type, None)

    def unregister(self, adapter_type: str) -> bool:
        """
        Unregister an adapter type.

        Args:
            adapter_type: Identifier of the adapter to remove.

        Returns:
            True if adapter was removed, False if it didn't exist.
        """
        existed = adapter_type in self._adapters or (adapter_type in self._lazy)
        self._adapters.pop(adapter_type, None)
        self._configs.pop(adapter_type, None)
        self._lazy.pop(adapter_type, None)
        return existed

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
        self._resolve(adapter_type)
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
        self._resolve(adapter_type)
        if adapter_type not in self._configs:
            raise AdapterNotFoundError(adapter_type)
        return self._configs[adapter_type]

    def create(
        self,
        adapter_type: str,
        config: dict[str, Any] | AdapterConfig,
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
        all_types = set(self._adapters.keys()) | set(self._lazy.keys())
        return sorted(all_types)

    def is_registered(self, adapter_type: str) -> bool:
        """
        Check if an adapter type is registered.

        Args:
            adapter_type: Adapter type identifier.

        Returns:
            True if adapter is registered, False otherwise.
        """
        return adapter_type in self._adapters or adapter_type in self._lazy


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
