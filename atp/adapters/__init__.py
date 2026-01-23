"""ATP Adapters for agent communication."""

from atp.adapters.autogen import AutoGenAdapter, AutoGenAdapterConfig
from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.adapters.cli import CLIAdapter, CLIAdapterConfig
from atp.adapters.container import (
    ContainerAdapter,
    ContainerAdapterConfig,
    ContainerResources,
)
from atp.adapters.crewai import CrewAIAdapter, CrewAIAdapterConfig
from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterNotFoundError,
    AdapterResponseError,
    AdapterTimeoutError,
)
from atp.adapters.http import HTTPAdapter, HTTPAdapterConfig
from atp.adapters.langgraph import LangGraphAdapter, LangGraphAdapterConfig
from atp.adapters.registry import AdapterRegistry, create_adapter, get_registry

__all__ = [
    # Base
    "AgentAdapter",
    "AdapterConfig",
    # HTTP
    "HTTPAdapter",
    "HTTPAdapterConfig",
    # Container
    "ContainerAdapter",
    "ContainerAdapterConfig",
    "ContainerResources",
    # CLI
    "CLIAdapter",
    "CLIAdapterConfig",
    # LangGraph
    "LangGraphAdapter",
    "LangGraphAdapterConfig",
    # CrewAI
    "CrewAIAdapter",
    "CrewAIAdapterConfig",
    # AutoGen
    "AutoGenAdapter",
    "AutoGenAdapterConfig",
    # Registry
    "AdapterRegistry",
    "get_registry",
    "create_adapter",
    # Exceptions
    "AdapterError",
    "AdapterTimeoutError",
    "AdapterConnectionError",
    "AdapterResponseError",
    "AdapterNotFoundError",
]
