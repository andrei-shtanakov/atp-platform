"""ATP Adapters for agent communication."""

import importlib
from typing import TYPE_CHECKING, Any

from atp.adapters.base import AdapterConfig, AgentAdapter, track_response_cost
from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterNotFoundError,
    AdapterResponseError,
    AdapterTimeoutError,
)
from atp.adapters.registry import AdapterRegistry, create_adapter, get_registry

if TYPE_CHECKING:
    from atp.adapters.autogen import AutoGenAdapter, AutoGenAdapterConfig
    from atp.adapters.azure_openai import (
        AzureOpenAIAdapter,
        AzureOpenAIAdapterConfig,
    )
    from atp.adapters.bedrock import BedrockAdapter, BedrockAdapterConfig
    from atp.adapters.cli import CLIAdapter, CLIAdapterConfig
    from atp.adapters.container import (
        ContainerAdapter,
        ContainerAdapterConfig,
        ContainerResources,
    )
    from atp.adapters.crewai import CrewAIAdapter, CrewAIAdapterConfig
    from atp.adapters.http import HTTPAdapter, HTTPAdapterConfig
    from atp.adapters.langgraph import (
        LangGraphAdapter,
        LangGraphAdapterConfig,
    )
    from atp.adapters.mcp import (
        MCPAdapter,
        MCPAdapterConfig,
        MCPPrompt,
        MCPResource,
        MCPServerInfo,
        MCPTool,
    )
    from atp.adapters.vertex import VertexAdapter, VertexAdapterConfig

# Mapping from public name to (module_path, attribute_name).
# These are resolved lazily on first access via __getattr__.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # HTTP
    "HTTPAdapter": ("atp.adapters.http", "HTTPAdapter"),
    "HTTPAdapterConfig": ("atp.adapters.http", "HTTPAdapterConfig"),
    # Container
    "ContainerAdapter": (
        "atp.adapters.container",
        "ContainerAdapter",
    ),
    "ContainerAdapterConfig": (
        "atp.adapters.container",
        "ContainerAdapterConfig",
    ),
    "ContainerResources": (
        "atp.adapters.container",
        "ContainerResources",
    ),
    # CLI
    "CLIAdapter": ("atp.adapters.cli", "CLIAdapter"),
    "CLIAdapterConfig": ("atp.adapters.cli", "CLIAdapterConfig"),
    # LangGraph
    "LangGraphAdapter": (
        "atp.adapters.langgraph",
        "LangGraphAdapter",
    ),
    "LangGraphAdapterConfig": (
        "atp.adapters.langgraph",
        "LangGraphAdapterConfig",
    ),
    # CrewAI
    "CrewAIAdapter": ("atp.adapters.crewai", "CrewAIAdapter"),
    "CrewAIAdapterConfig": (
        "atp.adapters.crewai",
        "CrewAIAdapterConfig",
    ),
    # AutoGen
    "AutoGenAdapter": ("atp.adapters.autogen", "AutoGenAdapter"),
    "AutoGenAdapterConfig": (
        "atp.adapters.autogen",
        "AutoGenAdapterConfig",
    ),
    # Azure OpenAI
    "AzureOpenAIAdapter": (
        "atp.adapters.azure_openai",
        "AzureOpenAIAdapter",
    ),
    "AzureOpenAIAdapterConfig": (
        "atp.adapters.azure_openai",
        "AzureOpenAIAdapterConfig",
    ),
    # MCP
    "MCPAdapter": ("atp.adapters.mcp", "MCPAdapter"),
    "MCPAdapterConfig": ("atp.adapters.mcp", "MCPAdapterConfig"),
    "MCPTool": ("atp.adapters.mcp", "MCPTool"),
    "MCPResource": ("atp.adapters.mcp", "MCPResource"),
    "MCPPrompt": ("atp.adapters.mcp", "MCPPrompt"),
    "MCPServerInfo": ("atp.adapters.mcp", "MCPServerInfo"),
    # Bedrock
    "BedrockAdapter": ("atp.adapters.bedrock", "BedrockAdapter"),
    "BedrockAdapterConfig": (
        "atp.adapters.bedrock",
        "BedrockAdapterConfig",
    ),
    # Vertex
    "VertexAdapter": ("atp.adapters.vertex", "VertexAdapter"),
    "VertexAdapterConfig": (
        "atp.adapters.vertex",
        "VertexAdapterConfig",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        value = getattr(mod, attr_name)
        # Cache in module globals so __getattr__ is not called again.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base
    "AgentAdapter",
    "AdapterConfig",
    "track_response_cost",
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
    # Azure OpenAI
    "AzureOpenAIAdapter",
    "AzureOpenAIAdapterConfig",
    # MCP
    "MCPAdapter",
    "MCPAdapterConfig",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPServerInfo",
    # Bedrock
    "BedrockAdapter",
    "BedrockAdapterConfig",
    # Vertex
    "VertexAdapter",
    "VertexAdapterConfig",
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
