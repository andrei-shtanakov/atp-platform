"""ATP Adapters for agent communication."""

from atp.adapters.autogen import AutoGenAdapter, AutoGenAdapterConfig
from atp.adapters.azure_openai import AzureOpenAIAdapter, AzureOpenAIAdapterConfig
from atp.adapters.base import AdapterConfig, AgentAdapter, track_response_cost
from atp.adapters.bedrock import BedrockAdapter, BedrockAdapterConfig
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
from atp.adapters.mcp import (
    MCPAdapter,
    MCPAdapterConfig,
    MCPPrompt,
    MCPResource,
    MCPServerInfo,
    MCPTool,
)
from atp.adapters.registry import AdapterRegistry, create_adapter, get_registry
from atp.adapters.vertex import VertexAdapter, VertexAdapterConfig

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
