"""MCP (Model Context Protocol) adapter for agent communication."""

from atp.adapters.mcp.adapter import (
    MCPAdapter,
    MCPAdapterConfig,
    MCPPrompt,
    MCPResource,
    MCPServerInfo,
    MCPTool,
)
from atp.adapters.mcp.transport import (
    MCPTransport,
    SSETransport,
    SSETransportConfig,
    StdioTransport,
    StdioTransportConfig,
    TransportConfig,
    TransportState,
)

__all__ = [
    # Adapter
    "MCPAdapter",
    "MCPAdapterConfig",
    # Models
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPServerInfo",
    # Transport
    "MCPTransport",
    "StdioTransport",
    "StdioTransportConfig",
    "SSETransport",
    "SSETransportConfig",
    "TransportConfig",
    "TransportState",
]
