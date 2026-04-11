"""MCP tool handlers — populated in Phase 7 tasks.

Importing this module registers tools as a side effect via the
``@mcp_server.tool()`` decorator. The factory imports this module
purely for those registrations.
"""

from __future__ import annotations

from atp.dashboard.mcp import mcp_server  # noqa: F401

# Tools will be registered in tasks 7.1-7.3.
