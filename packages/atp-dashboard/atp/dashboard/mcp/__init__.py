"""MCP server module — FastMCP instance, tools, auth, notifications."""

from __future__ import annotations

from fastmcp import FastMCP

from atp.dashboard.tournament.events import TournamentEventBus

# Module-level singletons. The bus is shared between the service layer
# (publish) and the MCP notification layer (subscribe). The FastMCP
# instance is mounted under /mcp in factory.py.
mcp_server: FastMCP = FastMCP("atp-platform-tournaments")
tournament_event_bus: TournamentEventBus = TournamentEventBus()
