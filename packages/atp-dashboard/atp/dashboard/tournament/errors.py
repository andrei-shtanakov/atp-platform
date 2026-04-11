"""Tournament service exceptions.

These are the only error channel out of TournamentService. Service
methods raise these; transport layers (MCP tool handlers, REST routes)
catch them and translate to ToolError / HTTPException.
"""

from __future__ import annotations


class TournamentError(Exception):
    """Base for all tournament service errors."""


class ValidationError(TournamentError):
    """Invalid input shape, missing required field, unknown game_type,
    action that doesn't match the game's action schema, etc.

    Maps to HTTP 422 / MCP ToolError(422).
    """


class ConflictError(TournamentError):
    """State machine violation: join in ACTIVE, double make_move,
    leave during ACTIVE, etc.

    Maps to HTTP 409 / MCP ToolError(409).
    """


class NotFoundError(TournamentError):
    """Resource does not exist OR is not owned by the requesting user.

    Per the enumeration-guard pattern (Issue 1 fix, design spec
    §Error handling), 404 is returned regardless of which case
    applies. Clients cannot distinguish 'doesn't exist' from
    'exists but not yours'.

    Maps to HTTP 404 / MCP ToolError(404).
    """
