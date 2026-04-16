"""Shared fixtures for dashboard-level E2E tests.

Re-exports the ``tournament_uvicorn`` fixture from the tournament SC-1
e2e suite so sibling dashboard tests (About quickstart, etc.) can
stand up a real uvicorn + FastMCP server without duplicating ~150
lines of bring-up code.
"""

from tests.e2e.dashboard.tournament.test_e2e_30_round_pd_with_reconnect import (  # noqa: F401
    _mint_jwt,
    tournament_uvicorn,
)
