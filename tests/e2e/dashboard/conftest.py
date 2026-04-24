"""Shared fixtures for dashboard-level E2E tests.

Re-exports the ``tournament_uvicorn`` fixture from the tournament SC-1
e2e suite so sibling dashboard tests (About quickstart, etc.) can
stand up a real uvicorn + FastMCP server without duplicating ~150
lines of bring-up code.

Note on conftest shape: we deliberately import the fixture symbols
directly rather than using ``pytest_plugins = [...]``. The
``pytest_plugins`` mechanism is only honored in the **top-level**
conftest (``tests/conftest.py``); declaring it in a nested conftest
is deprecated and, as of pytest 9, a hard error
(``Defining 'pytest_plugins' in a non-top-level conftest is no
longer supported``). A plain ``from ... import`` works because
pytest picks up fixtures defined in conftest modules regardless of
where they were imported from.
"""

from tests.e2e.dashboard.tournament.test_e2e_30_round_pd_with_reconnect import (  # noqa: F401
    _mint_jwt,
    tournament_uvicorn,
)
