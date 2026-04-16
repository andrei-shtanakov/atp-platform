"""Shared fixtures for dashboard-level E2E tests.

Exposes the tournament suite's shared bring-up fixture to sibling
dashboard tests without directly importing fixture symbols from the
tournament test module at conftest import time.
"""

from importlib import import_module

import pytest

pytest_plugins = [
    "tests.e2e.dashboard.tournament.test_e2e_30_round_pd_with_reconnect",
]


def _mint_jwt(*args, **kwargs):
    tournament_module = import_module(
        "tests.e2e.dashboard.tournament.test_e2e_30_round_pd_with_reconnect"
    )
    return tournament_module._mint_jwt(*args, **kwargs)


__all__ = ["_mint_jwt", "pytest_plugins"]
