"""Shared fixtures for dashboard unit tests."""

import os
from collections.abc import Generator

import pytest

from atp.dashboard.v2.config import get_config


@pytest.fixture
def disable_dashboard_auth() -> Generator[None, None, None]:
    """Disable authentication for dashboard tests.

    Clears the get_config lru_cache and sets ATP_DISABLE_AUTH=true
    so that require_permission() skips auth checks.

    Usage: include this fixture in test functions or other fixtures
    that need auth bypassed.
    """
    old_value = os.environ.get("ATP_DISABLE_AUTH")
    os.environ["ATP_DISABLE_AUTH"] = "true"
    get_config.cache_clear()
    yield
    get_config.cache_clear()
    if old_value is None:
        os.environ.pop("ATP_DISABLE_AUTH", None)
    else:
        os.environ["ATP_DISABLE_AUTH"] = old_value
