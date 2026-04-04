"""Shared fixtures for dashboard unit tests."""

import os
from collections.abc import Generator

import pytest

from atp.dashboard.v2.config import get_config

# Ensure test env vars are set for entire unit test session.
# Routes that call get_config() need ATP_SECRET_KEY to not raise in non-debug mode.
os.environ.setdefault("ATP_SECRET_KEY", "unit-test-secret-key")
os.environ.setdefault("ATP_DEBUG", "true")


@pytest.fixture
def disable_dashboard_auth() -> Generator[None, None, None]:
    """Disable authentication for dashboard tests.

    Clears the get_config lru_cache and sets ATP_DISABLE_AUTH=true
    so that require_permission() skips auth checks.

    Usage: include this fixture in test functions or other fixtures
    that need auth bypassed.
    """
    old_disable_auth = os.environ.get("ATP_DISABLE_AUTH")
    old_secret_key = os.environ.get("ATP_SECRET_KEY")
    old_debug = os.environ.get("ATP_DEBUG")
    os.environ["ATP_DISABLE_AUTH"] = "true"
    os.environ["ATP_SECRET_KEY"] = "test-secret-key"
    os.environ["ATP_DEBUG"] = "true"
    get_config.cache_clear()
    yield
    get_config.cache_clear()
    for key, old_val in [
        ("ATP_DISABLE_AUTH", old_disable_auth),
        ("ATP_SECRET_KEY", old_secret_key),
        ("ATP_DEBUG", old_debug),
    ]:
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val
