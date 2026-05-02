"""Shared fixtures for winners integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app
from atp.dashboard.v2.services.winners import reset_caches_for_tests


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """Drop QueryCache singletons between tests so cached entries from
    one test don't leak into another."""
    reset_caches_for_tests()
    yield
    reset_caches_for_tests()


@pytest.fixture
async def client(test_database: Database) -> AsyncIterator[AsyncClient]:
    """AsyncClient backed by a fresh test app whose ``get_db_session``
    dependency is rerouted to ``test_database``."""
    app = create_test_app()

    async def _override_session():
        async with test_database.session() as s:
            yield s

    app.dependency_overrides[get_db_session] = _override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
