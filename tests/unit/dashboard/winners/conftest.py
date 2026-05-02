"""Shared fixtures for winners aggregation tests.

In-memory SQLite + ``Base.metadata.create_all``, mirroring
``tests/unit/dashboard/tournament/conftest.py`` but without the
tournament-engine cache hooks (we don't run any game logic here, only
read SQL).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import atp.dashboard.tournament.models  # noqa: F401  (register tournament tables)
from atp.dashboard.models import Base
from atp.dashboard.v2.services.winners import reset_caches_for_tests


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Drop QueryCache singletons between tests."""
    reset_caches_for_tests()
    yield
    reset_caches_for_tests()
