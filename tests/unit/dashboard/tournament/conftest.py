"""Shared fixtures for tournament service tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import atp.dashboard.tournament.models  # noqa: F401  (ensure model registration)
from atp.dashboard.models import Base, User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Fresh in-memory SQLite + all tables, one per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_local = async_sessionmaker(engine, expire_on_commit=False)
    async with session_local() as sess:
        yield sess
    await engine.dispose()


@pytest.fixture
async def admin_user(session: AsyncSession) -> User:
    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password="x",
        is_admin=True,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest.fixture
async def alice(session: AsyncSession) -> User:
    user = User(
        username="alice",
        email="alice@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest.fixture
async def bob(session: AsyncSession) -> User:
    user = User(
        username="bob",
        email="bob@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


@pytest.fixture
def event_bus() -> TournamentEventBus:
    return TournamentEventBus()


@pytest.fixture(autouse=True)
def _clear_el_farol_cache():
    """Spec §7.1 fixture hygiene for _el_farol_for lru_cache."""
    from atp.dashboard.tournament.service import _el_farol_for

    yield
    _el_farol_for.cache_clear()
