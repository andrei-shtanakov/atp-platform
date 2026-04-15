"""Tests for the partial unique index on ownerless Agents (LABS-15).

SQL treats NULL != NULL in regular unique constraints, so the existing
uq_agent_tenant_owner_name_version does NOT prevent duplicates when
owner_id IS NULL. A partial unique index closes that gap.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from atp.dashboard.models import Agent, Base


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_local = async_sessionmaker(engine, expire_on_commit=False)
    async with session_local() as sess:
        yield sess
    await engine.dispose()


@pytest.mark.anyio
async def test_duplicate_ownerless_agent_is_rejected(session: AsyncSession) -> None:
    """LABS-15: two ownerless agents with the same (tenant_id, name, version)
    must not coexist. Without the partial index, the second insert would
    silently succeed because NULL != NULL in SQL unique constraints.
    """
    session.add(
        Agent(
            name="dup",
            agent_type="cli",
            version="v1",
        )
    )
    await session.commit()

    session.add(
        Agent(
            name="dup",
            agent_type="cli",
            version="v1",
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.anyio
async def test_ownerless_agents_with_different_version_allowed(
    session: AsyncSession,
) -> None:
    """Different versions of the same ownerless name must coexist."""
    session.add(Agent(name="multi", agent_type="cli", version="v1"))
    await session.commit()
    session.add(Agent(name="multi", agent_type="cli", version="v2"))
    await session.commit()  # must not raise


@pytest.mark.anyio
async def test_partial_index_does_not_affect_owned_agents(
    session: AsyncSession,
) -> None:
    """Owned agents are governed by uq_agent_tenant_owner_name_version and
    must keep working the same way: different owners with same name+version
    are allowed; same owner with same name+version is rejected.
    """
    from atp.dashboard.models import User

    u1 = User(username="u1", email="u1@e.com", hashed_password="x")
    u2 = User(username="u2", email="u2@e.com", hashed_password="x")
    session.add_all([u1, u2])
    await session.commit()

    # Two different owners, same (name, version) — allowed
    session.add(Agent(name="shared", agent_type="cli", version="v1", owner_id=u1.id))
    session.add(Agent(name="shared", agent_type="cli", version="v1", owner_id=u2.id))
    await session.commit()

    # Same owner, same (name, version) — rejected
    session.add(Agent(name="shared", agent_type="cli", version="v1", owner_id=u1.id))
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.anyio
async def test_partial_index_does_not_interfere_with_ownerless_plus_owned(
    session: AsyncSession,
) -> None:
    """An ownerless agent and an owned agent with identical (name, version)
    can coexist: the partial index fires only for NULL, the full
    unique constraint fires only for non-NULL. Neither triggers here.
    """
    from atp.dashboard.models import User

    u = User(username="u", email="u@e.com", hashed_password="x")
    session.add(u)
    await session.commit()

    session.add(Agent(name="shared", agent_type="cli", version="v1"))  # ownerless
    session.add(Agent(name="shared", agent_type="cli", version="v1", owner_id=u.id))
    await session.commit()  # must not raise
