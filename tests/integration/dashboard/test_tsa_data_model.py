"""Data-model tests for the tournament agent sandbox (PR-1)."""

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
async def db_session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    async with test_database.session() as session:
        yield session


@pytest.fixture
async def user(db_session: AsyncSession) -> User:
    u = User(username="u1", email="u1@t.com", hashed_password="x", is_active=True)
    db_session.add(u)
    await db_session.commit()
    await db_session.refresh(u)
    return u


class TestAgentPurpose:
    @pytest.mark.anyio
    async def test_purpose_defaults_to_benchmark(
        self, db_session: AsyncSession, user: User
    ) -> None:
        a = Agent(name="a1", agent_type="mcp", owner_id=user.id)
        db_session.add(a)
        await db_session.commit()
        await db_session.refresh(a)
        assert a.purpose == "benchmark"

    @pytest.mark.anyio
    async def test_purpose_tournament_roundtrips(
        self, db_session: AsyncSession, user: User
    ) -> None:
        a = Agent(name="a2", agent_type="mcp", owner_id=user.id, purpose="tournament")
        db_session.add(a)
        await db_session.commit()
        await db_session.refresh(a)
        assert a.purpose == "tournament"

    @pytest.mark.anyio
    async def test_purpose_invalid_rejected_by_check(
        self, db_session: AsyncSession, user: User
    ) -> None:
        a = Agent(name="a3", agent_type="mcp", owner_id=user.id, purpose="invalid")
        db_session.add(a)
        with pytest.raises(IntegrityError):
            await db_session.commit()
        # Explicit rollback so the session's pending-rollback state doesn't
        # propagate into the fixture teardown commit (which would otherwise
        # raise PendingRollbackError and mark the test as ERROR).
        await db_session.rollback()
