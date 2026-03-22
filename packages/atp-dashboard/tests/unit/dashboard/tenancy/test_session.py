"""Tests for TenantAwareSession and TenantSessionFactory."""

from pathlib import Path

import pytest
from sqlalchemy import select

from atp.dashboard.models import Agent
from atp.dashboard.tenancy.models import DEFAULT_TENANT_ID
from atp.dashboard.tenancy.session import (
    TenantAwareSession,
    TenantSessionFactory,
    get_tenant_session_factory,
    set_tenant_factory,
)


class TestTenantAwareSession:
    """Tests for TenantAwareSession."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    @pytest.fixture
    async def engine(self, temp_db_path: str):
        """Create a test engine with tables."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atp.dashboard.models import Base

        engine = create_async_engine(temp_db_path)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine

        await engine.dispose()

    def test_session_properties(self, temp_db_path: str) -> None:
        """Test TenantAwareSession properties."""
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(temp_db_path)
        session = TenantAwareSession(engine, "acme-corp")

        assert session.tenant_id == "acme-corp"
        assert session.schema_name == "tenant_acme_corp"

    def test_session_default_tenant(self, temp_db_path: str) -> None:
        """Test session with default tenant."""
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(temp_db_path)
        session = TenantAwareSession(engine, DEFAULT_TENANT_ID)

        assert session.tenant_id == DEFAULT_TENANT_ID
        assert session.schema_name == "public"

    @pytest.mark.anyio
    async def test_session_context_manager(self, engine) -> None:
        """Test using session as context manager."""
        async with TenantAwareSession(engine, DEFAULT_TENANT_ID) as session:
            assert session is not None
            # Session should be usable
            result = await session.execute(select(Agent))
            assert result is not None

    @pytest.mark.anyio
    async def test_session_rollback_on_error(self, engine) -> None:
        """Test session rolls back on error."""
        try:
            async with TenantAwareSession(engine, DEFAULT_TENANT_ID) as session:
                agent = Agent(
                    name="test-agent",
                    agent_type="http",
                    config={},
                )
                session.add(agent)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Agent should not be in database
        async with TenantAwareSession(engine, DEFAULT_TENANT_ID) as session:
            result = await session.execute(
                select(Agent).where(Agent.name == "test-agent")
            )
            assert result.scalar_one_or_none() is None

    @pytest.mark.anyio
    async def test_session_autocommit(self, engine) -> None:
        """Test session with autocommit enabled."""
        async with TenantAwareSession(
            engine, DEFAULT_TENANT_ID, autocommit=True
        ) as session:
            agent = Agent(
                name="autocommit-agent",
                agent_type="http",
                config={},
            )
            session.add(agent)

        # Agent should be in database
        async with TenantAwareSession(engine, DEFAULT_TENANT_ID) as session:
            result = await session.execute(
                select(Agent).where(Agent.name == "autocommit-agent")
            )
            agent = result.scalar_one_or_none()
            assert agent is not None


class TestTenantSessionFactory:
    """Tests for TenantSessionFactory."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    @pytest.fixture
    async def engine(self, temp_db_path: str):
        """Create a test engine with tables."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atp.dashboard.models import Base

        engine = create_async_engine(temp_db_path)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine

        await engine.dispose()

    @pytest.mark.anyio
    async def test_factory_creation(self, engine) -> None:
        """Test creating a TenantSessionFactory."""
        factory = TenantSessionFactory(engine)
        assert factory is not None

    @pytest.mark.anyio
    async def test_factory_for_tenant(self, engine) -> None:
        """Test creating a session for a tenant."""
        factory = TenantSessionFactory(engine)
        session = factory.for_tenant("acme-corp")
        assert session.tenant_id == "acme-corp"

    @pytest.mark.anyio
    async def test_factory_session_context(self, engine) -> None:
        """Test factory session context manager."""
        factory = TenantSessionFactory(engine)

        async with factory.session(DEFAULT_TENANT_ID) as session:
            assert session is not None
            result = await session.execute(select(Agent))
            assert result is not None

    @pytest.mark.anyio
    async def test_factory_session_commit(self, engine) -> None:
        """Test factory session commits on success."""
        factory = TenantSessionFactory(engine)

        async with factory.session(DEFAULT_TENANT_ID) as session:
            agent = Agent(
                name="factory-agent",
                agent_type="http",
                config={},
            )
            session.add(agent)

        # Agent should be in database
        async with factory.session(DEFAULT_TENANT_ID) as session:
            result = await session.execute(
                select(Agent).where(Agent.name == "factory-agent")
            )
            agent = result.scalar_one_or_none()
            assert agent is not None

    @pytest.mark.anyio
    async def test_factory_session_rollback(self, engine) -> None:
        """Test factory session rolls back on error."""
        factory = TenantSessionFactory(engine)

        try:
            async with factory.session(DEFAULT_TENANT_ID) as session:
                agent = Agent(
                    name="rollback-agent",
                    agent_type="http",
                    config={},
                )
                session.add(agent)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Agent should not be in database
        async with factory.session(DEFAULT_TENANT_ID) as session:
            result = await session.execute(
                select(Agent).where(Agent.name == "rollback-agent")
            )
            assert result.scalar_one_or_none() is None


class TestGlobalFactory:
    """Tests for global factory functions."""

    def test_set_and_get_factory(self, tmp_path: Path) -> None:
        """Test setting and getting global factory."""
        from sqlalchemy.ext.asyncio import create_async_engine

        temp_db_path = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        engine = create_async_engine(temp_db_path)

        factory = TenantSessionFactory(engine)
        set_tenant_factory(factory)

        from atp.dashboard.tenancy.session import get_tenant_factory

        retrieved = get_tenant_factory()
        assert retrieved is factory

        # Clean up
        set_tenant_factory(None)

    def test_get_factory_not_initialized(self) -> None:
        """Test getting factory when not initialized."""
        set_tenant_factory(None)

        from atp.dashboard.tenancy.session import get_tenant_factory

        with pytest.raises(RuntimeError, match="not initialized"):
            get_tenant_factory()

    def test_get_tenant_session_factory(self, tmp_path: Path) -> None:
        """Test get_tenant_session_factory helper."""
        from sqlalchemy.ext.asyncio import create_async_engine

        temp_db_path = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        engine = create_async_engine(temp_db_path)

        factory = get_tenant_session_factory(engine)
        assert isinstance(factory, TenantSessionFactory)
