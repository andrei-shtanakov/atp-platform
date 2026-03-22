"""Tests for ATP Analytics database module."""

from pathlib import Path

import pytest

from atp.analytics.database import (
    AnalyticsDatabase,
    get_analytics_database,
    init_analytics_database,
    set_analytics_database,
)
from atp.analytics.models import CostBudget, CostRecord


class TestAnalyticsDatabase:
    """Tests for AnalyticsDatabase class."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test_analytics.db'}"

    @pytest.mark.anyio
    async def test_database_creation(self, temp_db_path: str) -> None:
        """Test creating AnalyticsDatabase instance."""
        db = AnalyticsDatabase(url=temp_db_path)

        assert db._url == temp_db_path
        assert db._engine is None
        assert db._session_factory is None

        await db.close()

    @pytest.mark.anyio
    async def test_database_engine_creation(self, temp_db_path: str) -> None:
        """Test lazy engine creation."""
        db = AnalyticsDatabase(url=temp_db_path)

        # Engine should be created on first access
        engine = db.engine
        assert engine is not None
        assert db._engine is engine

        # Second access should return same engine
        assert db.engine is engine

        await db.close()

    @pytest.mark.anyio
    async def test_database_session_factory(self, temp_db_path: str) -> None:
        """Test lazy session factory creation."""
        db = AnalyticsDatabase(url=temp_db_path)

        # Session factory should be created on first access
        factory = db.session_factory
        assert factory is not None
        assert db._session_factory is factory

        # Second access should return same factory
        assert db.session_factory is factory

        await db.close()

    @pytest.mark.anyio
    async def test_create_tables(self, temp_db_path: str) -> None:
        """Test creating database tables."""
        db = AnalyticsDatabase(url=temp_db_path)

        # Should not raise
        await db.create_tables()

        await db.close()

    @pytest.mark.anyio
    async def test_drop_tables(self, temp_db_path: str) -> None:
        """Test dropping database tables."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        # Should not raise
        await db.drop_tables()

        await db.close()

    @pytest.mark.anyio
    async def test_session_context_manager(self, temp_db_path: str) -> None:
        """Test session context manager."""
        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            assert session is not None

        await db.close()

    @pytest.mark.anyio
    async def test_close(self, temp_db_path: str) -> None:
        """Test closing database connection."""
        db = AnalyticsDatabase(url=temp_db_path)

        # Force engine creation
        _ = db.engine

        await db.close()

        assert db._engine is None
        assert db._session_factory is None

    @pytest.mark.anyio
    async def test_default_database_path(self) -> None:
        """Test default database path is used when url is None."""
        db = AnalyticsDatabase(url=None)

        assert "analytics.db" in db._url
        assert "sqlite+aiosqlite" in db._url

        await db.close()

    @pytest.mark.anyio
    async def test_echo_mode(self, temp_db_path: str) -> None:
        """Test echo mode for SQL debugging."""
        db = AnalyticsDatabase(url=temp_db_path, echo=True)

        assert db._echo is True
        engine = db.engine
        # Check echo is set on engine
        assert engine.echo is True

        await db.close()


class TestDatabaseGlobals:
    """Tests for global database instance functions."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test_analytics.db'}"

    @pytest.mark.anyio
    async def test_get_analytics_database(self) -> None:
        """Test getting global database instance."""
        db = get_analytics_database()
        assert db is not None
        assert isinstance(db, AnalyticsDatabase)

        # Should return same instance
        db2 = get_analytics_database()
        assert db is db2

        await db.close()

    @pytest.mark.anyio
    async def test_set_analytics_database(self, temp_db_path: str) -> None:
        """Test setting global database instance."""
        custom_db = AnalyticsDatabase(url=temp_db_path)

        set_analytics_database(custom_db)

        db = get_analytics_database()
        assert db is custom_db
        assert db._url == temp_db_path

        await custom_db.close()

    @pytest.mark.anyio
    async def test_init_analytics_database(self, temp_db_path: str) -> None:
        """Test initializing analytics database."""
        db = await init_analytics_database(url=temp_db_path, echo=False)

        assert db is not None
        assert db._url == temp_db_path

        # Should have created tables
        # Verify by trying to create a session
        async with db.session() as session:
            assert session is not None

        await db.close()


class TestDatabaseIntegration:
    """Integration tests for database with models."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test_analytics.db'}"

    @pytest.mark.anyio
    async def test_insert_cost_record(self, temp_db_path: str) -> None:
        """Test inserting a CostRecord into the database."""
        from datetime import datetime
        from decimal import Decimal

        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            record = CostRecord(
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=Decimal("0.015"),
            )
            session.add(record)
            await session.flush()

            assert record.id is not None
            assert record.id > 0

        await db.close()

    @pytest.mark.anyio
    async def test_insert_cost_budget(self, temp_db_path: str) -> None:
        """Test inserting a CostBudget into the database."""
        from decimal import Decimal

        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            budget = CostBudget(
                name="test-budget",
                period="daily",
                limit_usd=Decimal("100.00"),
                alert_threshold=0.8,
            )
            session.add(budget)
            await session.flush()

            assert budget.id is not None
            assert budget.id > 0

        await db.close()

    @pytest.mark.anyio
    async def test_query_cost_records(self, temp_db_path: str) -> None:
        """Test querying CostRecords from the database."""
        from datetime import datetime
        from decimal import Decimal

        from sqlalchemy import select

        db = AnalyticsDatabase(url=temp_db_path)
        await db.create_tables()

        # Insert records
        async with db.session() as session:
            for i in range(3):
                record = CostRecord(
                    timestamp=datetime.now(),
                    provider="anthropic",
                    model=f"model-{i}",
                    input_tokens=100 * (i + 1),
                    output_tokens=50 * (i + 1),
                    cost_usd=Decimal("0.01") * (i + 1),
                )
                session.add(record)

        # Query records
        async with db.session() as session:
            stmt = select(CostRecord).order_by(CostRecord.id)
            result = await session.execute(stmt)
            records = list(result.scalars().all())

            assert len(records) == 3
            assert records[0].model == "model-0"
            assert records[1].model == "model-1"
            assert records[2].model == "model-2"

        await db.close()
