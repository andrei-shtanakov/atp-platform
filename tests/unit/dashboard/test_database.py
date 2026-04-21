"""Tests for ATP Dashboard database module."""

from pathlib import Path

import pytest

from atp.dashboard.database import (
    Database,
    get_database,
    init_database,
    set_database,
)


class TestDatabase:
    """Tests for Database class."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    def test_database_init_default(self) -> None:
        """Test database initialization with default URL."""
        db = Database()
        assert db._url is not None
        assert "sqlite" in db._url
        assert db._engine is None
        assert db._session_factory is None

    def test_database_init_custom_url(self, temp_db_path: str) -> None:
        """Test database initialization with custom URL."""
        db = Database(url=temp_db_path)
        assert db._url == temp_db_path

    def test_database_init_echo(self, temp_db_path: str) -> None:
        """Test database initialization with echo."""
        db = Database(url=temp_db_path, echo=True)
        assert db._echo is True

    def test_engine_property_creates_engine(self, temp_db_path: str) -> None:
        """Test that engine property creates engine lazily."""
        db = Database(url=temp_db_path)
        assert db._engine is None
        engine = db.engine
        assert engine is not None
        assert db._engine is engine

    def test_session_factory_property(self, temp_db_path: str) -> None:
        """Test that session_factory property creates factory lazily."""
        db = Database(url=temp_db_path)
        assert db._session_factory is None
        factory = db.session_factory
        assert factory is not None
        assert db._session_factory is factory

    @pytest.mark.anyio
    async def test_create_tables(self, temp_db_path: str) -> None:
        """Test creating database tables."""
        db = Database(url=temp_db_path)
        await db.create_tables()
        # Tables should be created without error
        await db.close()

    @pytest.mark.anyio
    async def test_drop_tables(self, temp_db_path: str) -> None:
        """Test dropping database tables."""
        db = Database(url=temp_db_path)
        await db.create_tables()
        await db.drop_tables()
        await db.close()

    @pytest.mark.anyio
    async def test_close(self, temp_db_path: str) -> None:
        """Test closing database connection."""
        db = Database(url=temp_db_path)
        _ = db.engine  # Force engine creation
        await db.close()
        assert db._engine is None
        assert db._session_factory is None

    @pytest.mark.anyio
    async def test_session_context_manager(self, temp_db_path: str) -> None:
        """Test session context manager."""
        db = Database(url=temp_db_path)
        await db.create_tables()

        async with db.session() as session:
            assert session is not None

        await db.close()

    @pytest.mark.anyio
    async def test_session_rollback_on_error(self, temp_db_path: str) -> None:
        """Test that session rolls back on error."""
        db = Database(url=temp_db_path)
        await db.create_tables()

        with pytest.raises(ValueError):
            async with db.session() as _:
                raise ValueError("Test error")

        await db.close()

    @pytest.mark.anyio
    async def test_get_session_generator(self, temp_db_path: str) -> None:
        """Test get_session generator."""
        db = Database(url=temp_db_path)
        await db.create_tables()

        async for session in db.get_session():
            assert session is not None
            break

        await db.close()


class TestDatabaseGlobals:
    """Tests for global database functions."""

    def test_get_database_creates_instance(self) -> None:
        """Test that get_database creates a database instance."""
        # Reset global
        set_database(None)  # type: ignore
        db = get_database()
        assert db is not None
        assert isinstance(db, Database)

    def test_get_database_returns_same_instance(self) -> None:
        """Test that get_database returns the same instance."""
        db1 = get_database()
        db2 = get_database()
        assert db1 is db2

    def test_set_database(self, tmp_path: Path) -> None:
        """Test setting a custom database instance."""
        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        custom_db = Database(url=url)
        set_database(custom_db)
        assert get_database() is custom_db

    @pytest.mark.anyio
    async def test_init_database(self, tmp_path: Path) -> None:
        """Test init_database function."""
        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        db = await init_database(url=url)
        assert db is not None
        assert get_database() is db
        await db.close()


class TestDatabaseWithPostgres:
    """Tests for PostgreSQL-specific behavior."""

    def test_postgres_url_handling(self) -> None:
        """Test that PostgreSQL URLs are handled differently."""
        # Note: This doesn't actually connect to PostgreSQL,
        # just tests URL detection
        db = Database(url="postgresql+asyncpg://user:pass@localhost/db")
        assert "postgresql" in db._url

    def test_sqlite_connect_args(self, tmp_path: Path) -> None:
        """Test SQLite-specific connection arguments."""
        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        db = Database(url=url)
        engine = db.engine
        # Engine should be created with SQLite-specific settings
        assert engine is not None


class TestAddMissingColumnsStringDefault:
    """Regression: a NOT NULL column with ``server_default=""`` used to
    emit ``ALTER TABLE ... DEFAULT `` (empty, unterminated) and crash
    startup with ``sqlite3.OperationalError: incomplete input``.

    Production incident 2026-04-21 (prod 502 after PR #58 rebuild)
    surfaced the latent bug introduced when ``suite_executions.agent_name``
    was added in LABS-54 Phase 1. Fix quotes string server_defaults.
    """

    @pytest.mark.anyio
    async def test_empty_string_server_default_is_quoted(self, tmp_path: Path) -> None:
        import sqlalchemy as sa
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

        from atp.dashboard.database import Database
        from atp.dashboard.database import (
            _add_missing_columns as add_missing_columns,
        )

        class _Base(DeclarativeBase):
            pass

        # Pre-existing table without the new column.
        class _Pre(_Base):
            __tablename__ = "_rt_empty_default"
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

        url = f"sqlite+aiosqlite:///{tmp_path / 'rt.db'}"
        db = Database(url=url)
        async with db.engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

        async with db.session() as s:
            await s.execute(sa.text("INSERT INTO _rt_empty_default (id) VALUES (1)"))
            await s.commit()

        # Evolve the model: add agent_name NOT NULL DEFAULT "".
        _Base.metadata.clear()

        class _Post(_Base):
            __tablename__ = "_rt_empty_default"
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
            agent_name: Mapped[str] = mapped_column(
                sa.String(100), nullable=False, server_default=""
            )

        import atp.dashboard.database as db_mod

        original_base = db_mod.Base
        db_mod.Base = _Base  # type: ignore[assignment]
        try:
            # Must NOT raise; previously crashed with
            # "sqlite3.OperationalError: incomplete input".
            await add_missing_columns(db)
        finally:
            db_mod.Base = original_base  # type: ignore[assignment]

        async with db.session() as s:
            row = (
                await s.execute(
                    sa.text("SELECT agent_name FROM _rt_empty_default WHERE id=1")
                )
            ).first()
            assert row is not None
            assert row[0] == ""

        await db.close()
