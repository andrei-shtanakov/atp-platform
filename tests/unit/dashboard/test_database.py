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


class TestAddMissingColumnsServerDefaults:
    """Regression coverage for ``_add_missing_columns`` across every
    ``server_default`` pattern used in the models (as of LABS-96 audit):

    - ``String, server_default=""`` (LABS-96 prod incident)
    - ``String, server_default="latest"`` (versioned string)
    - ``Integer, server_default="2"`` (numeric literal as str)
    - ``DateTime, server_default=func.now()`` (SQL function)

    All four patterns are rendered via SQLAlchemy's own ``CreateColumn``
    DDL compiler to avoid the hand-rolled quoting footguns we hit.
    """

    @staticmethod
    def _fresh_db_with_pre_row(tmp_path: Path, table_name: str):  # type: ignore[no-untyped-def]
        import sqlalchemy as sa
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

        from atp.dashboard.database import Database

        class _Base(DeclarativeBase):
            pass

        class _Pre(_Base):  # noqa: F841
            __tablename__ = table_name
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

        url = f"sqlite+aiosqlite:///{tmp_path / f'{table_name}.db'}"
        db = Database(url=url)
        return db, _Base, sa, mapped_column, Mapped

    @staticmethod
    async def _run_with_patched_base(db, _Base):  # type: ignore[no-untyped-def]
        import atp.dashboard.database as db_mod
        from atp.dashboard.database import (
            _add_missing_columns as add_missing_columns,
        )

        original = db_mod.Base
        db_mod.Base = _Base  # type: ignore[assignment]
        try:
            await add_missing_columns(db)
        finally:
            db_mod.Base = original  # type: ignore[assignment]

    @pytest.mark.anyio
    async def test_empty_string_server_default(self, tmp_path: Path) -> None:
        """PR #59 regression: ``DEFAULT ''`` used to produce ``DEFAULT ``."""
        db, _Base, sa, mapped_column, Mapped = self._fresh_db_with_pre_row(
            tmp_path, "_rt_empty"
        )

        async with db.engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        async with db.session() as s:
            await s.execute(sa.text("INSERT INTO _rt_empty (id) VALUES (1)"))
            await s.commit()

        _Base.metadata.clear()

        class _Post(_Base):  # noqa: F841
            __tablename__ = "_rt_empty"
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
            agent_name: Mapped[str] = mapped_column(
                sa.String(100), nullable=False, server_default=""
            )

        await self._run_with_patched_base(db, _Base)

        async with db.session() as s:
            row = (
                await s.execute(sa.text("SELECT agent_name FROM _rt_empty WHERE id=1"))
            ).first()
            assert row is not None and row[0] == ""
        await db.close()

    @pytest.mark.anyio
    async def test_nonempty_string_server_default(self, tmp_path: Path) -> None:
        db, _Base, sa, mapped_column, Mapped = self._fresh_db_with_pre_row(
            tmp_path, "_rt_str"
        )

        async with db.engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        async with db.session() as s:
            await s.execute(sa.text("INSERT INTO _rt_str (id) VALUES (1)"))
            await s.commit()

        _Base.metadata.clear()

        class _Post(_Base):  # noqa: F841
            __tablename__ = "_rt_str"
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
            version: Mapped[str] = mapped_column(
                sa.String(50), nullable=False, server_default="latest"
            )

        await self._run_with_patched_base(db, _Base)

        async with db.session() as s:
            row = (
                await s.execute(sa.text("SELECT version FROM _rt_str WHERE id=1"))
            ).first()
            assert row is not None and row[0] == "latest"
        await db.close()

    @pytest.mark.anyio
    async def test_integer_with_numeric_string_default(self, tmp_path: Path) -> None:
        """Tournament model uses ``Integer, server_default="2"``. The old
        hand-rolled code emitted ``DEFAULT 2`` (bare int literal); the
        interim LABS-96 fix would have emitted ``DEFAULT '2'`` (string
        literal for an Integer column). Either works in SQLite but the
        SQLAlchemy-native render is the one that matches ``create_all``
        and is dialect-consistent.
        """
        db, _Base, sa, mapped_column, Mapped = self._fresh_db_with_pre_row(
            tmp_path, "_rt_int"
        )

        async with db.engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        async with db.session() as s:
            await s.execute(sa.text("INSERT INTO _rt_int (id) VALUES (1)"))
            await s.commit()

        _Base.metadata.clear()

        class _Post(_Base):  # noqa: F841
            __tablename__ = "_rt_int"
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
            num_players: Mapped[int] = mapped_column(
                sa.Integer, nullable=False, server_default="2"
            )

        await self._run_with_patched_base(db, _Base)

        async with db.session() as s:
            row = (
                await s.execute(sa.text("SELECT num_players FROM _rt_int WHERE id=1"))
            ).first()
            # SQLite stores '2' as TEXT if the DEFAULT is a string literal.
            # With the DDL-compiler fix, the default matches what create_all
            # would produce in the same DB.
            assert row is not None and str(row[0]) == "2"
        await db.close()

    @pytest.mark.anyio
    async def test_function_server_default_falls_back_to_nullable(
        self, tmp_path: Path, caplog
    ) -> None:
        """SQLite rejects ``ALTER TABLE ... DEFAULT CURRENT_TIMESTAMP``
        with "Cannot add a column with non-constant default". The helper
        must detect this, strip the default, add the column as NULL,
        and log a warning pointing at Alembic for the real fix.

        Latent bug discovered by the LABS-96 audit before it reached
        prod.
        """
        db, _Base, sa, mapped_column, Mapped = self._fresh_db_with_pre_row(
            tmp_path, "_rt_now"
        )

        async with db.engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        async with db.session() as s:
            await s.execute(sa.text("INSERT INTO _rt_now (id) VALUES (1)"))
            await s.commit()

        _Base.metadata.clear()

        from datetime import datetime

        class _Post(_Base):  # noqa: F841
            __tablename__ = "_rt_now"
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
            created_at: Mapped[datetime] = mapped_column(
                sa.DateTime, nullable=False, server_default=sa.func.now()
            )

        import logging

        with caplog.at_level(logging.WARNING, logger="atp.dashboard"):
            await self._run_with_patched_base(db, _Base)

        # Column must be there (nullable, no default).
        async with db.session() as s:
            row = (
                await s.execute(sa.text("SELECT created_at FROM _rt_now WHERE id=1"))
            ).first()
            # The pre-existing row had no created_at — column was added
            # nullable, value is NULL.
            assert row is not None and row[0] is None

        # And a warning was emitted pointing at Alembic.
        assert any(
            "non-constant defaults" in rec.message
            and "_rt_now.created_at" in rec.message
            for rec in caplog.records
        )
        await db.close()
