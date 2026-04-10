"""Database connection and session management for ATP Dashboard."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from atp.dashboard.models import Base


class Database:
    """Database connection manager for ATP Dashboard.

    Supports both SQLite (for development) and PostgreSQL (for production).
    """

    def __init__(
        self,
        url: str | None = None,
        echo: bool = False,
    ):
        """Initialize database connection.

        Args:
            url: Database URL. If None, uses SQLite at ~/.atp/dashboard.db.
                 For SQLite: "sqlite+aiosqlite:///path/to/db.sqlite"
                 For PostgreSQL: "postgresql+asyncpg://user:pass@host/db"
            echo: Whether to echo SQL statements (for debugging).
        """
        if url is None:
            # Default to SQLite in user's home directory
            db_path = Path.home() / ".atp" / "dashboard.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+aiosqlite:///{db_path}"

        self._url = url
        self._echo = echo
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine, creating it if needed."""
        if self._engine is None:
            # Use different settings for SQLite vs PostgreSQL
            if self._url.startswith("sqlite"):
                self._engine = create_async_engine(
                    self._url,
                    echo=self._echo,
                    # SQLite-specific settings
                    connect_args={"check_same_thread": False},
                )
            else:
                self._engine = create_async_engine(
                    self._url,
                    echo=self._echo,
                    # PostgreSQL-specific settings
                    pool_size=5,
                    max_overflow=10,
                )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory, creating it if needed."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session as an async context manager.

        Example:
            async with db.session() as session:
                user = await session.get(User, 1)
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency for FastAPI to get a database session.

        Example (in FastAPI):
            @app.get("/users/{user_id}")
            async def get_user(
                user_id: int,
                session: AsyncSession = Depends(db.get_session)
            ):
                return await session.get(User, user_id)
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Global database instance (can be overridden for testing)
_database: Database | None = None


def get_database() -> Database:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database


def set_database(db: Database) -> None:
    """Set the global database instance (useful for testing)."""
    global _database
    _database = db


async def init_database(url: str | None = None, echo: bool = False) -> Database:
    """Initialize and return the database.

    Creates tables if they don't exist, adds missing columns to existing
    tables, and seeds default RBAC roles.

    Args:
        url: Database URL.
        echo: Whether to echo SQL statements.

    Returns:
        Database instance.
    """
    global _database
    _database = Database(url=url, echo=echo)
    await _database.create_tables()
    await _add_missing_columns(_database)
    await _backfill_run_user_id(_database)
    await _seed_default_roles(_database)
    return _database


async def _backfill_run_user_id(db: Database) -> None:
    """Assign ownership to legacy benchmark_runs with NULL user_id.

    The ``Run.user_id`` column started life as nullable and production
    accumulated rows without an owner.  The IDOR fix (2026-04-10) requires
    every run to have an owner so ``_load_run_for_user`` can enforce
    ownership.  This helper backfills any NULL user_id to the lowest-id
    admin user.  If no admin exists but NULL rows are present, we log a
    warning and skip — app still boots, but those rows are effectively
    dead weight until an admin is created.

    Mirrors the Alembic migration ``c8d5f2a91234`` but runs at every
    startup (idempotent) because ``init_database`` does not invoke
    Alembic automatically.
    """
    import logging

    from sqlalchemy import text

    logger = logging.getLogger("atp.dashboard")

    async with db.engine.begin() as conn:
        try:
            null_count_row = await conn.execute(
                text("SELECT COUNT(*) FROM benchmark_runs WHERE user_id IS NULL")
            )
        except Exception:
            # benchmark_runs table doesn't exist yet (fresh DB before
            # create_tables) — nothing to backfill.
            return

        null_count = null_count_row.scalar_one()
        if not null_count:
            return

        admin_row = await conn.execute(
            text("SELECT id FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1")
        )
        admin_id = admin_row.scalar_one_or_none()
        if admin_id is None:
            logger.warning(
                "benchmark_runs: %d rows have NULL user_id but no admin "
                "user exists — skipping backfill. Ownership-protected "
                "endpoints will return 404 for these runs.",
                null_count,
            )
            return

        await conn.execute(
            text("UPDATE benchmark_runs SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": admin_id},
        )
        logger.info(
            "Backfilled benchmark_runs.user_id for %d legacy rows -> admin id=%d",
            null_count,
            admin_id,
        )


async def _add_missing_columns(db: Database) -> None:
    """Add columns that exist in models but not in the DB.

    SQLAlchemy create_all() only creates new tables — it does not ALTER
    existing ones.  This helper inspects every mapped table and issues
    ALTER TABLE ADD COLUMN for anything the DB is missing.  Runs once at
    startup and is idempotent.
    """
    import logging

    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy import text

    logger = logging.getLogger("atp.dashboard")

    def _get_existing_columns(sync_conn: Any, table_name: str) -> set[str]:
        inspector = sa_inspect(sync_conn)
        assert inspector is not None
        return {c["name"] for c in inspector.get_columns(table_name)}

    async with db.engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            try:
                existing = await conn.run_sync(_get_existing_columns, table.name)
            except Exception:
                # Table doesn't exist yet — create_tables() handles it
                continue

            for column in table.columns:
                if column.name in existing:
                    continue

                col_type = column.type.compile(db.engine.dialect)
                nullable = "NULL" if column.nullable else "NOT NULL"
                default = ""
                if column.server_default is not None:
                    default = f" DEFAULT {column.server_default.arg}"  # type: ignore[union-attr]

                stmt = (
                    f"ALTER TABLE {table.name} "
                    f"ADD COLUMN {column.name} {col_type} {nullable}{default}"
                )
                logger.info("Auto-adding column: %s.%s", table.name, column.name)
                await conn.execute(text(stmt))


async def _seed_default_roles(db: Database) -> None:
    """Create default RBAC roles if they don't exist."""
    from atp.dashboard.rbac.models import (
        DEFAULT_ROLES,
        Role,
        RolePermission,
    )

    async with db.session_factory() as session:
        for role_name, role_def in DEFAULT_ROLES.items():
            from sqlalchemy import select

            result = await session.execute(select(Role).where(Role.name == role_name))
            if result.scalar_one_or_none() is not None:
                continue

            role = Role(
                name=role_name,
                description=role_def.description,
                is_system=role_def.is_system,
            )
            session.add(role)
            await session.flush()

            for perm in role_def.permissions:
                session.add(RolePermission(role_id=role.id, permission=perm.value))

        await session.commit()
