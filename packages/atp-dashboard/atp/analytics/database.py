"""Database connection and session management for ATP Analytics."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from atp.analytics.models import AnalyticsBase


class AnalyticsDatabase:
    """Database connection manager for ATP Analytics.

    Supports both SQLite (for development) and PostgreSQL (for production).
    """

    def __init__(
        self,
        url: str | None = None,
        echo: bool = False,
    ):
        """Initialize database connection.

        Args:
            url: Database URL. If None, uses SQLite at ~/.atp/analytics.db.
                 For SQLite: "sqlite+aiosqlite:///path/to/db.sqlite"
                 For PostgreSQL: "postgresql+asyncpg://user:pass@host/db"
            echo: Whether to echo SQL statements (for debugging).
        """
        if url is None:
            db_path = Path.home() / ".atp" / "analytics.db"
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
            if self._url.startswith("sqlite"):
                self._engine = create_async_engine(
                    self._url,
                    echo=self._echo,
                    connect_args={"check_same_thread": False},
                )
            else:
                self._engine = create_async_engine(
                    self._url,
                    echo=self._echo,
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
        """Create all analytics database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(AnalyticsBase.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all analytics database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(AnalyticsBase.metadata.drop_all)

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
                record = await session.get(CostRecord, 1)
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
            @app.get("/costs/{cost_id}")
            async def get_cost(
                cost_id: int,
                session: AsyncSession = Depends(db.get_session)
            ):
                return await session.get(CostRecord, cost_id)
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Global database instance
_analytics_database: AnalyticsDatabase | None = None


def get_analytics_database() -> AnalyticsDatabase:
    """Get the global analytics database instance."""
    global _analytics_database
    if _analytics_database is None:
        _analytics_database = AnalyticsDatabase()
    return _analytics_database


def set_analytics_database(db: AnalyticsDatabase) -> None:
    """Set the global analytics database instance (useful for testing)."""
    global _analytics_database
    _analytics_database = db


async def init_analytics_database(
    url: str | None = None, echo: bool = False
) -> AnalyticsDatabase:
    """Initialize and return the analytics database.

    Creates tables if they don't exist.

    Args:
        url: Database URL.
        echo: Whether to echo SQL statements.

    Returns:
        AnalyticsDatabase instance.
    """
    global _analytics_database
    _analytics_database = AnalyticsDatabase(url=url, echo=echo)
    await _analytics_database.create_tables()
    return _analytics_database
