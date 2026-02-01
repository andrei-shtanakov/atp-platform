"""Shared fixtures for dashboard integration tests.

These fixtures provide authentication support for tests that require
authenticated access to dashboard API endpoints.
"""

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.rbac import Role, RolePermission, UserRole
from atp.dashboard.tenancy.models import Tenant


@pytest.fixture
async def test_database():
    """Create and configure a test database with RBAC tables."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    # Create all tables
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Create additional RBAC tables
        await conn.run_sync(
            lambda c: Tenant.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: Role.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: RolePermission.__table__.create(c, checkfirst=True)  # type: ignore
        )
        await conn.run_sync(
            lambda c: UserRole.__table__.create(c, checkfirst=True)  # type: ignore
        )
    # Set as global database so auth functions use it
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore


@pytest.fixture
async def db_session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async with test_database.session() as session:
        yield session


@pytest.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=True,
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Generate JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.username, "user_id": admin_user.id}
    )


@pytest.fixture
def auth_headers(admin_token: str) -> dict[str, str]:
    """Return authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {admin_token}"}
