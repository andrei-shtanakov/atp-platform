"""Integration tests for RBAC authorization on protected endpoints.

These tests verify that endpoints correctly enforce permission requirements
and return appropriate responses based on user roles.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.rbac import (
    DEFAULT_ROLES,
    Role,
    RolePermission,
    UserRole,
)
from atp.dashboard.tenancy.models import Tenant
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def test_database():
    """Create and configure a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    # Create all tables
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Create additional tables
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
async def async_session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async with test_database.session() as session:
        yield session


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes."""
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


@pytest.fixture
async def admin_user(async_session: AsyncSession) -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=True,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def viewer_user(async_session: AsyncSession) -> User:
    """Create a viewer user for testing."""
    user = User(
        username="viewer_test",
        email="viewer@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=False,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def developer_user(async_session: AsyncSession) -> User:
    """Create a developer user for testing."""
    user = User(
        username="developer_test",
        email="developer@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=False,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def no_role_user(async_session: AsyncSession) -> User:
    """Create a user with no roles for testing."""
    user = User(
        username="norole_test",
        email="norole@test.com",
        hashed_password=get_password_hash("password123"),
        is_admin=False,
        is_active=True,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def viewer_role(async_session: AsyncSession) -> Role:
    """Create a viewer role with permissions."""
    role = Role(
        name="viewer",
        description="Viewer role",
        is_system=True,
        is_active=True,
    )
    async_session.add(role)
    await async_session.flush()

    # Add viewer permissions
    viewer_perms = DEFAULT_ROLES["viewer"].permissions
    for perm in viewer_perms:
        role_perm = RolePermission(role_id=role.id, permission=perm.value)
        async_session.add(role_perm)

    await async_session.commit()
    await async_session.refresh(role)
    return role


@pytest.fixture
async def developer_role(async_session: AsyncSession) -> Role:
    """Create a developer role with permissions."""
    role = Role(
        name="developer",
        description="Developer role",
        is_system=True,
        is_active=True,
    )
    async_session.add(role)
    await async_session.flush()

    # Add developer permissions
    dev_perms = DEFAULT_ROLES["developer"].permissions
    for perm in dev_perms:
        role_perm = RolePermission(role_id=role.id, permission=perm.value)
        async_session.add(role_perm)

    await async_session.commit()
    await async_session.refresh(role)
    return role


@pytest.fixture
async def user_with_viewer_role(
    async_session: AsyncSession, viewer_user: User, viewer_role: Role
) -> User:
    """Create a user with viewer role assigned."""
    user_role = UserRole(user_id=viewer_user.id, role_id=viewer_role.id)
    async_session.add(user_role)
    await async_session.commit()
    return viewer_user


@pytest.fixture
async def user_with_developer_role(
    async_session: AsyncSession, developer_user: User, developer_role: Role
) -> User:
    """Create a user with developer role assigned."""
    user_role = UserRole(user_id=developer_user.id, role_id=developer_role.id)
    async_session.add(user_role)
    await async_session.commit()
    return developer_user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Generate JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.username, "user_id": admin_user.id}
    )


@pytest.fixture
def viewer_token(user_with_viewer_role: User) -> str:
    """Generate JWT token for viewer user."""
    return create_access_token(
        data={
            "sub": user_with_viewer_role.username,
            "user_id": user_with_viewer_role.id,
        }
    )


@pytest.fixture
def developer_token(user_with_developer_role: User) -> str:
    """Generate JWT token for developer user."""
    return create_access_token(
        data={
            "sub": user_with_developer_role.username,
            "user_id": user_with_developer_role.id,
        }
    )


@pytest.fixture
def no_role_token(no_role_user: User) -> str:
    """Generate JWT token for user with no roles."""
    return create_access_token(
        data={"sub": no_role_user.username, "user_id": no_role_user.id}
    )


class TestAgentEndpointAuthorization:
    """Test RBAC authorization on agent endpoints."""

    @pytest.mark.anyio
    async def test_list_agents_requires_auth(self, v2_app):
        """Test listing agents requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/agents")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_list_agents_forbidden_without_permission(
        self, v2_app, no_role_token
    ):
        """Test listing agents forbidden without AGENTS_READ permission."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/agents",
                headers={"Authorization": f"Bearer {no_role_token}"},
            )
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_list_agents_allowed_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer can list agents (has AGENTS_READ)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/agents",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_create_agent_forbidden_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer cannot create agents (no AGENTS_WRITE)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/agents",
                json={"name": "test-agent", "agent_type": "http"},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_create_agent_allowed_for_developer(
        self, v2_app, user_with_developer_role, developer_token
    ):
        """Test developer can create agents (has AGENTS_WRITE)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/agents",
                json={"name": "test-agent", "agent_type": "http"},
                headers={"Authorization": f"Bearer {developer_token}"},
            )
            assert response.status_code == 201

    @pytest.mark.anyio
    async def test_delete_agent_forbidden_for_developer(
        self, v2_app, user_with_developer_role, developer_token, admin_token
    ):
        """Test developer cannot delete agents (no AGENTS_DELETE)."""
        # First create an agent as admin
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            create_response = await client.post(
                "/api/agents",
                json={"name": "to-delete", "agent_type": "http"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            agent_id = create_response.json()["id"]

            # Try to delete as developer
            response = await client.delete(
                f"/api/agents/{agent_id}",
                headers={"Authorization": f"Bearer {developer_token}"},
            )
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_delete_agent_allowed_for_admin(
        self, v2_app, admin_user, admin_token
    ):
        """Test admin can delete agents."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Create an agent
            create_response = await client.post(
                "/api/agents",
                json={"name": "to-delete", "agent_type": "http"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            agent_id = create_response.json()["id"]

            # Delete as admin
            response = await client.delete(
                f"/api/agents/{agent_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 204


class TestSuiteDefinitionAuthorization:
    """Test RBAC authorization on suite definition endpoints."""

    @pytest.mark.anyio
    async def test_list_suite_definitions_requires_auth(self, v2_app):
        """Test listing suite definitions requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/suite-definitions")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_list_suite_definitions_allowed_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer can list suite definitions (has SUITES_READ)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/suite-definitions",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_create_suite_definition_forbidden_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer cannot create suite definitions (no SUITES_WRITE)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/suite-definitions",
                json={
                    "name": "test-suite",
                    "version": "1.0",
                    "defaults": {"runs_per_test": 1},
                    "agents": [],
                    "tests": [],
                },
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 403


class TestAnalyticsAuthorization:
    """Test RBAC authorization on analytics endpoints."""

    @pytest.mark.anyio
    async def test_analytics_trends_requires_auth(self, v2_app):
        """Test analytics trends requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/analytics/trends")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_analytics_trends_allowed_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer can access analytics trends (has ANALYTICS_READ)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/trends",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            # May return 200 or error based on data, but not 401/403
            assert response.status_code != 401
            assert response.status_code != 403


class TestBudgetAuthorization:
    """Test RBAC authorization on budget endpoints."""

    @pytest.mark.anyio
    async def test_list_budgets_requires_auth(self, v2_app):
        """Test listing budgets requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/budgets")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_list_budgets_allowed_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer can list budgets (has BUDGETS_READ)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/budgets",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_create_budget_forbidden_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer cannot create budgets (no BUDGETS_WRITE)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/budgets",
                json={"name": "test-budget", "period": "daily", "limit_usd": 100.0},
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 403


class TestDashboardSummaryAuthorization:
    """Test RBAC authorization on dashboard summary endpoint."""

    @pytest.mark.anyio
    async def test_dashboard_summary_requires_auth(self, v2_app):
        """Test dashboard summary requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/dashboard/summary")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_dashboard_summary_allowed_for_viewer(
        self, v2_app, user_with_viewer_role, viewer_token
    ):
        """Test viewer can access dashboard summary (has SUITES_READ)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/dashboard/summary",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert response.status_code == 200


class TestAdminOnlyEndpoints:
    """Test that admin-only endpoints are properly protected."""

    @pytest.mark.anyio
    async def test_assign_role_requires_admin(
        self, v2_app, user_with_developer_role, developer_token, viewer_role
    ):
        """Test role assignment requires admin privileges."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles/users/assign",
                json={"user_id": 999, "role_id": viewer_role.id},
                headers={"Authorization": f"Bearer {developer_token}"},
            )
            # Should be forbidden for non-admin
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_tenant_list_requires_admin(
        self, v2_app, user_with_developer_role, developer_token
    ):
        """Test tenant management requires admin privileges."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants",
                headers={"Authorization": f"Bearer {developer_token}"},
            )
            # Should be forbidden for non-admin
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_tenant_list_allowed_for_admin(self, v2_app, admin_user, admin_token):
        """Test admin can list tenants."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/tenants",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
