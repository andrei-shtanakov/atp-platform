"""Integration tests for the role management API routes.

These tests verify the /api/v2/roles endpoints for role
CRUD operations and user-role assignments.
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
    Permission,
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
async def regular_user(async_session: AsyncSession) -> User:
    """Create a regular (non-admin) user for testing."""
    user = User(
        username="regular_test",
        email="regular@test.com",
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
def admin_token(admin_user: User) -> str:
    """Generate JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.username, "user_id": admin_user.id}
    )


@pytest.fixture
def regular_token(regular_user: User) -> str:
    """Generate JWT token for regular user."""
    return create_access_token(
        data={"sub": regular_user.username, "user_id": regular_user.id}
    )


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
async def custom_role(async_session: AsyncSession) -> Role:
    """Create a custom (non-system) role."""
    role = Role(
        name="custom-role",
        description="A custom role for testing",
        is_system=False,
        is_active=True,
    )
    async_session.add(role)
    await async_session.flush()

    # Add some permissions
    for perm in [Permission.SUITES_READ, Permission.AGENTS_READ]:
        role_perm = RolePermission(role_id=role.id, permission=perm.value)
        async_session.add(role_perm)

    await async_session.commit()
    await async_session.refresh(role)
    return role


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
def developer_token(user_with_developer_role: User) -> str:
    """Generate JWT token for developer user."""
    return create_access_token(
        data={
            "sub": user_with_developer_role.username,
            "user_id": user_with_developer_role.id,
        }
    )


class TestRoleListEndpoint:
    """Test GET /api/roles endpoint."""

    @pytest.mark.anyio
    async def test_list_roles_requires_auth(self, v2_app):
        """Test listing roles requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/roles")
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_list_roles_requires_permission(
        self, v2_app, regular_user, regular_token
    ):
        """Test listing roles requires ROLES_READ permission."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles",
                headers={"Authorization": f"Bearer {regular_token}"},
            )
            # Regular user without any roles should be forbidden
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_list_roles_as_admin(
        self, v2_app, admin_user, admin_token, developer_role, viewer_role
    ):
        """Test admin can list roles."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) >= 2  # At least developer and viewer roles

    @pytest.mark.anyio
    async def test_list_roles_with_developer_permission(
        self, v2_app, user_with_developer_role, developer_token, developer_role
    ):
        """Test developer can list roles (has ROLES_READ permission)."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles",
                headers={"Authorization": f"Bearer {developer_token}"},
            )
            assert response.status_code == 200


class TestRoleGetEndpoint:
    """Test GET /api/roles/{role_id} endpoint."""

    @pytest.mark.anyio
    async def test_get_role_success(
        self, v2_app, admin_user, admin_token, developer_role
    ):
        """Test getting a role by ID."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/roles/{developer_role.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == developer_role.id
            assert data["name"] == "developer"
            assert data["is_system"] is True
            assert "permissions" in data
            assert len(data["permissions"]) > 0

    @pytest.mark.anyio
    async def test_get_role_not_found(self, v2_app, admin_user, admin_token):
        """Test getting non-existent role returns 404."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles/99999",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 404


class TestRoleCreateEndpoint:
    """Test POST /api/roles endpoint."""

    @pytest.mark.anyio
    async def test_create_role_requires_permission(
        self, v2_app, regular_user, regular_token
    ):
        """Test creating role requires ROLES_WRITE permission."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles",
                json={"name": "new-role"},
                headers={"Authorization": f"Bearer {regular_token}"},
            )
            assert response.status_code == 403

    @pytest.mark.anyio
    async def test_create_role_success(self, v2_app, admin_user, admin_token):
        """Test creating a role successfully."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles",
                json={
                    "name": "tester",
                    "description": "Test runner role",
                    "permissions": ["suites:read", "suites:execute", "results:read"],
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["name"] == "tester"
            assert data["description"] == "Test runner role"
            assert data["is_system"] is False
            assert len(data["permissions"]) == 3

    @pytest.mark.anyio
    async def test_create_role_duplicate_name(
        self, v2_app, admin_user, admin_token, developer_role
    ):
        """Test creating role with duplicate name fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles",
                json={"name": "developer"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400

    @pytest.mark.anyio
    async def test_create_role_invalid_permission(
        self, v2_app, admin_user, admin_token
    ):
        """Test creating role with invalid permission fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles",
                json={
                    "name": "invalid-role",
                    "permissions": ["suites:read", "invalid:permission"],
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400


class TestRoleUpdateEndpoint:
    """Test PATCH /api/roles/{role_id} endpoint."""

    @pytest.mark.anyio
    async def test_update_custom_role(
        self, v2_app, admin_user, admin_token, custom_role
    ):
        """Test updating a custom role."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.patch(
                f"/api/roles/{custom_role.id}",
                json={
                    "name": "updated-custom-role",
                    "description": "Updated description",
                    "permissions": ["suites:read", "agents:read", "results:read"],
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "updated-custom-role"
            assert data["description"] == "Updated description"
            assert len(data["permissions"]) == 3

    @pytest.mark.anyio
    async def test_cannot_update_system_role_name(
        self, v2_app, admin_user, admin_token, developer_role
    ):
        """Test cannot change name of system role."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.patch(
                f"/api/roles/{developer_role.id}",
                json={"name": "renamed-developer"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400

    @pytest.mark.anyio
    async def test_cannot_update_system_role_permissions(
        self, v2_app, admin_user, admin_token, developer_role
    ):
        """Test cannot change permissions of system role."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.patch(
                f"/api/roles/{developer_role.id}",
                json={"permissions": ["suites:read"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400

    @pytest.mark.anyio
    async def test_can_update_system_role_description(
        self, v2_app, admin_user, admin_token, developer_role
    ):
        """Test can update description of system role."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.patch(
                f"/api/roles/{developer_role.id}",
                json={"description": "Updated developer description"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["description"] == "Updated developer description"


class TestRoleDeleteEndpoint:
    """Test DELETE /api/roles/{role_id} endpoint."""

    @pytest.mark.anyio
    async def test_delete_custom_role(
        self, v2_app, admin_user, admin_token, custom_role
    ):
        """Test deleting a custom role."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                f"/api/roles/{custom_role.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 204

    @pytest.mark.anyio
    async def test_cannot_delete_system_role(
        self, v2_app, admin_user, admin_token, developer_role
    ):
        """Test cannot delete system role."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                f"/api/roles/{developer_role.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400


class TestUserRoleAssignmentEndpoints:
    """Test user-role assignment endpoints."""

    @pytest.mark.anyio
    async def test_get_user_roles(
        self, v2_app, admin_user, admin_token, user_with_developer_role, developer_role
    ):
        """Test getting roles assigned to a user."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/roles/users/{user_with_developer_role.id}/roles",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["role_id"] == developer_role.id
            assert data[0]["role_name"] == "developer"

    @pytest.mark.anyio
    async def test_assign_role_to_user(
        self, v2_app, admin_user, admin_token, regular_user, viewer_role
    ):
        """Test assigning a role to a user."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles/users/assign",
                json={
                    "user_id": regular_user.id,
                    "role_id": viewer_role.id,
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["user_id"] == regular_user.id
            assert data["role_id"] == viewer_role.id
            assert data["role_name"] == "viewer"
            assert data["assigned_by_id"] == admin_user.id

    @pytest.mark.anyio
    async def test_assign_duplicate_role_fails(
        self, v2_app, admin_user, admin_token, user_with_developer_role, developer_role
    ):
        """Test assigning same role twice fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/roles/users/assign",
                json={
                    "user_id": user_with_developer_role.id,
                    "role_id": developer_role.id,
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 400

    @pytest.mark.anyio
    async def test_remove_role_from_user(
        self, v2_app, admin_user, admin_token, user_with_developer_role, developer_role
    ):
        """Test removing a role from a user."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                f"/api/roles/users/{user_with_developer_role.id}/roles/{developer_role.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 204

    @pytest.mark.anyio
    async def test_remove_unassigned_role_fails(
        self, v2_app, admin_user, admin_token, regular_user, viewer_role
    ):
        """Test removing unassigned role fails."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete(
                f"/api/roles/users/{regular_user.id}/roles/{viewer_role.id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 404


class TestUserPermissionsEndpoints:
    """Test user permissions endpoints."""

    @pytest.mark.anyio
    async def test_get_user_permissions(
        self, v2_app, admin_user, admin_token, user_with_developer_role
    ):
        """Test getting user's effective permissions."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/roles/users/{user_with_developer_role.id}/permissions",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == user_with_developer_role.id
            assert data["is_admin"] is False
            assert len(data["roles"]) == 1
            assert "suites:read" in data["permissions"]
            assert "suites:write" in data["permissions"]

    @pytest.mark.anyio
    async def test_get_my_permissions(
        self, v2_app, user_with_developer_role, developer_token
    ):
        """Test getting current user's permissions."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles/me/permissions",
                headers={"Authorization": f"Bearer {developer_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == user_with_developer_role.id
            assert data["username"] == user_with_developer_role.username
            assert len(data["permissions"]) > 0


class TestDefaultRolesEndpoints:
    """Test default roles and permissions endpoints."""

    @pytest.mark.anyio
    async def test_get_default_roles(self, v2_app, admin_user, admin_token):
        """Test getting default role definitions."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles/defaults",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "admin" in data
            assert "developer" in data
            assert "analyst" in data
            assert "viewer" in data
            # Admin should have all permissions
            assert len(data["admin"]) > len(data["developer"])
            # Viewer should have fewer permissions than analyst
            assert len(data["viewer"]) < len(data["analyst"])

    @pytest.mark.anyio
    async def test_list_permissions(self, v2_app, admin_user, admin_token):
        """Test listing all available permissions."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/roles/permissions",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert "suites:read" in data
            assert "agents:write" in data
            assert "users:delete" in data
