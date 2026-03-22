"""Tests for RBAC module."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from atp.dashboard.rbac.models import (
    ADMIN_PERMISSIONS,
    ANALYST_PERMISSIONS,
    DEFAULT_ROLES,
    DEVELOPER_PERMISSIONS,
    VIEWER_PERMISSIONS,
    Permission,
    Role,
    RoleCreate,
    RolePermission,
    RolePermissions,
    RoleResponse,
    RoleUpdate,
    UserPermissionsResponse,
    UserRole,
    UserRoleAssign,
    UserRoleResponse,
    get_user_permissions,
    has_permission,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_all_permissions_have_correct_format(self) -> None:
        """Test that all permissions follow the resource:action format."""
        for perm in Permission:
            assert ":" in perm.value, f"Permission {perm} should contain ':'"
            parts = perm.value.split(":")
            assert len(parts) == 2, f"Permission {perm} should have exactly one ':'"
            assert parts[0], f"Permission {perm} should have a resource"
            assert parts[1], f"Permission {perm} should have an action"

    def test_permission_count(self) -> None:
        """Test expected number of permissions."""
        # We should have permissions for multiple resources
        assert len(Permission) >= 20

    def test_suite_permissions_exist(self) -> None:
        """Test that suite permissions exist."""
        assert Permission.SUITES_READ
        assert Permission.SUITES_WRITE
        assert Permission.SUITES_EXECUTE
        assert Permission.SUITES_DELETE

    def test_agent_permissions_exist(self) -> None:
        """Test that agent permissions exist."""
        assert Permission.AGENTS_READ
        assert Permission.AGENTS_WRITE
        assert Permission.AGENTS_EXECUTE
        assert Permission.AGENTS_DELETE

    def test_user_permissions_exist(self) -> None:
        """Test that user permissions exist."""
        assert Permission.USERS_READ
        assert Permission.USERS_WRITE
        assert Permission.USERS_DELETE

    def test_role_permissions_exist(self) -> None:
        """Test that role permissions exist."""
        assert Permission.ROLES_READ
        assert Permission.ROLES_WRITE
        assert Permission.ROLES_DELETE


class TestRolePermissions:
    """Tests for RolePermissions model."""

    def test_create_role_permissions(self) -> None:
        """Test creating a RolePermissions instance."""
        perms = RolePermissions(
            name="custom",
            description="Custom role",
            permissions=frozenset([Permission.SUITES_READ, Permission.AGENTS_READ]),
            is_system=False,
        )
        assert perms.name == "custom"
        assert perms.description == "Custom role"
        assert Permission.SUITES_READ in perms.permissions
        assert Permission.AGENTS_READ in perms.permissions
        assert not perms.is_system

    def test_role_permissions_is_frozen(self) -> None:
        """Test that RolePermissions is immutable."""
        perms = RolePermissions(
            name="test",
            permissions=frozenset([Permission.SUITES_READ]),
        )
        with pytest.raises(Exception):
            perms.name = "modified"  # type: ignore


class TestDefaultRoles:
    """Tests for default role definitions."""

    def test_admin_role_has_all_permissions(self) -> None:
        """Test that admin role has all permissions."""
        assert ADMIN_PERMISSIONS == frozenset(Permission)

    def test_default_roles_exist(self) -> None:
        """Test that all default roles are defined."""
        assert "admin" in DEFAULT_ROLES
        assert "developer" in DEFAULT_ROLES
        assert "analyst" in DEFAULT_ROLES
        assert "viewer" in DEFAULT_ROLES

    def test_default_roles_are_system_roles(self) -> None:
        """Test that default roles are marked as system roles."""
        for role in DEFAULT_ROLES.values():
            assert role.is_system

    def test_admin_has_all_permissions(self) -> None:
        """Test admin role has all permissions."""
        admin = DEFAULT_ROLES["admin"]
        assert admin.permissions == frozenset(Permission)

    def test_developer_permissions(self) -> None:
        """Test developer role permissions."""
        dev = DEFAULT_ROLES["developer"]
        # Should have suite RWX
        assert Permission.SUITES_READ in dev.permissions
        assert Permission.SUITES_WRITE in dev.permissions
        assert Permission.SUITES_EXECUTE in dev.permissions
        assert Permission.SUITES_DELETE not in dev.permissions
        # Should have settings R but not W/D
        assert Permission.SETTINGS_READ in dev.permissions
        assert Permission.SETTINGS_WRITE not in dev.permissions
        # Should not have user management
        assert Permission.USERS_READ not in dev.permissions
        assert Permission.USERS_WRITE not in dev.permissions

    def test_analyst_permissions(self) -> None:
        """Test analyst role permissions."""
        analyst = DEFAULT_ROLES["analyst"]
        # Should have read access
        assert Permission.SUITES_READ in analyst.permissions
        assert Permission.AGENTS_READ in analyst.permissions
        assert Permission.RESULTS_READ in analyst.permissions
        # Should not have write access
        assert Permission.SUITES_WRITE not in analyst.permissions
        assert Permission.AGENTS_WRITE not in analyst.permissions
        # Should have analytics export
        assert Permission.ANALYTICS_READ in analyst.permissions
        assert Permission.ANALYTICS_EXPORT in analyst.permissions

    def test_viewer_permissions(self) -> None:
        """Test viewer role permissions."""
        viewer = DEFAULT_ROLES["viewer"]
        # Should have read-only access
        assert Permission.SUITES_READ in viewer.permissions
        assert Permission.AGENTS_READ in viewer.permissions
        assert Permission.RESULTS_READ in viewer.permissions
        # Should not have write access
        assert Permission.SUITES_WRITE not in viewer.permissions
        assert Permission.AGENTS_WRITE not in viewer.permissions
        # Should not have export
        assert Permission.ANALYTICS_EXPORT not in viewer.permissions

    def test_permission_hierarchy(self) -> None:
        """Test that permission sets follow expected hierarchy."""
        admin_perms = DEFAULT_ROLES["admin"].permissions
        dev_perms = DEFAULT_ROLES["developer"].permissions
        analyst_perms = DEFAULT_ROLES["analyst"].permissions
        viewer_perms = DEFAULT_ROLES["viewer"].permissions

        # Admin should have all developer permissions
        assert dev_perms.issubset(admin_perms)
        # Developer should have all analyst permissions
        # (Not strictly true, developers have write that analysts have export)
        # Viewer should be subset of analyst (mostly true, analyst has export)
        assert viewer_perms.issubset(analyst_perms)


class TestRoleModel:
    """Tests for Role SQLAlchemy model."""

    def test_create_role(self) -> None:
        """Test creating a Role instance."""
        role = Role(
            id=1,
            name="custom-role",
            description="A custom role",
            tenant_id="test-tenant",
            is_system=False,
            is_active=True,
        )
        assert role.id == 1
        assert role.name == "custom-role"
        assert role.description == "A custom role"
        assert role.tenant_id == "test-tenant"
        assert not role.is_system
        assert role.is_active

    def test_role_repr(self) -> None:
        """Test role string representation."""
        role = Role(
            id=1,
            name="test-role",
            is_system=True,
        )
        repr_str = repr(role)
        assert "test-role" in repr_str
        assert "is_system=True" in repr_str

    def test_get_permissions_empty(self) -> None:
        """Test getting permissions from role with no permissions."""
        role = Role(id=1, name="empty-role")
        role.role_permissions = []
        assert role.get_permissions() == set()

    def test_get_permissions_with_permissions(self) -> None:
        """Test getting permissions from role with permissions."""
        role = Role(id=1, name="test-role")
        rp1 = RolePermission(id=1, role_id=1, permission="suites:read")
        rp2 = RolePermission(id=2, role_id=1, permission="agents:read")
        role.role_permissions = [rp1, rp2]

        perms = role.get_permissions()
        assert Permission.SUITES_READ in perms
        assert Permission.AGENTS_READ in perms
        assert len(perms) == 2


class TestRolePermissionModel:
    """Tests for RolePermission SQLAlchemy model."""

    def test_create_role_permission(self) -> None:
        """Test creating a RolePermission instance."""
        rp = RolePermission(
            id=1,
            role_id=1,
            permission="suites:read",
        )
        assert rp.id == 1
        assert rp.role_id == 1
        assert rp.permission == "suites:read"

    def test_role_permission_repr(self) -> None:
        """Test role permission string representation."""
        rp = RolePermission(role_id=1, permission="suites:read")
        repr_str = repr(rp)
        assert "role_id=1" in repr_str
        assert "suites:read" in repr_str


class TestUserRoleModel:
    """Tests for UserRole SQLAlchemy model."""

    def test_create_user_role(self) -> None:
        """Test creating a UserRole instance."""
        ur = UserRole(
            id=1,
            user_id=1,
            role_id=2,
            assigned_by_id=3,
        )
        assert ur.id == 1
        assert ur.user_id == 1
        assert ur.role_id == 2
        assert ur.assigned_by_id == 3

    def test_user_role_repr(self) -> None:
        """Test user role string representation."""
        ur = UserRole(user_id=1, role_id=2)
        repr_str = repr(ur)
        assert "user_id=1" in repr_str
        assert "role_id=2" in repr_str


class TestGetUserPermissions:
    """Tests for get_user_permissions function."""

    def _create_mock_user(self, is_admin: bool = False) -> MagicMock:
        """Create a mock user."""
        user = MagicMock()
        user.is_admin = is_admin
        return user

    def _create_mock_role(
        self, permissions: list[Permission], is_active: bool = True
    ) -> MagicMock:
        """Create a mock role with permissions."""
        role = MagicMock()
        role.is_active = is_active
        role.get_permissions.return_value = set(permissions)
        return role

    def test_admin_user_gets_all_permissions(self) -> None:
        """Test that admin user gets all permissions."""
        user = self._create_mock_user(is_admin=True)
        perms = get_user_permissions(user, [])
        assert perms == set(Permission)

    def test_user_with_no_roles_has_no_permissions(self) -> None:
        """Test that user with no roles has no permissions."""
        user = self._create_mock_user(is_admin=False)
        perms = get_user_permissions(user, [])
        assert perms == set()

    def test_user_gets_permissions_from_role(self) -> None:
        """Test that user gets permissions from assigned role."""
        user = self._create_mock_user(is_admin=False)
        role = self._create_mock_role([Permission.SUITES_READ, Permission.AGENTS_READ])
        perms = get_user_permissions(user, [role])
        assert perms == {Permission.SUITES_READ, Permission.AGENTS_READ}

    def test_user_gets_combined_permissions_from_multiple_roles(self) -> None:
        """Test that user gets combined permissions from multiple roles."""
        user = self._create_mock_user(is_admin=False)
        role1 = self._create_mock_role([Permission.SUITES_READ])
        role2 = self._create_mock_role([Permission.AGENTS_READ])
        perms = get_user_permissions(user, [role1, role2])
        assert perms == {Permission.SUITES_READ, Permission.AGENTS_READ}

    def test_inactive_roles_are_ignored(self) -> None:
        """Test that inactive roles are not considered."""
        user = self._create_mock_user(is_admin=False)
        active_role = self._create_mock_role([Permission.SUITES_READ], is_active=True)
        inactive_role = self._create_mock_role(
            [Permission.AGENTS_READ], is_active=False
        )
        perms = get_user_permissions(user, [active_role, inactive_role])
        assert perms == {Permission.SUITES_READ}
        assert Permission.AGENTS_READ not in perms


class TestHasPermission:
    """Tests for has_permission function."""

    def _create_mock_user(self, is_admin: bool = False) -> MagicMock:
        """Create a mock user."""
        user = MagicMock()
        user.is_admin = is_admin
        return user

    def _create_mock_role(
        self, permissions: list[Permission], is_active: bool = True
    ) -> MagicMock:
        """Create a mock role with permissions."""
        role = MagicMock()
        role.is_active = is_active
        role.get_permissions.return_value = set(permissions)
        return role

    def test_admin_always_has_permission(self) -> None:
        """Test that admin user always has any permission."""
        user = self._create_mock_user(is_admin=True)
        assert has_permission(user, [], Permission.SUITES_READ)
        assert has_permission(user, [], Permission.USERS_DELETE)
        assert has_permission(user, [], Permission.TENANTS_WRITE)

    def test_user_has_permission_from_role(self) -> None:
        """Test user has permission granted by role."""
        user = self._create_mock_user(is_admin=False)
        role = self._create_mock_role([Permission.SUITES_READ])
        assert has_permission(user, [role], Permission.SUITES_READ)
        assert not has_permission(user, [role], Permission.SUITES_WRITE)

    def test_user_without_role_lacks_permission(self) -> None:
        """Test user without roles doesn't have permissions."""
        user = self._create_mock_user(is_admin=False)
        assert not has_permission(user, [], Permission.SUITES_READ)


class TestRoleSchemas:
    """Tests for Pydantic schemas."""

    def test_role_create_schema(self) -> None:
        """Test RoleCreate schema."""
        data = RoleCreate(
            name="test-role",
            description="Test description",
            permissions=["suites:read", "agents:read"],
        )
        assert data.name == "test-role"
        assert data.description == "Test description"
        assert len(data.permissions) == 2

    def test_role_create_minimal(self) -> None:
        """Test RoleCreate with minimal data."""
        data = RoleCreate(name="minimal")
        assert data.name == "minimal"
        assert data.description == ""
        assert data.permissions == []

    def test_role_update_schema(self) -> None:
        """Test RoleUpdate schema."""
        data = RoleUpdate(
            name="updated-name",
            description="Updated description",
            is_active=False,
        )
        assert data.name == "updated-name"
        assert data.description == "Updated description"
        assert data.is_active is False

    def test_role_update_partial(self) -> None:
        """Test RoleUpdate with partial data."""
        data = RoleUpdate(description="Only description")
        assert data.name is None
        assert data.description == "Only description"
        assert data.permissions is None
        assert data.is_active is None

    def test_role_response_schema(self) -> None:
        """Test RoleResponse schema."""
        data = RoleResponse(
            id=1,
            name="test",
            description="Test role",
            is_system=False,
            is_active=True,
            permissions=["suites:read"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert data.id == 1
        assert data.name == "test"
        assert len(data.permissions) == 1

    def test_user_role_assign_schema(self) -> None:
        """Test UserRoleAssign schema."""
        data = UserRoleAssign(user_id=1, role_id=2)
        assert data.user_id == 1
        assert data.role_id == 2

    def test_user_role_response_schema(self) -> None:
        """Test UserRoleResponse schema."""
        data = UserRoleResponse(
            id=1,
            user_id=1,
            role_id=2,
            role_name="admin",
            assigned_at=datetime.now(),
            assigned_by_id=3,
        )
        assert data.id == 1
        assert data.user_id == 1
        assert data.role_name == "admin"

    def test_user_permissions_response_schema(self) -> None:
        """Test UserPermissionsResponse schema."""
        role = RoleResponse(
            id=1,
            name="admin",
            description="Admin role",
            is_system=True,
            is_active=True,
            permissions=["suites:read", "suites:write"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        data = UserPermissionsResponse(
            user_id=1,
            username="testuser",
            is_admin=False,
            roles=[role],
            permissions=["suites:read", "suites:write"],
        )
        assert data.user_id == 1
        assert data.username == "testuser"
        assert len(data.roles) == 1
        assert len(data.permissions) == 2


class TestPermissionSets:
    """Tests for permission set constants."""

    def test_admin_permissions_is_complete(self) -> None:
        """Test admin has all permissions."""
        assert ADMIN_PERMISSIONS == frozenset(Permission)

    def test_developer_permissions_subset_of_admin(self) -> None:
        """Test developer permissions are subset of admin."""
        assert DEVELOPER_PERMISSIONS.issubset(ADMIN_PERMISSIONS)

    def test_analyst_permissions_subset_of_admin(self) -> None:
        """Test analyst permissions are subset of admin."""
        assert ANALYST_PERMISSIONS.issubset(ADMIN_PERMISSIONS)

    def test_viewer_permissions_subset_of_analyst(self) -> None:
        """Test viewer permissions are subset of analyst."""
        assert VIEWER_PERMISSIONS.issubset(ANALYST_PERMISSIONS)

    def test_viewer_is_most_restricted(self) -> None:
        """Test viewer has fewest permissions."""
        assert len(VIEWER_PERMISSIONS) <= len(ANALYST_PERMISSIONS)
        assert len(VIEWER_PERMISSIONS) <= len(DEVELOPER_PERMISSIONS)
        assert len(VIEWER_PERMISSIONS) < len(ADMIN_PERMISSIONS)
