"""RBAC (Role-Based Access Control) module for ATP Dashboard.

This package provides role-based access control functionality
for the ATP Dashboard.
"""

from atp.dashboard.rbac.models import (
    DEFAULT_ROLES,
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
    require_permission,
)

__all__ = [
    "DEFAULT_ROLES",
    "Permission",
    "Role",
    "RoleCreate",
    "RolePermission",
    "RolePermissions",
    "RoleResponse",
    "RoleUpdate",
    "UserPermissionsResponse",
    "UserRole",
    "UserRoleAssign",
    "UserRoleResponse",
    "get_user_permissions",
    "has_permission",
    "require_permission",
]
