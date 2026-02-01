"""Role-Based Access Control (RBAC) module for ATP Dashboard.

This module provides fine-grained access control with roles and permissions
for managing access to resources in the ATP Dashboard.

Permissions Matrix:
    Resource        | Admin | Developer | Analyst | Viewer
    ----------------|-------|-----------|---------|--------
    Suites          | RWXD  | RWX       | R       | R
    Agents          | RWXD  | RWX       | R       | R
    Results         | RWXD  | RW        | R       | R
    Baselines       | RWXD  | RW        | R       | R
    Settings        | RWXD  | R         | -       | -
    Users           | RWXD  | -         | -       | -

Legend: R=Read, W=Write, X=Execute, D=Delete
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import Base, User


class Permission(str, Enum):
    """Enumeration of all available permissions.

    Permissions follow the format: RESOURCE_ACTION where:
    - RESOURCE: The type of resource being accessed
    - ACTION: The action being performed (read, write, execute, delete)
    """

    # Suite permissions
    SUITES_READ = "suites:read"
    SUITES_WRITE = "suites:write"
    SUITES_EXECUTE = "suites:execute"
    SUITES_DELETE = "suites:delete"

    # Agent permissions
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    AGENTS_EXECUTE = "agents:execute"
    AGENTS_DELETE = "agents:delete"

    # Results permissions
    RESULTS_READ = "results:read"
    RESULTS_WRITE = "results:write"
    RESULTS_DELETE = "results:delete"

    # Baseline permissions
    BASELINES_READ = "baselines:read"
    BASELINES_WRITE = "baselines:write"
    BASELINES_DELETE = "baselines:delete"

    # Settings permissions
    SETTINGS_READ = "settings:read"
    SETTINGS_WRITE = "settings:write"
    SETTINGS_DELETE = "settings:delete"

    # User management permissions
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"

    # Role management permissions
    ROLES_READ = "roles:read"
    ROLES_WRITE = "roles:write"
    ROLES_DELETE = "roles:delete"

    # Budget permissions
    BUDGETS_READ = "budgets:read"
    BUDGETS_WRITE = "budgets:write"
    BUDGETS_DELETE = "budgets:delete"

    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_WRITE = "analytics:write"
    ANALYTICS_DELETE = "analytics:delete"
    ANALYTICS_EXPORT = "analytics:export"

    # Tenant management (admin only)
    TENANTS_READ = "tenants:read"
    TENANTS_WRITE = "tenants:write"
    TENANTS_DELETE = "tenants:delete"

    # Audit log permissions (admin only by default)
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"
    AUDIT_MANAGE = "audit:manage"  # For retention policy management


class RolePermissions(BaseModel):
    """Permissions configuration for a role.

    This model defines what permissions a role grants to its members.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=50, description="Role name")
    description: str = Field(default="", max_length=255, description="Role description")
    permissions: frozenset[Permission] = Field(
        default_factory=frozenset, description="Set of permissions granted by this role"
    )
    is_system: bool = Field(
        default=False, description="Whether this is a system-defined role"
    )


# Define default roles with their permissions
ADMIN_PERMISSIONS: frozenset[Permission] = frozenset(Permission)  # All permissions

DEVELOPER_PERMISSIONS: frozenset[Permission] = frozenset(
    [
        # Suites: RWX
        Permission.SUITES_READ,
        Permission.SUITES_WRITE,
        Permission.SUITES_EXECUTE,
        # Agents: RWX
        Permission.AGENTS_READ,
        Permission.AGENTS_WRITE,
        Permission.AGENTS_EXECUTE,
        # Results: RW
        Permission.RESULTS_READ,
        Permission.RESULTS_WRITE,
        # Baselines: RW
        Permission.BASELINES_READ,
        Permission.BASELINES_WRITE,
        # Settings: R
        Permission.SETTINGS_READ,
        # Budgets: RW
        Permission.BUDGETS_READ,
        Permission.BUDGETS_WRITE,
        # Analytics: RW + export
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_WRITE,
        Permission.ANALYTICS_DELETE,
        Permission.ANALYTICS_EXPORT,
        # Roles: R (can see roles but not modify)
        Permission.ROLES_READ,
    ]
)

ANALYST_PERMISSIONS: frozenset[Permission] = frozenset(
    [
        # Suites: R
        Permission.SUITES_READ,
        # Agents: R
        Permission.AGENTS_READ,
        # Results: R
        Permission.RESULTS_READ,
        # Baselines: R
        Permission.BASELINES_READ,
        # Budgets: R
        Permission.BUDGETS_READ,
        # Analytics: RW + export
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_WRITE,
        Permission.ANALYTICS_EXPORT,
        # Roles: R
        Permission.ROLES_READ,
    ]
)

VIEWER_PERMISSIONS: frozenset[Permission] = frozenset(
    [
        # Suites: R
        Permission.SUITES_READ,
        # Agents: R
        Permission.AGENTS_READ,
        # Results: R
        Permission.RESULTS_READ,
        # Baselines: R
        Permission.BASELINES_READ,
        # Budgets: R
        Permission.BUDGETS_READ,
        # Analytics: R
        Permission.ANALYTICS_READ,
        # Roles: R
        Permission.ROLES_READ,
    ]
)

# Default role definitions
DEFAULT_ROLES: dict[str, RolePermissions] = {
    "admin": RolePermissions(
        name="admin",
        description="Full administrative access to all resources",
        permissions=ADMIN_PERMISSIONS,
        is_system=True,
    ),
    "developer": RolePermissions(
        name="developer",
        description="Create and execute test suites and manage agents",
        permissions=DEVELOPER_PERMISSIONS,
        is_system=True,
    ),
    "analyst": RolePermissions(
        name="analyst",
        description="View results and analytics, export data",
        permissions=ANALYST_PERMISSIONS,
        is_system=True,
    ),
    "viewer": RolePermissions(
        name="viewer",
        description="Read-only access to view test results",
        permissions=VIEWER_PERMISSIONS,
        is_system=True,
    ),
}


class Role(Base):
    """Role model for RBAC.

    Roles define a set of permissions that can be assigned to users.
    System roles (admin, developer, analyst, viewer) are predefined and
    cannot be deleted.
    """

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="default", index=True
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    is_system: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Relationships
    user_roles: Mapped[list["UserRole"]] = relationship(
        back_populates="role", cascade="all, delete-orphan"
    )
    role_permissions: Mapped[list["RolePermission"]] = relationship(
        back_populates="role", cascade="all, delete-orphan"
    )

    # Unique constraint: role name unique within tenant
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_role_tenant_name"),
        Index("idx_role_tenant", "tenant_id"),
        Index("idx_role_name", "name"),
    )

    def __repr__(self) -> str:
        return f"Role(id={self.id}, name={self.name!r}, is_system={self.is_system})"

    def get_permissions(self) -> set[Permission]:
        """Get all permissions for this role.

        Returns:
            Set of Permission enums granted by this role.
        """
        return {
            Permission(rp.permission)
            for rp in self.role_permissions
            if rp.permission in [p.value for p in Permission]
        }


class RolePermission(Base):
    """Association table for Role-Permission relationship.

    Stores the permissions associated with each role.
    """

    __tablename__ = "role_permissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False
    )
    permission: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationships
    role: Mapped["Role"] = relationship(back_populates="role_permissions")

    # Unique constraint: each permission only once per role
    __table_args__ = (
        UniqueConstraint("role_id", "permission", name="uq_role_permission"),
        Index("idx_role_permission_role", "role_id"),
    )

    def __repr__(self) -> str:
        return f"RolePermission(role_id={self.role_id}, permission={self.permission!r})"


class UserRole(Base):
    """Association table for User-Role relationship.

    Links users to their assigned roles within a tenant.
    """

    __tablename__ = "user_roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    assigned_by_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    role: Mapped["Role"] = relationship(back_populates="user_roles")
    assigned_by: Mapped["User | None"] = relationship(foreign_keys=[assigned_by_id])

    # Unique constraint: each user can only have each role once
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_user_role"),
        Index("idx_user_role_user", "user_id"),
        Index("idx_user_role_role", "role_id"),
    )

    def __repr__(self) -> str:
        return f"UserRole(user_id={self.user_id}, role_id={self.role_id})"


def get_user_permissions(user: User, roles: list[Role]) -> set[Permission]:
    """Get all permissions for a user based on their roles.

    Args:
        user: The user to get permissions for.
        roles: List of roles assigned to the user.

    Returns:
        Set of all permissions the user has from their roles.
    """
    permissions: set[Permission] = set()

    # If user is admin (legacy field), grant all permissions
    if user.is_admin:
        return set(Permission)

    # Collect permissions from all roles
    for role in roles:
        if role.is_active:
            permissions.update(role.get_permissions())

    return permissions


def has_permission(
    user: User,
    roles: list[Role],
    permission: Permission,
) -> bool:
    """Check if a user has a specific permission.

    Args:
        user: The user to check.
        roles: List of roles assigned to the user.
        permission: The permission to check for.

    Returns:
        True if the user has the permission, False otherwise.
    """
    # Admin users always have all permissions
    if user.is_admin:
        return True

    user_permissions = get_user_permissions(user, roles)
    return permission in user_permissions


def require_permission(
    permission: Permission,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory to require a specific permission for an endpoint.

    This creates a FastAPI dependency that checks if the current user
    has the specified permission. If not, raises 403 Forbidden.

    Args:
        permission: The permission required to access the endpoint.

    Returns:
        A dependency function that can be used with FastAPI's Depends.

    Example:
        @router.post("/suites")
        async def create_suite(
            _: Annotated[None, Depends(require_permission(Permission.SUITES_WRITE))],
            ...
        ):
            ...
    """

    def dependency_factory() -> Callable[..., Any]:
        """Create the actual dependency function.

        This is necessary to allow the decorator to work with FastAPI's
        dependency injection system.
        """
        from atp.dashboard.auth import get_current_active_user
        from atp.dashboard.database import get_database

        async def check_permission(
            current_user: Annotated[User, Depends(get_current_active_user)],
        ) -> None:
            """Check if the current user has the required permission.

            Args:
                current_user: The authenticated user.

            Raises:
                HTTPException: If user doesn't have the required permission.
            """
            # Admin users bypass permission checks
            if current_user.is_admin:
                return

            # Get user's roles from database
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            db = get_database()
            async with db.session() as session:
                stmt = (
                    select(Role)
                    .options(selectinload(Role.role_permissions))
                    .join(UserRole, UserRole.role_id == Role.id)
                    .where(UserRole.user_id == current_user.id)
                    .where(Role.is_active.is_(True))
                )
                result = await session.execute(stmt)
                roles = list(result.scalars().all())

                if not has_permission(current_user, roles, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied: {permission.value} required",
                    )

        return check_permission

    return dependency_factory()


# Pydantic schemas for API responses
class RoleResponse(BaseModel):
    """Schema for role response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    is_system: bool
    is_active: bool
    permissions: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class RoleCreate(BaseModel):
    """Schema for creating a role."""

    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(default="", max_length=255)
    permissions: list[str] = Field(default_factory=list)


class RoleUpdate(BaseModel):
    """Schema for updating a role."""

    name: str | None = Field(None, min_length=1, max_length=50)
    description: str | None = Field(None, max_length=255)
    permissions: list[str] | None = None
    is_active: bool | None = None


class UserRoleResponse(BaseModel):
    """Schema for user role response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    role_id: int
    role_name: str = ""
    assigned_at: datetime
    assigned_by_id: int | None = None


class UserRoleAssign(BaseModel):
    """Schema for assigning a role to a user."""

    user_id: int
    role_id: int


class UserRoleRemove(BaseModel):
    """Schema for removing a role from a user."""

    user_id: int
    role_id: int


class UserPermissionsResponse(BaseModel):
    """Schema for user permissions response."""

    user_id: int
    username: str
    is_admin: bool
    roles: list[RoleResponse]
    permissions: list[str]
