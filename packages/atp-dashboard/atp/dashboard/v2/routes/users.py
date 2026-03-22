"""User management routes for RBAC.

This module provides API endpoints for managing users and their
role assignments in the ATP Dashboard.

Permissions:
    - GET /users: USERS_READ
    - GET /users/{id}: USERS_READ
    - GET /users/{id}/roles: USERS_READ
    - POST /users/{id}/roles: USERS_WRITE (assign role to user)
    - DELETE /users/{id}/roles/{role_id}: USERS_WRITE (remove role from user)
    - GET /users/{id}/permissions: USERS_READ
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import User
from atp.dashboard.rbac import (
    Permission,
    Role,
    RoleResponse,
    UserPermissionsResponse,
    UserRole,
    UserRoleResponse,
    get_user_permissions,
    require_permission,
)
from atp.dashboard.v2.dependencies import AdminUser, DBSession, Pagination

router = APIRouter(prefix="/users", tags=["users"])


class UserResponse(BaseModel):
    """Schema for user response."""

    id: int
    username: str
    email: str
    is_admin: bool
    is_active: bool


class UserListResponse(BaseModel):
    """Schema for paginated user list response."""

    total: int
    items: list[UserResponse]
    limit: int
    offset: int


class RoleAssignment(BaseModel):
    """Schema for assigning a role to a user."""

    role_id: int


def _role_to_response(role: Role) -> RoleResponse:
    """Convert a Role model to a RoleResponse.

    Args:
        role: The Role model.

    Returns:
        RoleResponse with permissions list.
    """
    return RoleResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        is_system=role.is_system,
        is_active=role.is_active,
        permissions=[rp.permission for rp in role.role_permissions],
        created_at=role.created_at,
        updated_at=role.updated_at,
    )


@router.get("", response_model=UserListResponse)
async def list_users(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.USERS_READ))],
    pagination: Pagination,
    include_inactive: bool = Query(False, description="Include inactive users"),
) -> UserListResponse:
    """List all users.

    Requires USERS_READ permission.

    Args:
        session: Database session.
        pagination: Pagination parameters.
        include_inactive: Whether to include inactive users.

    Returns:
        Paginated list of users.
    """
    # Build query
    stmt = select(User)
    count_stmt = select(func.count(User.id))

    if not include_inactive:
        stmt = stmt.where(User.is_active.is_(True))
        count_stmt = count_stmt.where(User.is_active.is_(True))

    # Get total count
    total = (await session.execute(count_stmt)).scalar() or 0

    # Get paginated results
    stmt = stmt.offset(pagination.offset).limit(pagination.limit)
    result = await session.execute(stmt)
    users = result.scalars().all()

    return UserListResponse(
        total=total,
        items=[
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                is_admin=user.is_admin,
                is_active=user.is_active,
            )
            for user in users
        ],
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.USERS_READ))],
    user_id: int,
) -> UserResponse:
    """Get a user by ID.

    Requires USERS_READ permission.

    Args:
        session: Database session.
        user_id: The user ID.

    Returns:
        The user details.

    Raises:
        HTTPException: If user not found.
    """
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        is_active=user.is_active,
    )


@router.get("/{user_id}/roles", response_model=list[UserRoleResponse])
async def get_user_roles(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.USERS_READ))],
    user_id: int,
) -> list[UserRoleResponse]:
    """Get all roles assigned to a user.

    Requires USERS_READ permission.

    Args:
        session: Database session.
        user_id: The user ID.

    Returns:
        List of user role assignments.

    Raises:
        HTTPException: If user not found.
    """
    # Verify user exists
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    stmt = (
        select(UserRole)
        .options(selectinload(UserRole.role))
        .where(UserRole.user_id == user_id)
    )
    result = await session.execute(stmt)
    user_roles = result.scalars().all()

    return [
        UserRoleResponse(
            id=ur.id,
            user_id=ur.user_id,
            role_id=ur.role_id,
            role_name=ur.role.name,
            assigned_at=ur.assigned_at,
            assigned_by_id=ur.assigned_by_id,
        )
        for ur in user_roles
    ]


@router.post(
    "/{user_id}/roles",
    response_model=UserRoleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def assign_role_to_user(
    session: DBSession,
    current_user: AdminUser,
    user_id: int,
    assignment: RoleAssignment,
) -> UserRoleResponse:
    """Assign a role to a user.

    Requires admin privileges (USERS_WRITE permission).

    Args:
        session: Database session.
        current_user: The authenticated admin user.
        user_id: The user ID.
        assignment: Role assignment data.

    Returns:
        The created user role assignment.

    Raises:
        HTTPException: If user or role not found, or already assigned.
    """
    # Check user exists
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    # Check role exists and is active
    stmt = (
        select(Role)
        .options(selectinload(Role.role_permissions))
        .where(Role.id == assignment.role_id)
    )
    result = await session.execute(stmt)
    role = result.scalar_one_or_none()

    if role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {assignment.role_id} not found",
        )

    if not role.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role '{role.name}' is not active",
        )

    # Check if already assigned
    stmt = select(UserRole).where(
        UserRole.user_id == user_id,
        UserRole.role_id == assignment.role_id,
    )
    result = await session.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User already has role '{role.name}'",
        )

    # Create assignment
    user_role = UserRole(
        user_id=user_id,
        role_id=assignment.role_id,
        assigned_by_id=current_user.id,
    )
    session.add(user_role)
    await session.flush()

    return UserRoleResponse(
        id=user_role.id,
        user_id=user_role.user_id,
        role_id=user_role.role_id,
        role_name=role.name,
        assigned_at=user_role.assigned_at,
        assigned_by_id=user_role.assigned_by_id,
    )


@router.delete(
    "/{user_id}/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_role_from_user(
    session: DBSession,
    _: AdminUser,
    user_id: int,
    role_id: int,
) -> None:
    """Remove a role from a user.

    Requires admin privileges (USERS_WRITE permission).

    Args:
        session: Database session.
        user_id: The user ID.
        role_id: The role ID to remove.

    Raises:
        HTTPException: If assignment not found.
    """
    stmt = select(UserRole).where(
        UserRole.user_id == user_id,
        UserRole.role_id == role_id,
    )
    result = await session.execute(stmt)
    user_role = result.scalar_one_or_none()

    if user_role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} does not have role {role_id}",
        )

    await session.delete(user_role)


@router.get("/{user_id}/permissions", response_model=UserPermissionsResponse)
async def get_user_permissions_endpoint(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.USERS_READ))],
    user_id: int,
) -> UserPermissionsResponse:
    """Get all permissions for a user based on their roles.

    Requires USERS_READ permission.

    Args:
        session: Database session.
        user_id: The user ID.

    Returns:
        User's roles and effective permissions.

    Raises:
        HTTPException: If user not found.
    """
    # Get user
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    # Get user's roles
    stmt = (
        select(Role)
        .join(UserRole, UserRole.role_id == Role.id)
        .options(selectinload(Role.role_permissions))
        .where(UserRole.user_id == user_id)
        .where(Role.is_active.is_(True))
    )
    result = await session.execute(stmt)
    roles = list(result.scalars().all())

    # Get effective permissions
    permissions = get_user_permissions(user, roles)

    return UserPermissionsResponse(
        user_id=user.id,
        username=user.username,
        is_admin=user.is_admin,
        roles=[_role_to_response(role) for role in roles],
        permissions=sorted([p.value for p in permissions]),
    )
