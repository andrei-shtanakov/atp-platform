"""Role management routes for RBAC.

This module provides API endpoints for managing roles and user-role
assignments in the ATP Dashboard.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import User
from atp.dashboard.rbac import (
    DEFAULT_ROLES,
    Permission,
    Role,
    RoleCreate,
    RolePermission,
    RoleResponse,
    RoleUpdate,
    UserPermissionsResponse,
    UserRole,
    UserRoleAssign,
    UserRoleResponse,
    get_user_permissions,
    require_permission,
)
from atp.dashboard.v2.dependencies import AdminUser, DBSession, Pagination, RequiredUser

router = APIRouter(prefix="/roles", tags=["roles"])


async def _get_role_by_id(session: DBSession, role_id: int) -> Role:
    """Get a role by ID or raise 404.

    Args:
        session: Database session.
        role_id: The role ID to look up.

    Returns:
        The Role object.

    Raises:
        HTTPException: If role not found.
    """
    stmt = (
        select(Role)
        .options(selectinload(Role.role_permissions))
        .where(Role.id == role_id)
    )
    result = await session.execute(stmt)
    role = result.scalar_one_or_none()
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found",
        )
    return role


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


@router.get("", response_model=list[RoleResponse])
async def list_roles(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ROLES_READ))],
    pagination: Pagination,
    include_inactive: bool = False,
) -> list[RoleResponse]:
    """List all roles.

    Args:
        session: Database session.
        pagination: Pagination parameters.
        include_inactive: Whether to include inactive roles.

    Returns:
        List of roles.
    """
    stmt = select(Role).options(selectinload(Role.role_permissions))
    if not include_inactive:
        stmt = stmt.where(Role.is_active.is_(True))
    stmt = stmt.offset(pagination.offset).limit(pagination.limit)

    result = await session.execute(stmt)
    roles = result.scalars().all()
    return [_role_to_response(role) for role in roles]


@router.get("/defaults", response_model=dict[str, list[str]])
async def get_default_roles(
    _: Annotated[None, Depends(require_permission(Permission.ROLES_READ))],
) -> dict[str, list[str]]:
    """Get the default role definitions with their permissions.

    Returns:
        Dictionary mapping role names to their permissions.
    """
    return {
        name: [p.value for p in role_def.permissions]
        for name, role_def in DEFAULT_ROLES.items()
    }


@router.get("/permissions", response_model=list[str])
async def list_permissions(
    _: Annotated[None, Depends(require_permission(Permission.ROLES_READ))],
) -> list[str]:
    """List all available permissions.

    Returns:
        List of all permission values.
    """
    return [p.value for p in Permission]


@router.get("/{role_id}", response_model=RoleResponse)
async def get_role(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ROLES_READ))],
    role_id: int,
) -> RoleResponse:
    """Get a role by ID.

    Args:
        session: Database session.
        role_id: The role ID.

    Returns:
        The role details.
    """
    role = await _get_role_by_id(session, role_id)
    return _role_to_response(role)


@router.post("", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ROLES_WRITE))],
    role_data: RoleCreate,
) -> RoleResponse:
    """Create a new custom role.

    Args:
        session: Database session.
        role_data: Role creation data.

    Returns:
        The created role.

    Raises:
        HTTPException: If role name already exists or permissions are invalid.
    """
    # Check if role name already exists
    stmt = select(Role).where(Role.name == role_data.name)
    result = await session.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role with name '{role_data.name}' already exists",
        )

    # Validate permissions
    valid_permissions = {p.value for p in Permission}
    invalid_permissions = set(role_data.permissions) - valid_permissions
    if invalid_permissions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permissions: {', '.join(invalid_permissions)}",
        )

    # Create role
    role = Role(
        name=role_data.name,
        description=role_data.description,
        is_system=False,
    )
    session.add(role)
    await session.flush()

    # Add permissions
    for perm in role_data.permissions:
        role_perm = RolePermission(role_id=role.id, permission=perm)
        session.add(role_perm)

    await session.flush()

    # Reload with permissions
    role = await _get_role_by_id(session, role.id)
    return _role_to_response(role)


@router.patch("/{role_id}", response_model=RoleResponse)
async def update_role(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ROLES_WRITE))],
    role_id: int,
    role_data: RoleUpdate,
) -> RoleResponse:
    """Update a role.

    Args:
        session: Database session.
        role_id: The role ID to update.
        role_data: Role update data.

    Returns:
        The updated role.

    Raises:
        HTTPException: If role is a system role or permissions are invalid.
    """
    role = await _get_role_by_id(session, role_id)

    # System roles cannot have their name or permissions changed
    if role.is_system:
        if role_data.name is not None and role_data.name != role.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change name of system role",
            )
        if role_data.permissions is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change permissions of system role",
            )

    # Update fields
    if role_data.name is not None:
        # Check for duplicate name
        stmt = select(Role).where(Role.name == role_data.name, Role.id != role_id)
        result = await session.execute(stmt)
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Role with name '{role_data.name}' already exists",
            )
        role.name = role_data.name

    if role_data.description is not None:
        role.description = role_data.description

    if role_data.is_active is not None:
        role.is_active = role_data.is_active

    # Update permissions if provided
    if role_data.permissions is not None:
        # Validate permissions
        valid_permissions = {p.value for p in Permission}
        invalid_permissions = set(role_data.permissions) - valid_permissions
        if invalid_permissions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid permissions: {', '.join(invalid_permissions)}",
            )

        # Remove old permissions
        for rp in list(role.role_permissions):
            await session.delete(rp)

        # Flush to ensure deletes are committed before inserts
        await session.flush()

        # Add new permissions
        for perm in role_data.permissions:
            role_perm = RolePermission(role_id=role.id, permission=perm)
            session.add(role_perm)

    await session.flush()

    # Expire the role to force a fresh load of relationships
    session.expire(role)

    # Reload with permissions
    role = await _get_role_by_id(session, role_id)
    return _role_to_response(role)


@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ROLES_DELETE))],
    role_id: int,
) -> None:
    """Delete a role.

    Args:
        session: Database session.
        role_id: The role ID to delete.

    Raises:
        HTTPException: If role is a system role.
    """
    role = await _get_role_by_id(session, role_id)

    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete system role",
        )

    await session.delete(role)


# User-Role assignment endpoints


@router.get("/users/{user_id}/roles", response_model=list[UserRoleResponse])
async def get_user_roles(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.USERS_READ))],
    user_id: int,
) -> list[UserRoleResponse]:
    """Get all roles assigned to a user.

    Args:
        session: Database session.
        user_id: The user ID.

    Returns:
        List of user role assignments.
    """
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
    "/users/assign",
    response_model=UserRoleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def assign_role_to_user(
    session: DBSession,
    current_user: AdminUser,
    assignment: UserRoleAssign,
) -> UserRoleResponse:
    """Assign a role to a user.

    Args:
        session: Database session.
        current_user: The authenticated admin user.
        assignment: Role assignment data.

    Returns:
        The created user role assignment.

    Raises:
        HTTPException: If user or role not found, or already assigned.
    """
    # Check user exists
    stmt = select(User).where(User.id == assignment.user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {assignment.user_id} not found",
        )

    # Check role exists and is active
    role = await _get_role_by_id(session, assignment.role_id)
    if not role.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role '{role.name}' is not active",
        )

    # Check if already assigned
    stmt = select(UserRole).where(
        UserRole.user_id == assignment.user_id,
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
        user_id=assignment.user_id,
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
    "/users/{user_id}/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def remove_role_from_user(
    session: DBSession,
    _: AdminUser,
    user_id: int,
    role_id: int,
) -> None:
    """Remove a role from a user.

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


@router.get("/users/{user_id}/permissions", response_model=UserPermissionsResponse)
async def get_user_permissions_endpoint(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.USERS_READ))],
    user_id: int,
) -> UserPermissionsResponse:
    """Get all permissions for a user based on their roles.

    Args:
        session: Database session.
        user_id: The user ID.

    Returns:
        User's roles and effective permissions.

    Raises:
        HTTPException: If user not found.
    """
    # Get user
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
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


@router.get("/me/permissions", response_model=UserPermissionsResponse)
async def get_my_permissions(
    session: DBSession,
    current_user: RequiredUser,
) -> UserPermissionsResponse:
    """Get current user's roles and permissions.

    Args:
        session: Database session.
        current_user: The authenticated user.

    Returns:
        Current user's roles and effective permissions.
    """
    # Get user's roles
    stmt = (
        select(Role)
        .join(UserRole, UserRole.role_id == Role.id)
        .options(selectinload(Role.role_permissions))
        .where(UserRole.user_id == current_user.id)
        .where(Role.is_active.is_(True))
    )
    result = await session.execute(stmt)
    roles = list(result.scalars().all())

    # Get effective permissions
    permissions = get_user_permissions(current_user, roles)

    return UserPermissionsResponse(
        user_id=current_user.id,
        username=current_user.username,
        is_admin=current_user.is_admin,
        roles=[_role_to_response(role) for role in roles],
        permissions=sorted([p.value for p in permissions]),
    )
