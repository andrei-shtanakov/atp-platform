"""Common post-authentication pipeline.

Shared by SSO (OIDC), SAML, and Device Flow routes.
Handles: provision user -> assign roles -> create ATP token.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token
from atp.dashboard.models import User
from atp.dashboard.schemas import Token


class PostAuthError(Exception):
    """Error during post-auth pipeline."""


async def complete_auth(
    session: AsyncSession,
    username: str,
    email: str,
    role_names: list[str] | None = None,
    tenant_id: str = "default",
) -> Token:
    """Run the post-auth pipeline: provision user, assign roles, issue token.

    Args:
        session: Database session (caller manages commit).
        username: Username for the user.
        email: Email address (used as unique identifier).
        role_names: Roles to assign (from IdP group mappings). None = skip.
        tenant_id: Tenant ID for multi-tenancy.

    Returns:
        ATP access Token.
    """
    try:
        user = await _provision_user(session, username, email, tenant_id)

        if role_names:
            await _assign_roles(session, user, role_names)

        await session.commit()

        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            is_admin=user.is_admin,
        )
        return Token(access_token=access_token)

    except PostAuthError:
        raise
    except Exception as e:
        raise PostAuthError(f"Post-auth pipeline failed: {e}") from e


async def _provision_user(
    session: AsyncSession,
    username: str,
    email: str,
    tenant_id: str,
) -> User:
    """Provision or update a user (JIT provisioning)."""
    stmt = select(User).where(
        User.tenant_id == tenant_id,
        User.email == email,
    )
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            tenant_id=tenant_id,
            username=username,
            email=email,
            hashed_password="SSO_USER_NO_LOCAL_PASSWORD",
            is_active=True,
            is_admin=False,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        session.add(user)
        await session.flush()
    else:
        user.updated_at = datetime.now()
        if user.username != username:
            user.username = username

    return user


async def _assign_roles(
    session: AsyncSession,
    user: User,
    role_names: list[str],
) -> None:
    """Assign roles to user based on IdP group mappings."""
    from atp.dashboard.rbac.models import Role, UserRole

    for role_name in role_names:
        stmt = select(Role).where(
            Role.tenant_id == user.tenant_id,
            Role.name == role_name,
            Role.is_active.is_(True),
        )
        result = await session.execute(stmt)
        role = result.scalar_one_or_none()

        if role is None:
            continue

        check_stmt = select(UserRole).where(
            UserRole.user_id == user.id,
            UserRole.role_id == role.id,
        )
        check_result = await session.execute(check_stmt)
        existing = check_result.scalar_one_or_none()

        if existing is None:
            user_role = UserRole(
                user_id=user.id,
                role_id=role.id,
                assigned_at=datetime.now(),
            )
            session.add(user_role)

    await session.flush()
