"""Authentication routes.

This module provides authentication endpoints for login, registration,
and user information retrieval.
"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select, update

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    create_user,
)
from atp.dashboard.models import User
from atp.dashboard.rbac.models import Role, UserRole
from atp.dashboard.schemas import Token, UserCreate, UserResponse
from atp.dashboard.tokens import Invite
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession, RequiredUser
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login(
    request: Request,
    session: DBSession,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """Authenticate and get access token.

    Args:
        session: Database session.
        form_data: OAuth2 password request form with username and password.

    Returns:
        Token with access token.

    Raises:
        HTTPException: If authentication fails.
    """
    user = await authenticate_user(session, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("5/minute")
async def register(
    request: Request, session: DBSession, user_data: UserCreate
) -> UserResponse:
    """Register a new user.

    Args:
        session: Database session.
        user_data: User registration data.

    Returns:
        The created user.

    Raises:
        HTTPException: If registration fails (e.g., username already exists).
    """
    try:
        config = get_config()

        # Bootstrap: the first user ever to register becomes admin without
        # needing an invite (there is no admin yet to issue one). After that,
        # normal registration_mode rules apply.
        result = await session.execute(select(func.count(User.id)))
        user_count = result.scalar_one()
        is_first_user = user_count == 0

        if config.registration_mode == "invite" and not is_first_user:
            if not user_data.invite_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invite code required",
                )
            # Atomically increment use_count only if invite is valid
            claim_result = await session.execute(
                update(Invite)
                .where(
                    Invite.code == user_data.invite_code,
                    Invite.use_count < Invite.max_uses,
                    (
                        Invite.expires_at.is_(None)
                        | (Invite.expires_at >= datetime.now())
                    ),
                )
                .values(use_count=Invite.use_count + 1)
            )
            if claim_result.rowcount == 0:  # type: ignore[union-attr]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired invite code",
                )

        user = await create_user(
            session,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            is_admin=is_first_user,
        )

        # Assign default role
        if is_first_user:
            default_role_name = "admin"
        elif config.registration_mode == "invite":
            default_role_name = "developer"
        else:
            default_role_name = "viewer"

        role_result = await session.execute(
            select(Role).where(Role.name == default_role_name)
        )
        role = role_result.scalar_one_or_none()
        if role is not None:
            session.add(UserRole(user_id=user.id, role_id=role.id))

        # Set used_by on the invite (skipped for bootstrap path)
        if (
            config.registration_mode == "invite"
            and user_data.invite_code
            and not is_first_user
        ):
            await session.execute(
                update(Invite)
                .where(Invite.code == user_data.invite_code)
                .values(used_by_id=user.id, used_at=datetime.now())
            )

        await session.commit()
        return UserResponse.model_validate(user)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: RequiredUser) -> UserResponse:
    """Get current user information.

    Args:
        current_user: The authenticated user.

    Returns:
        User information.
    """
    return UserResponse.model_validate(current_user)
