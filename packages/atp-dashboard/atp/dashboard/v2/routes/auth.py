"""Authentication routes.

This module provides authentication endpoints for login, registration,
and user information retrieval.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    create_user,
)
from atp.dashboard.schemas import Token, UserCreate, UserResponse
from atp.dashboard.v2.dependencies import DBSession, RequiredUser

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=Token)
async def login(
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
async def register(session: DBSession, user_data: UserCreate) -> UserResponse:
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
        user = await create_user(
            session,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
        )
        await session.commit()
        return UserResponse.model_validate(user)
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
