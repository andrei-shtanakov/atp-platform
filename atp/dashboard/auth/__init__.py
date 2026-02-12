"""Authentication module for ATP Dashboard.

This module provides authentication and SSO support including:
- JWT token-based authentication
- OIDC/OAuth2 SSO integration
- Just-In-Time user provisioning
- Group-to-role mapping

Re-exports from the core authentication module for backward compatibility.
"""

import os
import secrets
import warnings
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth.sso import OIDCProvider, SSOConfig, SSOManager
from atp.dashboard.database import get_database
from atp.dashboard.models import User
from atp.dashboard.schemas import TokenData

# Configuration
SECRET_KEY = os.getenv("ATP_SECRET_KEY", "")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    warnings.warn(
        "ATP_SECRET_KEY is not set. Using a random secret key. "
        "Tokens will be invalidated on restart. "
        "Set ATP_SECRET_KEY for production use.",
        stacklevel=2,
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ATP_TOKEN_EXPIRE_MINUTES", "60"))

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        plain_password: Plain text password.
        hashed_password: Hashed password.

    Returns:
        True if password matches hash.
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    """Hash a password.

    Args:
        password: Plain text password.

    Returns:
        Hashed password.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token.
        expires_delta: Token expiration time.

    Returns:
        Encoded JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(tz=UTC) + expires_delta
    else:
        expire = datetime.now(tz=UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    """Get user by username.

    Args:
        session: Database session.
        username: Username to search for.

    Returns:
        User if found, None otherwise.
    """
    stmt = select(User).where(User.username == username)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Get user by email.

    Args:
        session: Database session.
        email: Email to search for.

    Returns:
        User if found, None otherwise.
    """
    stmt = select(User).where(User.email == email)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def authenticate_user(
    session: AsyncSession, username: str, password: str
) -> User | None:
    """Authenticate a user.

    Args:
        session: Database session.
        username: Username.
        password: Plain text password.

    Returns:
        User if authentication succeeds, None otherwise.
    """
    user = await get_user_by_username(session, username)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def create_user(
    session: AsyncSession,
    username: str,
    email: str,
    password: str,
    is_admin: bool = False,
) -> User:
    """Create a new user.

    Args:
        session: Database session.
        username: Username.
        email: Email address.
        password: Plain text password.
        is_admin: Whether user is admin.

    Returns:
        Created user.

    Raises:
        ValueError: If username or email already exists.
    """
    # Check for existing user
    existing = await get_user_by_username(session, username)
    if existing:
        raise ValueError(f"Username '{username}' already exists")

    existing = await get_user_by_email(session, email)
    if existing:
        raise ValueError(f"Email '{email}' already exists")

    # Create user
    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=is_admin,
    )
    session.add(user)
    await session.flush()
    return user


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> User | None:
    """Get current user from JWT token.

    This is a dependency that can be used in FastAPI routes.

    Args:
        token: JWT token from Authorization header.

    Returns:
        User if token is valid, None otherwise.
    """
    if token is None:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        user_id: int | None = payload.get("user_id")
        if username is None:
            return None
        token_data = TokenData(username=username, user_id=user_id)
    except InvalidTokenError:
        return None

    # Get user from database
    db = get_database()
    async with db.session() as session:
        user = await get_user_by_username(session, token_data.username or "")
        if user is None or not user.is_active:
            return None
        return user


async def get_current_active_user(
    current_user: Annotated[User | None, Depends(get_current_user)],
) -> User:
    """Get current active user, raising error if not authenticated.

    Args:
        current_user: Current user from token.

    Returns:
        Active user.

    Raises:
        HTTPException: If not authenticated or user is inactive.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """Get current admin user, raising error if not admin.

    Args:
        current_user: Current active user.

    Returns:
        Admin user.

    Raises:
        HTTPException: If user is not admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


__all__ = [
    # Core auth
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "ALGORITHM",
    "SECRET_KEY",
    "authenticate_user",
    "create_access_token",
    "create_user",
    "get_current_active_user",
    "get_current_admin_user",
    "get_current_user",
    "get_password_hash",
    "get_user_by_email",
    "get_user_by_username",
    "oauth2_scheme",
    "verify_password",
    # SSO
    "OIDCProvider",
    "SSOConfig",
    "SSOManager",
]
