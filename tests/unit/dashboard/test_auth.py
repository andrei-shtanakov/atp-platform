"""Tests for ATP Dashboard authentication module."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jose import jwt

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    SECRET_KEY,
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    get_password_hash,
    get_user_by_email,
    get_user_by_username,
    verify_password,
)
from atp.dashboard.models import User


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_get_password_hash(self) -> None:
        """Test password hashing."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long

    def test_verify_password_correct(self) -> None:
        """Test verifying correct password."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self) -> None:
        """Test verifying incorrect password."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        assert verify_password("wrongpassword", hashed) is False

    def test_different_passwords_different_hashes(self) -> None:
        """Test that different passwords produce different hashes."""
        hash1 = get_password_hash("password1")
        hash2 = get_password_hash("password2")
        assert hash1 != hash2

    def test_same_password_different_hashes(self) -> None:
        """Test that same password produces different hashes (salt)."""
        password = "testpassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        assert hash1 != hash2  # Different salts


class TestAccessToken:
    """Tests for JWT token functions."""

    def test_create_access_token(self) -> None:
        """Test creating access token."""
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)
        assert token is not None
        assert len(token) > 0

    def test_create_access_token_with_expiry(self) -> None:
        """Test creating access token with custom expiry."""
        data = {"sub": "testuser"}
        expires = timedelta(minutes=30)
        token = create_access_token(data, expires_delta=expires)
        assert token is not None

    def test_decode_access_token(self) -> None:
        """Test decoding access token."""
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert decoded["sub"] == "testuser"
        assert decoded["user_id"] == 1
        assert "exp" in decoded

    def test_token_expiry(self) -> None:
        """Test that token has expiry claim."""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in decoded


class TestUserQueries:
    """Tests for user query functions."""

    @pytest.mark.anyio
    async def test_get_user_by_username_found(self) -> None:
        """Test getting user by username when found."""
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        user = await get_user_by_username(mock_session, "testuser")
        assert user is not None
        assert user.username == "testuser"

    @pytest.mark.anyio
    async def test_get_user_by_username_not_found(self) -> None:
        """Test getting user by username when not found."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        user = await get_user_by_username(mock_session, "nonexistent")
        assert user is None

    @pytest.mark.anyio
    async def test_get_user_by_email_found(self) -> None:
        """Test getting user by email when found."""
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        user = await get_user_by_email(mock_session, "test@example.com")
        assert user is not None
        assert user.email == "test@example.com"


class TestAuthenticateUser:
    """Tests for user authentication."""

    @pytest.mark.anyio
    async def test_authenticate_user_success(self) -> None:
        """Test successful authentication."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password=hashed,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        user = await authenticate_user(mock_session, "testuser", password)
        assert user is not None
        assert user.username == "testuser"

    @pytest.mark.anyio
    async def test_authenticate_user_wrong_password(self) -> None:
        """Test authentication with wrong password."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password=hashed,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        user = await authenticate_user(mock_session, "testuser", "wrongpassword")
        assert user is None

    @pytest.mark.anyio
    async def test_authenticate_user_not_found(self) -> None:
        """Test authentication when user not found."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        user = await authenticate_user(mock_session, "nonexistent", "password")
        assert user is None


class TestCreateUser:
    """Tests for user creation."""

    @pytest.mark.anyio
    async def test_create_user_success(self) -> None:
        """Test successful user creation."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        user = await create_user(
            mock_session,
            username="newuser",
            email="new@example.com",
            password="password123",
        )

        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.hashed_password != "password123"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_create_user_existing_username(self) -> None:
        """Test creating user with existing username."""
        existing_user = User(
            id=1,
            username="existinguser",
            email="existing@example.com",
            hashed_password="hashed",
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_session.execute.return_value = mock_result

        with pytest.raises(ValueError, match="already exists"):
            await create_user(
                mock_session,
                username="existinguser",
                email="new@example.com",
                password="password123",
            )

    @pytest.mark.anyio
    async def test_create_admin_user(self) -> None:
        """Test creating admin user."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        user = await create_user(
            mock_session,
            username="admin",
            email="admin@example.com",
            password="password123",
            is_admin=True,
        )

        assert user.is_admin is True


class TestGetCurrentUser:
    """Tests for current user retrieval from token."""

    @pytest.mark.anyio
    async def test_get_current_user_no_token(self) -> None:
        """Test getting current user with no token."""
        user = await get_current_user(None)
        assert user is None

    @pytest.mark.anyio
    async def test_get_current_user_invalid_token(self) -> None:
        """Test getting current user with invalid token."""
        user = await get_current_user("invalid_token")
        assert user is None

    @pytest.mark.anyio
    async def test_get_current_user_valid_token(self) -> None:
        """Test getting current user with valid token."""
        # Create a valid token
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)

        # Mock the database
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            is_active=True,
        )

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        # Use context manager mock
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_db.session.return_value = mock_session

        with patch("atp.dashboard.auth.get_database", return_value=mock_db):
            user = await get_current_user(token)
            assert user is not None
            assert user.username == "testuser"

    @pytest.mark.anyio
    async def test_get_current_user_inactive(self) -> None:
        """Test getting current user when user is inactive."""
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)

        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            is_active=False,  # Inactive
        )

        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_db.session.return_value = mock_session

        with patch("atp.dashboard.auth.get_database", return_value=mock_db):
            user = await get_current_user(token)
            assert user is None


class TestConstants:
    """Tests for auth constants."""

    def test_secret_key_exists(self) -> None:
        """Test that secret key is set."""
        assert SECRET_KEY is not None
        assert len(SECRET_KEY) > 10

    def test_algorithm_is_valid(self) -> None:
        """Test that algorithm is a valid JWT algorithm."""
        assert ALGORITHM in ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]

    def test_expire_minutes_is_positive(self) -> None:
        """Test that token expiry is positive."""
        assert ACCESS_TOKEN_EXPIRE_MINUTES > 0
