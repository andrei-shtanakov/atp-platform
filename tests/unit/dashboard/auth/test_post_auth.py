"""Tests for the unified post-auth pipeline (complete_auth)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.dashboard.auth.post_auth import PostAuthError, complete_auth


class TestCompleteAuth:
    """Tests for complete_auth() — the single pipeline for all auth flows."""

    @pytest.mark.anyio
    async def test_provision_new_user(self) -> None:
        """Test provisioning a brand new user."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        with patch("atp.dashboard.auth.post_auth.create_access_token") as mock_token:
            mock_token.return_value = "jwt-token-123"
            token = await complete_auth(
                session=mock_session,
                username="testuser",
                email="test@example.com",
            )

        assert token.access_token == "jwt-token-123"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_update_existing_user(self) -> None:
        """Test updating an existing user's username."""
        mock_user = MagicMock()
        mock_user.username = "olduser"
        mock_user.email = "test@example.com"
        mock_user.id = 42

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        with patch("atp.dashboard.auth.post_auth.create_access_token") as mock_token:
            mock_token.return_value = "jwt-token-456"
            token = await complete_auth(
                session=mock_session,
                username="newuser",
                email="test@example.com",
            )

        assert token.access_token == "jwt-token-456"
        assert mock_user.username == "newuser"

    @pytest.mark.anyio
    async def test_with_roles(self) -> None:
        """Test role assignment via complete_auth."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.tenant_id = "default"

        mock_session = AsyncMock()
        # First call: user lookup, second: role lookup, third: check existing
        mock_user_result = MagicMock()
        mock_user_result.scalar_one_or_none.return_value = mock_user

        mock_role = MagicMock()
        mock_role.id = 10
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = mock_role

        mock_check_result = MagicMock()
        mock_check_result.scalar_one_or_none.return_value = None  # not assigned

        mock_session.execute.side_effect = [
            mock_user_result,
            mock_role_result,
            mock_check_result,
        ]
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        with patch("atp.dashboard.auth.post_auth.create_access_token") as mock_token:
            mock_token.return_value = "jwt-with-roles"
            token = await complete_auth(
                session=mock_session,
                username="testuser",
                email="test@example.com",
                role_names=["admin"],
            )

        assert token.access_token == "jwt-with-roles"
        # user add + role add
        assert mock_session.add.call_count >= 1

    @pytest.mark.anyio
    async def test_no_roles(self) -> None:
        """Test that role_names=None skips role assignment."""
        mock_user = MagicMock()
        mock_user.id = 1

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        with patch("atp.dashboard.auth.post_auth.create_access_token") as mock_token:
            mock_token.return_value = "jwt-no-roles"
            token = await complete_auth(
                session=mock_session,
                username="testuser",
                email="test@example.com",
                role_names=None,
            )

        assert token.access_token == "jwt-no-roles"

    @pytest.mark.anyio
    async def test_database_error_raises_post_auth_error(self) -> None:
        """Test that database errors are wrapped in PostAuthError."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database unreachable")

        with pytest.raises(PostAuthError, match="Post-auth pipeline failed"):
            await complete_auth(
                session=mock_session,
                username="testuser",
                email="test@example.com",
            )
