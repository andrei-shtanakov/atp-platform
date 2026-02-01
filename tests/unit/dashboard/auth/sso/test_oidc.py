"""Tests for OIDC SSO module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from atp.dashboard.auth.sso.oidc import (
    ConfigurationError,
    GroupRoleMapping,
    OIDCDiscoveryDocument,
    OIDCProvider,
    ProviderPresets,
    SSOConfig,
    SSOError,
    SSOManager,
    SSOUserInfo,
    TokenResponse,
    TokenValidationError,
    UserProvisioningError,
    assign_sso_roles,
    provision_sso_user,
)


class TestOIDCProvider:
    """Tests for OIDCProvider enum."""

    def test_provider_values(self) -> None:
        """Test that provider enum values are correct."""
        assert OIDCProvider.OKTA.value == "okta"
        assert OIDCProvider.AUTH0.value == "auth0"
        assert OIDCProvider.AZURE_AD.value == "azure_ad"
        assert OIDCProvider.GOOGLE.value == "google"
        assert OIDCProvider.GENERIC.value == "generic"


class TestGroupRoleMapping:
    """Tests for GroupRoleMapping model."""

    def test_create_mapping(self) -> None:
        """Test creating a group-role mapping."""
        mapping = GroupRoleMapping(idp_group="admins", atp_role="admin")
        assert mapping.idp_group == "admins"
        assert mapping.atp_role == "admin"

    def test_mapping_immutable(self) -> None:
        """Test that mapping is immutable (frozen)."""
        mapping = GroupRoleMapping(idp_group="admins", atp_role="admin")
        with pytest.raises(Exception):  # Pydantic ValidationError
            mapping.idp_group = "other"  # type: ignore[misc]


class TestSSOConfig:
    """Tests for SSOConfig model."""

    @pytest.fixture
    def valid_config_data(self) -> dict[str, Any]:
        """Return valid config data."""
        return {
            "provider": OIDCProvider.OKTA,
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "issuer_url": "https://example.okta.com",
            "redirect_uri": "https://example.com/callback",
        }

    def test_create_config(self, valid_config_data: dict[str, Any]) -> None:
        """Test creating SSO config."""
        config = SSOConfig(**valid_config_data)
        assert config.provider == OIDCProvider.OKTA
        assert config.client_id == "test-client-id"
        assert config.client_secret == "test-client-secret"
        assert config.issuer_url == "https://example.okta.com"
        assert config.redirect_uri == "https://example.com/callback"

    def test_default_scopes(self, valid_config_data: dict[str, Any]) -> None:
        """Test default scopes."""
        config = SSOConfig(**valid_config_data)
        assert "openid" in config.scopes
        assert "profile" in config.scopes
        assert "email" in config.scopes

    def test_issuer_url_trailing_slash_removed(
        self, valid_config_data: dict[str, Any]
    ) -> None:
        """Test that trailing slash is removed from issuer URL."""
        valid_config_data["issuer_url"] = "https://example.okta.com/"
        config = SSOConfig(**valid_config_data)
        assert config.issuer_url == "https://example.okta.com"

    def test_group_role_mappings(self, valid_config_data: dict[str, Any]) -> None:
        """Test group-role mappings."""
        valid_config_data["group_role_mappings"] = [
            GroupRoleMapping(idp_group="admins", atp_role="admin"),
            GroupRoleMapping(idp_group="developers", atp_role="developer"),
        ]
        config = SSOConfig(**valid_config_data)
        assert len(config.group_role_mappings) == 2

    def test_default_role(self, valid_config_data: dict[str, Any]) -> None:
        """Test default role setting."""
        config = SSOConfig(**valid_config_data)
        assert config.default_role == "viewer"

        valid_config_data["default_role"] = "developer"
        config = SSOConfig(**valid_config_data)
        assert config.default_role == "developer"


class TestSSOUserInfo:
    """Tests for SSOUserInfo model."""

    def test_create_user_info(self) -> None:
        """Test creating user info."""
        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
            email_verified=True,
            name="Test User",
            groups=["admins", "developers"],
        )
        assert user_info.sub == "user-123"
        assert user_info.email == "user@example.com"
        assert user_info.email_verified is True
        assert user_info.name == "Test User"
        assert user_info.groups == ["admins", "developers"]

    def test_username_from_preferred_username(self) -> None:
        """Test username extracted from preferred_username."""
        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
            preferred_username="testuser",
        )
        assert user_info.username == "testuser"

    def test_username_from_email(self) -> None:
        """Test username extracted from email prefix."""
        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
        )
        assert user_info.username == "user"

    def test_default_groups(self) -> None:
        """Test default empty groups."""
        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
        )
        assert user_info.groups == []


class TestProviderPresets:
    """Tests for ProviderPresets."""

    def test_okta_preset(self) -> None:
        """Test Okta preset configuration."""
        preset = ProviderPresets.okta(
            domain="example.okta.com",
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://example.com/callback",
        )
        assert preset["provider"] == OIDCProvider.OKTA
        assert preset["client_id"] == "client-id"
        assert preset["issuer_url"] == "https://example.okta.com/oauth2/default"
        assert "groups" in preset["scopes"]

    def test_okta_preset_custom_auth_server(self) -> None:
        """Test Okta preset with custom authorization server."""
        preset = ProviderPresets.okta(
            domain="example.okta.com",
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://example.com/callback",
            authorization_server_id="custom",
        )
        assert preset["issuer_url"] == "https://example.okta.com/oauth2/custom"

    def test_auth0_preset(self) -> None:
        """Test Auth0 preset configuration."""
        preset = ProviderPresets.auth0(
            domain="example.auth0.com",
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://example.com/callback",
        )
        assert preset["provider"] == OIDCProvider.AUTH0
        assert preset["issuer_url"] == "https://example.auth0.com"

    def test_azure_ad_preset(self) -> None:
        """Test Azure AD preset configuration."""
        preset = ProviderPresets.azure_ad(
            tenant_id="tenant-id",
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://example.com/callback",
        )
        assert preset["provider"] == OIDCProvider.AZURE_AD
        assert (
            preset["issuer_url"] == "https://login.microsoftonline.com/tenant-id/v2.0"
        )

    def test_google_preset(self) -> None:
        """Test Google preset configuration."""
        preset = ProviderPresets.google(
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://example.com/callback",
        )
        assert preset["provider"] == OIDCProvider.GOOGLE
        assert preset["issuer_url"] == "https://accounts.google.com"

    def test_google_preset_with_hd(self) -> None:
        """Test Google preset with hosted domain restriction."""
        preset = ProviderPresets.google(
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://example.com/callback",
            hd="example.com",
        )
        assert preset["hd"] == "example.com"


class TestSSOManager:
    """Tests for SSOManager class."""

    @pytest.fixture
    def sso_config(self) -> SSOConfig:
        """Return SSO config for testing."""
        return SSOConfig(
            provider=OIDCProvider.GENERIC,
            client_id="test-client-id",
            client_secret="test-client-secret",
            issuer_url="https://example.com",
            redirect_uri="https://app.example.com/callback",
            group_role_mappings=[
                GroupRoleMapping(idp_group="admins", atp_role="admin"),
                GroupRoleMapping(idp_group="developers", atp_role="developer"),
            ],
            default_role="viewer",
        )

    @pytest.fixture
    def sso_manager(self, sso_config: SSOConfig) -> SSOManager:
        """Return SSO manager for testing."""
        return SSOManager(sso_config)

    @pytest.fixture
    def discovery_document(self) -> dict[str, Any]:
        """Return mock discovery document."""
        return {
            "issuer": "https://example.com",
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "userinfo_endpoint": "https://example.com/userinfo",
            "jwks_uri": "https://example.com/.well-known/jwks.json",
        }

    def test_generate_state(self, sso_manager: SSOManager) -> None:
        """Test state generation."""
        state1 = sso_manager.generate_state()
        state2 = sso_manager.generate_state()
        assert len(state1) > 20
        assert state1 != state2

    def test_generate_nonce(self, sso_manager: SSOManager) -> None:
        """Test nonce generation."""
        nonce1 = sso_manager.generate_nonce()
        nonce2 = sso_manager.generate_nonce()
        assert len(nonce1) > 20
        assert nonce1 != nonce2

    def test_map_groups_to_roles(self, sso_manager: SSOManager) -> None:
        """Test group to role mapping."""
        # Admin group
        roles = sso_manager.map_groups_to_roles(["admins"])
        assert "admin" in roles

        # Developer group
        roles = sso_manager.map_groups_to_roles(["developers"])
        assert "developer" in roles

        # Multiple groups
        roles = sso_manager.map_groups_to_roles(["admins", "developers"])
        assert "admin" in roles
        assert "developer" in roles

        # Unknown group -> default role
        roles = sso_manager.map_groups_to_roles(["unknown"])
        assert roles == ["viewer"]

        # Empty groups -> default role
        roles = sso_manager.map_groups_to_roles([])
        assert roles == ["viewer"]

    @pytest.mark.anyio
    async def test_fetch_discovery_document(
        self,
        sso_manager: SSOManager,
        discovery_document: dict[str, Any],
    ) -> None:
        """Test fetching discovery document."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = discovery_document
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            doc = await sso_manager._fetch_discovery_document()
            assert doc.issuer == "https://example.com"
            assert doc.authorization_endpoint == "https://example.com/authorize"

    @pytest.mark.anyio
    async def test_fetch_discovery_document_error(
        self,
        sso_manager: SSOManager,
    ) -> None:
        """Test error handling when fetching discovery document."""
        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPError("Connection failed"),
        ):
            with pytest.raises(ConfigurationError):
                await sso_manager._fetch_discovery_document()

    @pytest.mark.anyio
    async def test_get_authorization_url(
        self,
        sso_manager: SSOManager,
        discovery_document: dict[str, Any],
    ) -> None:
        """Test authorization URL generation."""
        # Mock discovery fetch
        sso_manager._discovery = OIDCDiscoveryDocument(**discovery_document)

        auth_url = await sso_manager.get_authorization_url(
            state="test-state",
            nonce="test-nonce",
        )

        assert "https://example.com/authorize" in auth_url
        assert "client_id=test-client-id" in auth_url
        assert "state=test-state" in auth_url
        assert "nonce=test-nonce" in auth_url
        assert "response_type=code" in auth_url

    @pytest.mark.anyio
    async def test_exchange_code_state_mismatch(
        self,
        sso_manager: SSOManager,
    ) -> None:
        """Test state validation in code exchange."""
        with pytest.raises(TokenValidationError, match="State mismatch"):
            await sso_manager.exchange_code(
                code="auth-code",
                expected_state="expected",
                received_state="different",
            )

    @pytest.mark.anyio
    async def test_exchange_code_success(
        self,
        sso_manager: SSOManager,
        discovery_document: dict[str, Any],
    ) -> None:
        """Test successful code exchange."""
        sso_manager._discovery = OIDCDiscoveryDocument(**discovery_document)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-token",
            "token_type": "Bearer",
            "id_token": "id-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            tokens = await sso_manager.exchange_code(
                code="auth-code",
                expected_state="state",
                received_state="state",
            )
            assert tokens.access_token == "access-token"
            assert tokens.id_token == "id-token"

    def test_extract_user_info(self, sso_manager: SSOManager) -> None:
        """Test user info extraction from claims."""
        id_token_claims = {
            "sub": "user-123",
            "email": "user@example.com",
            "email_verified": True,
            "name": "Test User",
            "groups": ["admins"],
        }

        user_info = sso_manager.extract_user_info(id_token_claims)
        assert user_info.sub == "user-123"
        assert user_info.email == "user@example.com"
        assert user_info.name == "Test User"
        assert user_info.groups == ["admins"]

    def test_extract_user_info_with_userinfo_claims(
        self, sso_manager: SSOManager
    ) -> None:
        """Test user info extraction with userinfo endpoint data."""
        id_token_claims = {
            "sub": "user-123",
            "email": "user@example.com",
        }
        userinfo_claims = {
            "name": "Full Name from UserInfo",
            "groups": ["developers"],
        }

        user_info = sso_manager.extract_user_info(id_token_claims, userinfo_claims)
        assert user_info.name == "Full Name from UserInfo"
        assert user_info.groups == ["developers"]

    def test_extract_user_info_missing_email(self, sso_manager: SSOManager) -> None:
        """Test error when email claim is missing."""
        id_token_claims = {
            "sub": "user-123",
        }

        with pytest.raises(TokenValidationError, match="Email claim"):
            sso_manager.extract_user_info(id_token_claims)


class TestOIDCDiscoveryDocument:
    """Tests for OIDCDiscoveryDocument model."""

    def test_create_document(self) -> None:
        """Test creating discovery document."""
        doc = OIDCDiscoveryDocument(
            issuer="https://example.com",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            jwks_uri="https://example.com/jwks",
        )
        assert doc.issuer == "https://example.com"
        assert doc.userinfo_endpoint is None


class TestTokenResponse:
    """Tests for TokenResponse model."""

    def test_create_response(self) -> None:
        """Test creating token response."""
        response = TokenResponse(
            access_token="access-token",
            token_type="Bearer",
            id_token="id-token",
            expires_in=3600,
        )
        assert response.access_token == "access-token"
        assert response.token_type == "Bearer"
        assert response.id_token == "id-token"


class TestExceptions:
    """Tests for SSO exceptions."""

    def test_sso_error(self) -> None:
        """Test SSOError exception."""
        error = SSOError("Test error")
        assert str(error) == "Test error"

    def test_configuration_error(self) -> None:
        """Test ConfigurationError exception."""
        error = ConfigurationError("Config error")
        assert isinstance(error, SSOError)

    def test_token_validation_error(self) -> None:
        """Test TokenValidationError exception."""
        error = TokenValidationError("Token error")
        assert isinstance(error, SSOError)

    def test_user_provisioning_error(self) -> None:
        """Test UserProvisioningError exception."""
        error = UserProvisioningError("Provisioning error")
        assert isinstance(error, SSOError)


class TestUserProvisioning:
    """Tests for user provisioning functions."""

    @pytest.mark.anyio
    async def test_provision_sso_user_new(self) -> None:
        """Test provisioning a new SSO user."""
        mock_session = AsyncMock()

        # Mock no existing user
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
            preferred_username="testuser",
        )

        user = await provision_sso_user(
            session=mock_session,
            user_info=user_info,
            tenant_id="default",
        )

        assert user.email == "user@example.com"
        assert user.username == "testuser"
        assert user.tenant_id == "default"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_provision_sso_user_existing(self) -> None:
        """Test provisioning an existing SSO user."""
        mock_user = MagicMock()
        mock_user.username = "olduser"
        mock_user.email = "user@example.com"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        mock_session.flush = AsyncMock()

        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
            preferred_username="newuser",
        )

        user = await provision_sso_user(
            session=mock_session,
            user_info=user_info,
            tenant_id="default",
        )

        assert user == mock_user
        # Username should be updated
        assert mock_user.username == "newuser"

    @pytest.mark.anyio
    async def test_provision_sso_user_error(self) -> None:
        """Test error handling in user provisioning."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database error")

        user_info = SSOUserInfo(
            sub="user-123",
            email="user@example.com",
        )

        with pytest.raises(UserProvisioningError):
            await provision_sso_user(
                session=mock_session,
                user_info=user_info,
                tenant_id="default",
            )


class TestRoleAssignment:
    """Tests for role assignment functions."""

    @pytest.mark.anyio
    async def test_assign_sso_roles(self) -> None:
        """Test assigning roles to SSO user."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.tenant_id = "default"

        mock_role = MagicMock()
        mock_role.id = 1

        mock_session = AsyncMock()

        # First query returns role, second returns no existing assignment
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = mock_role

        mock_check_result = MagicMock()
        mock_check_result.scalar_one_or_none.return_value = None

        mock_session.execute.side_effect = [mock_role_result, mock_check_result]
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        await assign_sso_roles(
            session=mock_session,
            user=mock_user,
            role_names=["admin"],
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_assign_sso_roles_already_assigned(self) -> None:
        """Test skipping already assigned roles."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.tenant_id = "default"

        mock_role = MagicMock()
        mock_role.id = 1

        mock_existing = MagicMock()  # Existing assignment

        mock_session = AsyncMock()

        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = mock_role

        mock_check_result = MagicMock()
        mock_check_result.scalar_one_or_none.return_value = mock_existing

        mock_session.execute.side_effect = [mock_role_result, mock_check_result]
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        await assign_sso_roles(
            session=mock_session,
            user=mock_user,
            role_names=["admin"],
        )

        # Should not add since already assigned
        mock_session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_assign_sso_roles_unknown_role(self) -> None:
        """Test handling unknown roles."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.tenant_id = "default"

        mock_session = AsyncMock()

        # Role not found
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = None

        mock_session.execute.return_value = mock_role_result
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        await assign_sso_roles(
            session=mock_session,
            user=mock_user,
            role_names=["unknown_role"],
        )

        # Should not add since role not found
        mock_session.add.assert_not_called()
