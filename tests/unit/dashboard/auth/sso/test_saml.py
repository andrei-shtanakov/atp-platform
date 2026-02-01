"""Tests for SAML SSO module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.auth.sso.saml import (
    SAMLAttributeMapping,
    SAMLConfig,
    SAMLConfigurationError,
    SAMLError,
    SAMLGroupRoleMapping,
    SAMLManager,
    SAMLNameIDFormat,
    SAMLProvider,
    SAMLProviderPresets,
    SAMLUserInfo,
    SAMLUserProvisioningError,
    SAMLValidationError,
    _prepare_request_from_url,
    assign_saml_roles,
    parse_idp_metadata,
    provision_saml_user,
)


class TestSAMLProvider:
    """Tests for SAMLProvider enum."""

    def test_provider_values(self) -> None:
        """Test that provider enum values are correct."""
        assert SAMLProvider.OKTA.value == "okta"
        assert SAMLProvider.AZURE_AD.value == "azure_ad"
        assert SAMLProvider.ADFS.value == "adfs"
        assert SAMLProvider.PING_IDENTITY.value == "ping_identity"
        assert SAMLProvider.ONELOGIN.value == "onelogin"
        assert SAMLProvider.GOOGLE.value == "google"
        assert SAMLProvider.GENERIC.value == "generic"


class TestSAMLGroupRoleMapping:
    """Tests for SAMLGroupRoleMapping model."""

    def test_create_mapping(self) -> None:
        """Test creating a group-role mapping."""
        mapping = SAMLGroupRoleMapping(idp_value="admins", atp_role="admin")
        assert mapping.idp_value == "admins"
        assert mapping.atp_role == "admin"

    def test_mapping_immutable(self) -> None:
        """Test that mapping is immutable (frozen)."""
        mapping = SAMLGroupRoleMapping(idp_value="admins", atp_role="admin")
        with pytest.raises(Exception):  # Pydantic ValidationError
            mapping.idp_value = "other"  # type: ignore[misc]


class TestSAMLAttributeMapping:
    """Tests for SAMLAttributeMapping model."""

    def test_default_mapping(self) -> None:
        """Test default attribute mapping."""
        mapping = SAMLAttributeMapping()
        assert "emailaddress" in mapping.email.lower()
        assert "givenname" in mapping.first_name.lower()
        assert "surname" in mapping.last_name.lower()
        assert mapping.username is None

    def test_custom_mapping(self) -> None:
        """Test custom attribute mapping."""
        mapping = SAMLAttributeMapping(
            email="custom_email",
            first_name="custom_first",
            last_name="custom_last",
            username="custom_user",
            groups="custom_groups",
        )
        assert mapping.email == "custom_email"
        assert mapping.first_name == "custom_first"
        assert mapping.username == "custom_user"


class TestSAMLNameIDFormat:
    """Tests for SAMLNameIDFormat enum."""

    def test_format_values(self) -> None:
        """Test NameID format values."""
        assert "emailAddress" in SAMLNameIDFormat.EMAIL.value
        assert "persistent" in SAMLNameIDFormat.PERSISTENT.value
        assert "transient" in SAMLNameIDFormat.TRANSIENT.value
        assert "unspecified" in SAMLNameIDFormat.UNSPECIFIED.value


class TestSAMLConfig:
    """Tests for SAMLConfig model."""

    @pytest.fixture
    def valid_config_data(self) -> dict[str, Any]:
        """Return valid config data."""
        return {
            "sp_entity_id": "https://app.example.com/saml/metadata",
            "sp_acs_url": "https://app.example.com/saml/acs",
            "idp_entity_id": "https://idp.example.com",
            "idp_sso_url": "https://idp.example.com/saml/sso",
            "idp_x509_cert": "MIICpDCCAYwCCQDU+pQ3ZUA30jANBgkqhkiG9w0BAQsFADAUMRIwEAYD",
        }

    def test_create_config(self, valid_config_data: dict[str, Any]) -> None:
        """Test creating SAML config."""
        config = SAMLConfig(**valid_config_data)
        assert config.sp_entity_id == "https://app.example.com/saml/metadata"
        assert config.sp_acs_url == "https://app.example.com/saml/acs"
        assert config.idp_entity_id == "https://idp.example.com"
        assert config.idp_sso_url == "https://idp.example.com/saml/sso"
        assert config.provider == SAMLProvider.GENERIC

    def test_default_values(self, valid_config_data: dict[str, Any]) -> None:
        """Test default config values."""
        config = SAMLConfig(**valid_config_data)
        assert config.sp_name_id_format == SAMLNameIDFormat.EMAIL
        assert config.want_assertions_signed is True
        assert config.want_messages_signed is False
        assert config.strict_mode is True
        assert config.debug_mode is False
        assert config.default_role == "viewer"

    def test_certificate_normalization(self, valid_config_data: dict[str, Any]) -> None:
        """Test that certificate headers are removed."""
        valid_config_data["idp_x509_cert"] = (
            "-----BEGIN CERTIFICATE-----\n"
            "MIICpDCCAYwCCQDU+pQ3ZUA30jANBgkqhkiG9w0BAQsFADAUMRIwEAYD\n"
            "-----END CERTIFICATE-----"
        )
        config = SAMLConfig(**valid_config_data)
        assert "BEGIN CERTIFICATE" not in config.idp_x509_cert
        assert "END CERTIFICATE" not in config.idp_x509_cert
        assert "\n" not in config.idp_x509_cert

    def test_group_role_mappings(self, valid_config_data: dict[str, Any]) -> None:
        """Test group-role mappings."""
        valid_config_data["group_role_mappings"] = [
            SAMLGroupRoleMapping(idp_value="admins", atp_role="admin"),
            SAMLGroupRoleMapping(idp_value="developers", atp_role="developer"),
        ]
        config = SAMLConfig(**valid_config_data)
        assert len(config.group_role_mappings) == 2


class TestSAMLUserInfo:
    """Tests for SAMLUserInfo model."""

    def test_create_user_info(self) -> None:
        """Test creating user info."""
        user_info = SAMLUserInfo(
            name_id="user@example.com",
            email="user@example.com",
            first_name="Test",
            last_name="User",
            groups=["admins", "developers"],
        )
        assert user_info.name_id == "user@example.com"
        assert user_info.email == "user@example.com"
        assert user_info.first_name == "Test"
        assert user_info.last_name == "User"
        assert user_info.groups == ["admins", "developers"]

    def test_effective_username_from_username(self) -> None:
        """Test effective username extracted from username field."""
        user_info = SAMLUserInfo(
            name_id="user123",
            email="user@example.com",
            username="testuser",
        )
        assert user_info.effective_username == "testuser"

    def test_effective_username_from_email(self) -> None:
        """Test effective username extracted from email prefix."""
        user_info = SAMLUserInfo(
            name_id="user123",
            email="user@example.com",
        )
        assert user_info.effective_username == "user"

    def test_default_groups(self) -> None:
        """Test default empty groups."""
        user_info = SAMLUserInfo(
            name_id="user123",
            email="user@example.com",
        )
        assert user_info.groups == []

    def test_session_index(self) -> None:
        """Test session index field."""
        user_info = SAMLUserInfo(
            name_id="user123",
            email="user@example.com",
            session_index="_session123",
        )
        assert user_info.session_index == "_session123"


class TestSAMLProviderPresets:
    """Tests for SAMLProviderPresets."""

    def test_okta_preset(self) -> None:
        """Test Okta preset configuration."""
        preset = SAMLProviderPresets.okta()
        assert preset.email == "email"
        assert preset.first_name == "firstName"
        assert preset.last_name == "lastName"
        assert preset.username == "login"
        assert preset.groups == "groups"

    def test_azure_ad_preset(self) -> None:
        """Test Azure AD preset configuration."""
        preset = SAMLProviderPresets.azure_ad()
        assert "emailaddress" in preset.email.lower()
        assert "givenname" in preset.first_name.lower()
        assert "surname" in preset.last_name.lower()
        assert "displayname" in preset.display_name.lower()

    def test_adfs_preset(self) -> None:
        """Test ADFS preset configuration."""
        preset = SAMLProviderPresets.adfs()
        assert "emailaddress" in preset.email.lower()
        assert "upn" in preset.username.lower()  # type: ignore[union-attr]
        assert "Group" in preset.groups

    def test_google_preset(self) -> None:
        """Test Google preset configuration."""
        preset = SAMLProviderPresets.google()
        assert preset.email == "email"
        assert preset.first_name == "firstName"
        assert preset.username is None

    def test_ping_identity_preset(self) -> None:
        """Test Ping Identity preset configuration."""
        preset = SAMLProviderPresets.ping_identity()
        assert preset.email == "email"
        assert preset.first_name == "givenName"
        assert preset.last_name == "sn"
        assert preset.username == "uid"

    def test_onelogin_preset(self) -> None:
        """Test OneLogin preset configuration."""
        preset = SAMLProviderPresets.onelogin()
        assert preset.email == "User.email"
        assert "FirstName" in preset.first_name
        assert "Username" in preset.username  # type: ignore[operator]

    def test_get_preset_known_provider(self) -> None:
        """Test getting preset for known provider."""
        preset = SAMLProviderPresets.get_preset(SAMLProvider.OKTA)
        assert preset.email == "email"

    def test_get_preset_unknown_provider(self) -> None:
        """Test getting preset for generic provider."""
        preset = SAMLProviderPresets.get_preset(SAMLProvider.GENERIC)
        assert preset == SAMLAttributeMapping()


class TestPrepareRequest:
    """Tests for _prepare_request_from_url helper."""

    def test_simple_url(self) -> None:
        """Test preparing request from simple URL."""
        req = _prepare_request_from_url("https://app.example.com/saml/acs")
        assert req["https"] == "on"
        assert req["http_host"] == "app.example.com"
        assert req["server_port"] == 443
        assert req["script_name"] == "/saml/acs"

    def test_url_with_port(self) -> None:
        """Test preparing request from URL with explicit port."""
        req = _prepare_request_from_url("https://app.example.com:8443/saml/acs")
        assert req["server_port"] == 8443

    def test_url_with_query(self) -> None:
        """Test preparing request from URL with query parameters."""
        req = _prepare_request_from_url(
            "https://app.example.com/saml/acs?RelayState=abc"
        )
        assert req["get_data"] == {"RelayState": "abc"}

    def test_http_url(self) -> None:
        """Test preparing request from HTTP URL."""
        req = _prepare_request_from_url("http://localhost/saml/acs", https=False)
        assert req["https"] == "off"
        assert req["server_port"] == 80

    def test_with_post_data(self) -> None:
        """Test preparing request with POST data."""
        req = _prepare_request_from_url(
            "https://app.example.com/saml/acs",
            post_data={"SAMLResponse": "base64data"},
        )
        assert req["post_data"] == {"SAMLResponse": "base64data"}


class TestSAMLManager:
    """Tests for SAMLManager class."""

    @pytest.fixture
    def saml_config(self) -> SAMLConfig:
        """Return SAML config for testing."""
        return SAMLConfig(
            provider=SAMLProvider.GENERIC,
            sp_entity_id="https://app.example.com/saml/metadata",
            sp_acs_url="https://app.example.com/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/saml/sso",
            idp_x509_cert="MIICpDCCAYwCCQDU+pQ3ZUA30jANBgkqhkiG9w0BAQsFADAUMRIwEAYD",
            group_role_mappings=[
                SAMLGroupRoleMapping(idp_value="admins", atp_role="admin"),
                SAMLGroupRoleMapping(idp_value="developers", atp_role="developer"),
            ],
            default_role="viewer",
            strict_mode=False,  # Disable for testing
        )

    @pytest.fixture
    def saml_manager(self, saml_config: SAMLConfig) -> SAMLManager:
        """Return SAML manager for testing."""
        return SAMLManager(saml_config)

    def test_generate_relay_state(self, saml_manager: SAMLManager) -> None:
        """Test relay state generation."""
        state1 = saml_manager.generate_relay_state()
        state2 = saml_manager.generate_relay_state()
        assert len(state1) > 20
        assert state1 != state2

    def test_map_groups_to_roles(self, saml_manager: SAMLManager) -> None:
        """Test group to role mapping."""
        # Admin group
        roles = saml_manager.map_groups_to_roles(["admins"])
        assert "admin" in roles

        # Developer group
        roles = saml_manager.map_groups_to_roles(["developers"])
        assert "developer" in roles

        # Multiple groups
        roles = saml_manager.map_groups_to_roles(["admins", "developers"])
        assert "admin" in roles
        assert "developer" in roles

        # Unknown group -> default role
        roles = saml_manager.map_groups_to_roles(["unknown"])
        assert roles == ["viewer"]

        # Empty groups -> default role
        roles = saml_manager.map_groups_to_roles([])
        assert roles == ["viewer"]

    def test_get_settings_dict(self, saml_manager: SAMLManager) -> None:
        """Test settings dictionary generation."""
        settings = saml_manager._get_settings_dict()

        assert settings["strict"] is False
        assert settings["debug"] is False
        assert settings["sp"]["entityId"] == "https://app.example.com/saml/metadata"
        assert settings["idp"]["entityId"] == "https://idp.example.com"
        assert (
            settings["idp"]["singleSignOnService"]["url"]
            == "https://idp.example.com/saml/sso"
        )

    def test_get_settings_with_slo(self) -> None:
        """Test settings with SLO configured."""
        config = SAMLConfig(
            sp_entity_id="https://app.example.com/saml/metadata",
            sp_acs_url="https://app.example.com/saml/acs",
            sp_sls_url="https://app.example.com/saml/sls",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/saml/sso",
            idp_slo_url="https://idp.example.com/saml/slo",
            idp_x509_cert="MIICpDCCAYwCCQDU+pQ3ZUA30jANBgkqhkiG9w0BAQsFADAUMRIwEAYD",
        )
        manager = SAMLManager(config)
        settings = manager._get_settings_dict()

        assert "singleLogoutService" in settings["sp"]
        assert (
            settings["sp"]["singleLogoutService"]["url"]
            == "https://app.example.com/saml/sls"
        )
        assert "singleLogoutService" in settings["idp"]

    def test_extract_user_info_basic(self, saml_manager: SAMLManager) -> None:
        """Test user info extraction from auth object."""
        mock_auth = MagicMock()
        mock_auth.get_nameid.return_value = "user@example.com"
        mock_auth.get_session_index.return_value = "_session123"
        mock_auth.get_attributes.return_value = {
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": [
                "user@example.com"
            ],
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname": ["John"],
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname": ["Doe"],
            "http://schemas.xmlsoap.org/claims/Group": ["admins", "developers"],
        }

        user_info = saml_manager._extract_user_info(mock_auth)

        assert user_info.name_id == "user@example.com"
        assert user_info.session_index == "_session123"
        assert user_info.email == "user@example.com"
        assert user_info.first_name == "John"
        assert user_info.last_name == "Doe"
        assert user_info.groups == ["admins", "developers"]

    def test_extract_user_info_email_from_nameid(
        self, saml_manager: SAMLManager
    ) -> None:
        """Test email extraction from NameID when attribute missing."""
        mock_auth = MagicMock()
        mock_auth.get_nameid.return_value = "user@example.com"
        mock_auth.get_session_index.return_value = None
        mock_auth.get_attributes.return_value = {}

        user_info = saml_manager._extract_user_info(mock_auth)

        assert user_info.email == "user@example.com"

    def test_extract_user_info_missing_email(self, saml_manager: SAMLManager) -> None:
        """Test error when email is missing."""
        mock_auth = MagicMock()
        mock_auth.get_nameid.return_value = "user123"  # Not an email
        mock_auth.get_session_index.return_value = None
        mock_auth.get_attributes.return_value = {}

        with pytest.raises(SAMLValidationError, match="Email attribute"):
            saml_manager._extract_user_info(mock_auth)


class TestParseIdPMetadata:
    """Tests for IdP metadata parsing."""

    def test_parse_valid_metadata(self) -> None:
        """Test parsing valid IdP metadata."""
        metadata_xml = """<?xml version="1.0"?>
        <md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                             entityID="https://idp.example.com">
            <md:IDPSSODescriptor>
                <md:SingleSignOnService
                    Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                    Location="https://idp.example.com/saml/sso"/>
                <md:KeyDescriptor use="signing">
                    <ds:KeyInfo xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
                        <ds:X509Data>
                            <ds:X509Certificate>MIIC...</ds:X509Certificate>
                        </ds:X509Data>
                    </ds:KeyInfo>
                </md:KeyDescriptor>
            </md:IDPSSODescriptor>
        </md:EntityDescriptor>
        """
        idp_data = parse_idp_metadata(metadata_xml)

        assert idp_data.get("entityId") == "https://idp.example.com"

    def test_parse_invalid_metadata(self) -> None:
        """Test parsing invalid metadata."""
        with pytest.raises(SAMLConfigurationError):
            parse_idp_metadata("not valid xml")


class TestExceptions:
    """Tests for SAML exceptions."""

    def test_saml_error(self) -> None:
        """Test SAMLError exception."""
        error = SAMLError("Test error")
        assert str(error) == "Test error"

    def test_configuration_error(self) -> None:
        """Test SAMLConfigurationError exception."""
        error = SAMLConfigurationError("Config error")
        assert isinstance(error, SAMLError)

    def test_validation_error(self) -> None:
        """Test SAMLValidationError exception."""
        error = SAMLValidationError("Validation error")
        assert isinstance(error, SAMLError)

    def test_user_provisioning_error(self) -> None:
        """Test SAMLUserProvisioningError exception."""
        error = SAMLUserProvisioningError("Provisioning error")
        assert isinstance(error, SAMLError)


class TestUserProvisioning:
    """Tests for user provisioning functions."""

    @pytest.mark.anyio
    async def test_provision_saml_user_new(self) -> None:
        """Test provisioning a new SAML user."""
        mock_session = AsyncMock()

        # Mock no existing user
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        user_info = SAMLUserInfo(
            name_id="user@example.com",
            email="user@example.com",
            username="testuser",
        )

        user = await provision_saml_user(
            session=mock_session,
            user_info=user_info,
            tenant_id="default",
        )

        assert user.email == "user@example.com"
        assert user.username == "testuser"
        assert user.tenant_id == "default"
        assert user.hashed_password == "SAML_USER_NO_LOCAL_PASSWORD"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_provision_saml_user_existing(self) -> None:
        """Test provisioning an existing SAML user."""
        mock_user = MagicMock()
        mock_user.username = "olduser"
        mock_user.email = "user@example.com"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        mock_session.flush = AsyncMock()

        user_info = SAMLUserInfo(
            name_id="user@example.com",
            email="user@example.com",
            username="newuser",
        )

        user = await provision_saml_user(
            session=mock_session,
            user_info=user_info,
            tenant_id="default",
        )

        assert user == mock_user
        # Username should be updated
        assert mock_user.username == "newuser"

    @pytest.mark.anyio
    async def test_provision_saml_user_error(self) -> None:
        """Test error handling in user provisioning."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database error")

        user_info = SAMLUserInfo(
            name_id="user@example.com",
            email="user@example.com",
        )

        with pytest.raises(SAMLUserProvisioningError):
            await provision_saml_user(
                session=mock_session,
                user_info=user_info,
                tenant_id="default",
            )


class TestRoleAssignment:
    """Tests for role assignment functions."""

    @pytest.mark.anyio
    async def test_assign_saml_roles(self) -> None:
        """Test assigning roles to SAML user."""
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

        await assign_saml_roles(
            session=mock_session,
            user=mock_user,
            role_names=["admin"],
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_assign_saml_roles_already_assigned(self) -> None:
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

        await assign_saml_roles(
            session=mock_session,
            user=mock_user,
            role_names=["admin"],
        )

        # Should not add since already assigned
        mock_session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_assign_saml_roles_unknown_role(self) -> None:
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

        await assign_saml_roles(
            session=mock_session,
            user=mock_user,
            role_names=["unknown_role"],
        )

        # Should not add since role not found
        mock_session.add.assert_not_called()
