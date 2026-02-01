"""SAML 2.0 implementation for SSO.

This module provides SAML 2.0-based SSO functionality including:
- SAML Service Provider (SP) implementation
- SAML assertion parsing and validation
- Attribute mapping to user info
- Just-In-Time (JIT) user provisioning
- Group-to-role mapping
- IdP metadata configuration

The implementation uses python3-saml library which is based on OneLogin's toolkit.
"""

import secrets
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser
from onelogin.saml2.settings import OneLogin_Saml2_Settings
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User


class SAMLProvider(str, Enum):
    """Supported SAML identity providers."""

    OKTA = "okta"
    AZURE_AD = "azure_ad"
    ADFS = "adfs"
    PING_IDENTITY = "ping_identity"
    ONELOGIN = "onelogin"
    GOOGLE = "google"
    GENERIC = "generic"


class SAMLGroupRoleMapping(BaseModel):
    """Mapping from IdP group/role to ATP role.

    Maps groups or roles from the identity provider to roles in ATP.
    """

    model_config = ConfigDict(frozen=True)

    idp_value: str = Field(
        ..., description="Group/role value from identity provider assertion"
    )
    atp_role: str = Field(..., description="Role name in ATP (admin, developer, etc.)")


class SAMLAttributeMapping(BaseModel):
    """Configuration for mapping SAML attributes to user fields.

    Different IdPs use different attribute names for the same data.
    This configuration allows mapping to standard user fields.
    """

    model_config = ConfigDict(extra="forbid")

    email: str = Field(
        default="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
        description="Attribute name for email",
    )
    first_name: str = Field(
        default="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
        description="Attribute name for first name",
    )
    last_name: str = Field(
        default="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
        description="Attribute name for last name",
    )
    display_name: str = Field(
        default="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
        description="Attribute name for display name",
    )
    username: str | None = Field(
        default=None,
        description="Attribute name for username (defaults to email prefix)",
    )
    groups: str = Field(
        default="http://schemas.xmlsoap.org/claims/Group",
        description="Attribute name for groups/roles",
    )


class SAMLNameIDFormat(str, Enum):
    """Supported NameID formats for SAML assertions."""

    EMAIL = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    PERSISTENT = "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"
    TRANSIENT = "urn:oasis:names:tc:SAML:2.0:nameid-format:transient"
    UNSPECIFIED = "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified"


class SAMLConfig(BaseModel):
    """SAML Service Provider configuration.

    Contains all necessary configuration for SAML-based SSO.
    """

    model_config = ConfigDict(extra="forbid")

    provider: SAMLProvider = Field(
        default=SAMLProvider.GENERIC, description="SAML provider type"
    )

    # Service Provider (ATP) configuration
    sp_entity_id: str = Field(
        ..., min_length=1, description="Service Provider Entity ID (unique identifier)"
    )
    sp_acs_url: str = Field(
        ...,
        min_length=1,
        description="Assertion Consumer Service URL (where IdP posts SAML response)",
    )
    sp_sls_url: str | None = Field(
        default=None, description="Single Logout Service URL (optional)"
    )
    sp_name_id_format: SAMLNameIDFormat = Field(
        default=SAMLNameIDFormat.EMAIL, description="Preferred NameID format"
    )

    # Identity Provider configuration
    idp_entity_id: str = Field(
        ..., min_length=1, description="Identity Provider Entity ID"
    )
    idp_sso_url: str = Field(
        ..., min_length=1, description="IdP Single Sign-On URL (SAML endpoint)"
    )
    idp_slo_url: str | None = Field(
        default=None, description="IdP Single Logout URL (optional)"
    )
    idp_x509_cert: str = Field(
        ...,
        min_length=1,
        description="IdP X.509 certificate (PEM format without headers)",
    )
    idp_x509_cert_multi: list[str] = Field(
        default_factory=list,
        description="Additional IdP certificates for key rollover",
    )

    # Attribute mapping
    attribute_mapping: SAMLAttributeMapping = Field(
        default_factory=SAMLAttributeMapping,
        description="Mapping of SAML attributes to user fields",
    )

    # Group-to-role mapping
    group_role_mappings: list[SAMLGroupRoleMapping] = Field(
        default_factory=list, description="Mappings from IdP groups to ATP roles"
    )
    default_role: str = Field(
        default="viewer", description="Default role for users without group mapping"
    )

    # Security settings
    want_assertions_signed: bool = Field(
        default=True, description="Require signed assertions"
    )
    want_messages_signed: bool = Field(
        default=False, description="Require signed messages (full response)"
    )
    want_name_id_encrypted: bool = Field(
        default=False, description="Require encrypted NameID"
    )
    want_assertions_encrypted: bool = Field(
        default=False, description="Require encrypted assertions"
    )

    # SP signing/encryption (optional, for production)
    sp_private_key: str | None = Field(
        default=None, description="SP private key for signing requests (PEM format)"
    )
    sp_x509_cert: str | None = Field(
        default=None, description="SP certificate for signature verification (PEM)"
    )

    # Advanced settings
    authn_requests_signed: bool = Field(default=False, description="Sign AuthnRequests")
    logout_requests_signed: bool = Field(
        default=False, description="Sign logout requests"
    )
    strict_mode: bool = Field(
        default=True, description="Enable strict validation (recommended)"
    )
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    @field_validator("idp_x509_cert")
    @classmethod
    def validate_cert_format(cls, v: str) -> str:
        """Remove certificate headers if present and normalize."""
        v = v.strip()
        v = v.replace("-----BEGIN CERTIFICATE-----", "")
        v = v.replace("-----END CERTIFICATE-----", "")
        v = v.replace("\n", "").replace("\r", "").replace(" ", "")
        return v


class SAMLUserInfo(BaseModel):
    """User information extracted from SAML assertion.

    Contains the user's identity information from the SAML response.
    """

    model_config = ConfigDict(extra="allow")

    name_id: str = Field(..., description="NameID from SAML assertion")
    session_index: str | None = Field(
        default=None, description="Session index for logout"
    )
    email: str = Field(..., description="User's email address")
    first_name: str | None = Field(default=None, description="User's first name")
    last_name: str | None = Field(default=None, description="User's last name")
    display_name: str | None = Field(default=None, description="User's display name")
    username: str | None = Field(default=None, description="User's username")
    groups: list[str] = Field(default_factory=list, description="Groups from IdP")
    raw_attributes: dict[str, list[str]] = Field(
        default_factory=dict, description="All attributes from assertion"
    )

    @property
    def effective_username(self) -> str:
        """Get username from available fields."""
        if self.username:
            return self.username
        return self.email.split("@")[0]


class SAMLProviderPresets:
    """Preset configurations for popular SAML providers.

    These presets provide default attribute mappings for common IdPs.
    """

    @staticmethod
    def okta() -> SAMLAttributeMapping:
        """Get Okta attribute mapping preset."""
        return SAMLAttributeMapping(
            email="email",
            first_name="firstName",
            last_name="lastName",
            display_name="displayName",
            username="login",
            groups="groups",
        )

    @staticmethod
    def azure_ad() -> SAMLAttributeMapping:
        """Get Azure AD attribute mapping preset."""
        return SAMLAttributeMapping(
            email="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            first_name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
            last_name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
            display_name="http://schemas.microsoft.com/identity/claims/displayname",
            username="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            groups="http://schemas.microsoft.com/ws/2008/06/identity/claims/groups",
        )

    @staticmethod
    def adfs() -> SAMLAttributeMapping:
        """Get ADFS attribute mapping preset."""
        return SAMLAttributeMapping(
            email="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            first_name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
            last_name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
            display_name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            username="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn",
            groups="http://schemas.xmlsoap.org/claims/Group",
        )

    @staticmethod
    def google() -> SAMLAttributeMapping:
        """Get Google Workspace SAML attribute mapping preset."""
        return SAMLAttributeMapping(
            email="email",
            first_name="firstName",
            last_name="lastName",
            display_name="displayName",
            username=None,
            groups="groups",
        )

    @staticmethod
    def ping_identity() -> SAMLAttributeMapping:
        """Get Ping Identity attribute mapping preset."""
        return SAMLAttributeMapping(
            email="email",
            first_name="givenName",
            last_name="sn",
            display_name="cn",
            username="uid",
            groups="memberOf",
        )

    @staticmethod
    def onelogin() -> SAMLAttributeMapping:
        """Get OneLogin attribute mapping preset."""
        return SAMLAttributeMapping(
            email="User.email",
            first_name="User.FirstName",
            last_name="User.LastName",
            display_name="User.DisplayName",
            username="User.Username",
            groups="memberOf",
        )

    @staticmethod
    def get_preset(provider: SAMLProvider) -> SAMLAttributeMapping:
        """Get attribute mapping preset for a provider."""
        presets = {
            SAMLProvider.OKTA: SAMLProviderPresets.okta,
            SAMLProvider.AZURE_AD: SAMLProviderPresets.azure_ad,
            SAMLProvider.ADFS: SAMLProviderPresets.adfs,
            SAMLProvider.GOOGLE: SAMLProviderPresets.google,
            SAMLProvider.PING_IDENTITY: SAMLProviderPresets.ping_identity,
            SAMLProvider.ONELOGIN: SAMLProviderPresets.onelogin,
        }
        preset_func = presets.get(provider)
        if preset_func:
            return preset_func()
        return SAMLAttributeMapping()


class SAMLError(Exception):
    """Base exception for SAML errors."""

    pass


class SAMLConfigurationError(SAMLError):
    """SAML configuration error."""

    pass


class SAMLValidationError(SAMLError):
    """SAML assertion validation error."""

    pass


class SAMLUserProvisioningError(SAMLError):
    """User provisioning error."""

    pass


def _prepare_request_from_url(
    url: str,
    post_data: dict[str, Any] | None = None,
    https: bool = True,
) -> dict[str, Any]:
    """Prepare a request dict for python3-saml from URL components.

    Args:
        url: The full URL including path
        post_data: POST data dictionary (for ACS)
        https: Whether to use HTTPS

    Returns:
        Request dict suitable for OneLogin_Saml2_Auth
    """
    parsed = urlparse(url)
    port = parsed.port

    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    return {
        "https": "on" if https else "off",
        "http_host": parsed.netloc.split(":")[0],
        "server_port": port,
        "script_name": parsed.path,
        "get_data": dict(x.split("=") for x in parsed.query.split("&"))
        if parsed.query
        else {},
        "post_data": post_data or {},
    }


class SAMLManager:
    """Manager for SAML-based SSO operations.

    Handles the complete SAML flow including:
    - AuthnRequest generation
    - SAML Response parsing and validation
    - Attribute extraction
    - JIT user provisioning
    - Group-to-role mapping
    """

    def __init__(self, config: SAMLConfig) -> None:
        """Initialize SAML manager.

        Args:
            config: SAML configuration
        """
        self.config = config
        self._settings: OneLogin_Saml2_Settings | None = None

    def _get_settings_dict(self) -> dict[str, Any]:
        """Build settings dictionary for python3-saml.

        Returns:
            Settings dictionary for OneLogin_Saml2_Settings
        """
        # Build IDP x509cert_multi if additional certs provided
        idp_cert_multi = None
        if self.config.idp_x509_cert_multi:
            idp_cert_multi = {
                "signing": [self.config.idp_x509_cert] + self.config.idp_x509_cert_multi
            }

        settings: dict[str, Any] = {
            "strict": self.config.strict_mode,
            "debug": self.config.debug_mode,
            "sp": {
                "entityId": self.config.sp_entity_id,
                "assertionConsumerService": {
                    "url": self.config.sp_acs_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "NameIDFormat": self.config.sp_name_id_format.value,
            },
            "idp": {
                "entityId": self.config.idp_entity_id,
                "singleSignOnService": {
                    "url": self.config.idp_sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": self.config.idp_x509_cert,
            },
            "security": {
                "authnRequestsSigned": self.config.authn_requests_signed,
                "logoutRequestSigned": self.config.logout_requests_signed,
                "wantAssertionsSigned": self.config.want_assertions_signed,
                "wantMessagesSigned": self.config.want_messages_signed,
                "wantNameIdEncrypted": self.config.want_name_id_encrypted,
                "wantAssertionsEncrypted": self.config.want_assertions_encrypted,
            },
        }

        # Add SLO if configured
        if self.config.sp_sls_url:
            settings["sp"]["singleLogoutService"] = {
                "url": self.config.sp_sls_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            }

        if self.config.idp_slo_url:
            settings["idp"]["singleLogoutService"] = {
                "url": self.config.idp_slo_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            }

        # Add SP certificate if configured
        if self.config.sp_x509_cert:
            settings["sp"]["x509cert"] = self.config.sp_x509_cert

        # Add SP private key if configured
        if self.config.sp_private_key:
            settings["sp"]["privateKey"] = self.config.sp_private_key

        # Add multiple IdP certificates if configured
        if idp_cert_multi:
            settings["idp"]["x509certMulti"] = idp_cert_multi

        return settings

    def get_settings(self) -> OneLogin_Saml2_Settings:
        """Get or create SAML settings.

        Returns:
            OneLogin_Saml2_Settings instance

        Raises:
            SAMLConfigurationError: If settings are invalid
        """
        if self._settings is None:
            try:
                settings_dict = self._get_settings_dict()
                self._settings = OneLogin_Saml2_Settings(settings_dict)
            except Exception as e:
                raise SAMLConfigurationError(
                    f"Failed to create SAML settings: {e}"
                ) from e
        return self._settings

    def generate_relay_state(self) -> str:
        """Generate a cryptographically secure relay state.

        Returns:
            Random state string for CSRF protection
        """
        return secrets.token_urlsafe(32)

    def get_authn_request_url(
        self,
        request_url: str,
        relay_state: str | None = None,
        force_authn: bool = False,
    ) -> str:
        """Generate SAML AuthnRequest redirect URL.

        Args:
            request_url: Current request URL for building the request
            relay_state: State parameter for CSRF protection (optional)
            force_authn: Force re-authentication at IdP

        Returns:
            URL to redirect the user to for SAML authentication

        Raises:
            SAMLConfigurationError: If unable to generate request
        """
        try:
            req = _prepare_request_from_url(request_url)
            settings = self.get_settings()
            auth = OneLogin_Saml2_Auth(req, settings)

            return auth.login(
                return_to=relay_state,
                force_authn=force_authn,
            )
        except Exception as e:
            raise SAMLConfigurationError(f"Failed to generate AuthnRequest: {e}") from e

    def process_response(
        self,
        request_url: str,
        saml_response: str,
        relay_state: str | None = None,
    ) -> tuple[SAMLUserInfo, list[str]]:
        """Process SAML response from IdP.

        Validates the SAML response and extracts user information.

        Args:
            request_url: Current request URL
            saml_response: Base64-encoded SAML response from POST
            relay_state: Expected relay state for validation (optional)

        Returns:
            Tuple of (user_info, roles)

        Raises:
            SAMLValidationError: If response validation fails
        """
        try:
            post_data = {"SAMLResponse": saml_response}
            if relay_state:
                post_data["RelayState"] = relay_state

            req = _prepare_request_from_url(request_url, post_data=post_data)
            settings = self.get_settings()
            auth = OneLogin_Saml2_Auth(req, settings)

            auth.process_response()
            errors = auth.get_errors()

            if errors:
                error_reason = auth.get_last_error_reason()
                raise SAMLValidationError(
                    f"SAML response validation failed: {', '.join(errors)}. "
                    f"Reason: {error_reason}"
                )

            if not auth.is_authenticated():
                raise SAMLValidationError("SAML authentication failed")

            # Extract user info
            user_info = self._extract_user_info(auth)
            roles = self.map_groups_to_roles(user_info.groups)

            return user_info, roles

        except SAMLValidationError:
            raise
        except Exception as e:
            raise SAMLValidationError(f"Failed to process SAML response: {e}") from e

    def _extract_user_info(self, auth: OneLogin_Saml2_Auth) -> SAMLUserInfo:
        """Extract user information from authenticated SAML auth object.

        Args:
            auth: Authenticated OneLogin_Saml2_Auth instance

        Returns:
            Structured user information

        Raises:
            SAMLValidationError: If required attributes are missing
        """
        attributes = auth.get_attributes()
        raw_name_id = auth.get_nameid()
        raw_session_index = auth.get_session_index()

        # Ensure name_id is a string
        name_id: str = str(raw_name_id) if raw_name_id else ""
        session_index: str | None = (
            str(raw_session_index) if raw_session_index else None
        )

        mapping = self.config.attribute_mapping

        # Extract email (required)
        email = self._get_attribute(attributes, mapping.email)
        if not email:
            # Fall back to NameID if it's an email format
            if name_id and "@" in name_id:
                email = name_id
            else:
                raise SAMLValidationError(
                    f"Email attribute '{mapping.email}' not found in SAML assertion"
                )

        # Extract optional fields
        first_name = self._get_attribute(attributes, mapping.first_name)
        last_name = self._get_attribute(attributes, mapping.last_name)
        display_name = self._get_attribute(attributes, mapping.display_name)
        username = None
        if mapping.username:
            username = self._get_attribute(attributes, mapping.username)

        # Extract groups
        groups: list[str] = []
        groups_attr = attributes.get(mapping.groups)
        if groups_attr:
            groups = groups_attr if isinstance(groups_attr, list) else [groups_attr]

        return SAMLUserInfo(
            name_id=name_id,
            session_index=session_index,
            email=email,
            first_name=first_name,
            last_name=last_name,
            display_name=display_name,
            username=username,
            groups=groups,
            raw_attributes={
                k: v if isinstance(v, list) else [v] for k, v in attributes.items()
            },
        )

    def _get_attribute(
        self, attributes: dict[str, list[str]], attr_name: str
    ) -> str | None:
        """Get single attribute value from attributes dict.

        Args:
            attributes: SAML attributes dictionary
            attr_name: Attribute name to extract

        Returns:
            First value of the attribute, or None if not found
        """
        values = attributes.get(attr_name)
        if values and len(values) > 0:
            return values[0]
        return None

    def map_groups_to_roles(self, groups: list[str]) -> list[str]:
        """Map IdP groups to ATP roles.

        Args:
            groups: Groups from the identity provider

        Returns:
            List of ATP role names
        """
        roles: set[str] = set()

        for mapping in self.config.group_role_mappings:
            if mapping.idp_value in groups:
                roles.add(mapping.atp_role)

        # Add default role if no mappings matched
        if not roles:
            roles.add(self.config.default_role)

        return list(roles)

    def get_logout_url(
        self,
        request_url: str,
        name_id: str,
        session_index: str | None = None,
        relay_state: str | None = None,
    ) -> str | None:
        """Generate SAML logout URL.

        Args:
            request_url: Current request URL
            name_id: User's NameID from authentication
            session_index: Session index from authentication
            relay_state: State parameter for redirect after logout

        Returns:
            Logout URL, or None if SLO not configured

        Raises:
            SAMLConfigurationError: If unable to generate logout request
        """
        if not self.config.idp_slo_url:
            return None

        try:
            req = _prepare_request_from_url(request_url)
            settings = self.get_settings()
            auth = OneLogin_Saml2_Auth(req, settings)

            return auth.logout(
                return_to=relay_state,
                name_id=name_id,
                session_index=session_index,
            )
        except Exception as e:
            raise SAMLConfigurationError(
                f"Failed to generate logout request: {e}"
            ) from e

    def process_logout_response(
        self,
        request_url: str,
        saml_response: str | None = None,
        get_data: dict[str, str] | None = None,
    ) -> bool:
        """Process SAML logout response.

        Args:
            request_url: Current request URL
            saml_response: SAML logout response (if POST binding)
            get_data: GET parameters (if Redirect binding)

        Returns:
            True if logout was successful

        Raises:
            SAMLValidationError: If logout validation fails
        """
        try:
            post_data = {}
            if saml_response:
                post_data["SAMLResponse"] = saml_response

            req = _prepare_request_from_url(request_url, post_data=post_data)
            if get_data:
                req["get_data"] = get_data

            settings = self.get_settings()
            auth = OneLogin_Saml2_Auth(req, settings)

            # For redirect binding logout
            auth.process_slo(keep_local_session=True)

            errors = auth.get_errors()
            if errors:
                raise SAMLValidationError(
                    f"Logout validation failed: {', '.join(errors)}"
                )

            return True

        except SAMLValidationError:
            raise
        except Exception as e:
            raise SAMLValidationError(f"Failed to process logout response: {e}") from e

    def get_sp_metadata(self) -> str:
        """Generate SP metadata XML.

        Returns:
            SP metadata XML string for IdP configuration

        Raises:
            SAMLConfigurationError: If unable to generate metadata
        """
        try:
            settings = self.get_settings()
            metadata = settings.get_sp_metadata()
            errors = settings.validate_metadata(metadata)

            if errors:
                raise SAMLConfigurationError(
                    f"Generated metadata validation failed: {', '.join(errors)}"
                )

            # Ensure metadata is returned as a string
            if isinstance(metadata, bytes):
                return metadata.decode("utf-8")
            return str(metadata)
        except SAMLConfigurationError:
            raise
        except Exception as e:
            raise SAMLConfigurationError(f"Failed to generate SP metadata: {e}") from e


def parse_idp_metadata(metadata_xml: str) -> dict[str, Any]:
    """Parse IdP metadata XML and extract configuration.

    Args:
        metadata_xml: IdP metadata XML string

    Returns:
        Dictionary with extracted IdP configuration

    Raises:
        SAMLConfigurationError: If metadata parsing fails
    """
    try:
        idp_data = OneLogin_Saml2_IdPMetadataParser.parse(metadata_xml)
        return idp_data.get("idp", {})
    except Exception as e:
        raise SAMLConfigurationError(f"Failed to parse IdP metadata: {e}") from e


def parse_idp_metadata_url(metadata_url: str) -> dict[str, Any]:
    """Fetch and parse IdP metadata from URL.

    Args:
        metadata_url: URL to IdP metadata XML

    Returns:
        Dictionary with extracted IdP configuration

    Raises:
        SAMLConfigurationError: If metadata fetch or parsing fails
    """
    try:
        idp_data = OneLogin_Saml2_IdPMetadataParser.parse_remote(metadata_url)
        return idp_data.get("idp", {})
    except Exception as e:
        raise SAMLConfigurationError(
            f"Failed to fetch IdP metadata from {metadata_url}: {e}"
        ) from e


async def provision_saml_user(
    session: AsyncSession,
    user_info: SAMLUserInfo,
    tenant_id: str = "default",
) -> User:
    """Provision or update a user from SAML.

    Implements Just-In-Time (JIT) user provisioning:
    - Creates new users if they don't exist
    - Updates existing users with latest info from IdP

    Args:
        session: Database session
        user_info: User information from SAML
        tenant_id: Tenant ID for the user

    Returns:
        Created or updated User

    Raises:
        SAMLUserProvisioningError: If user provisioning fails
    """
    try:
        # Check if user exists by email
        stmt = select(User).where(
            User.tenant_id == tenant_id,
            User.email == user_info.email,
        )
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            # Create new user
            user = User(
                tenant_id=tenant_id,
                username=user_info.effective_username,
                email=user_info.email,
                # SAML users don't have a local password
                hashed_password="SAML_USER_NO_LOCAL_PASSWORD",
                is_active=True,
                is_admin=False,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(user)
            await session.flush()
        else:
            # Update existing user
            user.updated_at = datetime.now()
            # Optionally update username if changed
            if user.username != user_info.effective_username:
                user.username = user_info.effective_username

        return user

    except Exception as e:
        raise SAMLUserProvisioningError(f"Failed to provision user: {e}") from e


async def assign_saml_roles(
    session: AsyncSession,
    user: User,
    role_names: list[str],
    assigned_by_id: int | None = None,
) -> None:
    """Assign roles to a SAML user based on group mappings.

    Args:
        session: Database session
        user: User to assign roles to
        role_names: List of role names to assign
        assigned_by_id: ID of user making the assignment (None for system)

    Raises:
        SAMLUserProvisioningError: If role assignment fails
    """
    from atp.dashboard.rbac.models import Role, UserRole

    try:
        for role_name in role_names:
            # Find role by name
            stmt = select(Role).where(
                Role.tenant_id == user.tenant_id,
                Role.name == role_name,
                Role.is_active.is_(True),
            )
            result = await session.execute(stmt)
            role = result.scalar_one_or_none()

            if role is None:
                continue  # Skip unknown roles

            # Check if user already has this role
            check_stmt = select(UserRole).where(
                UserRole.user_id == user.id,
                UserRole.role_id == role.id,
            )
            check_result = await session.execute(check_stmt)
            existing = check_result.scalar_one_or_none()

            if existing is None:
                # Assign role
                user_role = UserRole(
                    user_id=user.id,
                    role_id=role.id,
                    assigned_at=datetime.now(),
                    assigned_by_id=assigned_by_id,
                )
                session.add(user_role)

        await session.flush()

    except Exception as e:
        raise SAMLUserProvisioningError(f"Failed to assign roles: {e}") from e
