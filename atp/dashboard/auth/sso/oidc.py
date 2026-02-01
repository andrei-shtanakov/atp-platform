"""OIDC (OpenID Connect) implementation for SSO.

This module provides OIDC-based SSO functionality including:
- Authorization code flow
- Token validation with JWT verification
- User info extraction from ID tokens
- Just-In-Time (JIT) user provisioning
- Group-to-role mapping

Supported providers:
- Okta
- Auth0
- Azure AD (Microsoft Entra ID)
- Google Workspace
- Generic OIDC provider
"""

import secrets
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
from authlib.jose import JsonWebKey, JsonWebToken
from authlib.jose.errors import DecodeError, ExpiredTokenError, InvalidClaimError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User


class OIDCProvider(str, Enum):
    """Supported OIDC providers."""

    OKTA = "okta"
    AUTH0 = "auth0"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    GENERIC = "generic"


class GroupRoleMapping(BaseModel):
    """Mapping from IdP group to ATP role.

    Maps groups from the identity provider to roles in ATP.
    """

    model_config = ConfigDict(frozen=True)

    idp_group: str = Field(..., description="Group name from identity provider")
    atp_role: str = Field(..., description="Role name in ATP (admin, developer, etc.)")


class SSOConfig(BaseModel):
    """SSO configuration for OIDC providers.

    Contains all necessary configuration for connecting to an OIDC provider.
    """

    model_config = ConfigDict(extra="forbid")

    provider: OIDCProvider = Field(
        default=OIDCProvider.GENERIC, description="OIDC provider type"
    )
    client_id: str = Field(..., min_length=1, description="OAuth2 client ID")
    client_secret: str = Field(..., min_length=1, description="OAuth2 client secret")
    issuer_url: str = Field(
        ...,
        min_length=1,
        description="OIDC issuer URL (e.g., https://your-domain.okta.com)",
    )
    redirect_uri: str = Field(..., min_length=1, description="OAuth2 callback URL")

    # Optional overrides for discovery endpoints
    authorization_endpoint: str | None = Field(
        default=None, description="Override authorization endpoint"
    )
    token_endpoint: str | None = Field(
        default=None, description="Override token endpoint"
    )
    userinfo_endpoint: str | None = Field(
        default=None, description="Override userinfo endpoint"
    )
    jwks_uri: str | None = Field(
        default=None, description="Override JWKS URI for token validation"
    )

    # Scopes to request
    scopes: list[str] = Field(
        default_factory=lambda: ["openid", "profile", "email", "groups"],
        description="OAuth2 scopes to request",
    )

    # Group-to-role mapping
    group_role_mappings: list[GroupRoleMapping] = Field(
        default_factory=list, description="Mappings from IdP groups to ATP roles"
    )
    default_role: str = Field(
        default="viewer", description="Default role for users without group mapping"
    )

    # Claim names (can vary by provider)
    email_claim: str = Field(default="email", description="Claim name for email")
    name_claim: str = Field(default="name", description="Claim name for full name")
    groups_claim: str = Field(default="groups", description="Claim name for groups")
    username_claim: str | None = Field(
        default=None,
        description="Claim name for username (defaults to email prefix)",
    )

    # Advanced settings
    verify_at_hash: bool = Field(
        default=True, description="Verify access token hash in ID token"
    )
    allow_http: bool = Field(
        default=False, description="Allow HTTP URLs (for development only)"
    )

    @field_validator("issuer_url")
    @classmethod
    def validate_issuer_url(cls, v: str) -> str:
        """Ensure issuer URL is properly formatted."""
        # Remove trailing slash for consistency
        return v.rstrip("/")


class SSOUserInfo(BaseModel):
    """User information extracted from OIDC tokens.

    Contains the user's identity information from the ID token and userinfo endpoint.
    """

    model_config = ConfigDict(extra="allow")

    sub: str = Field(..., description="Subject identifier (unique user ID from IdP)")
    email: str = Field(..., description="User's email address")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    name: str | None = Field(default=None, description="User's full name")
    given_name: str | None = Field(default=None, description="User's first name")
    family_name: str | None = Field(default=None, description="User's last name")
    preferred_username: str | None = Field(
        default=None, description="User's preferred username"
    )
    picture: str | None = Field(
        default=None, description="URL to user's profile picture"
    )
    groups: list[str] = Field(default_factory=list, description="Groups from IdP")

    @property
    def username(self) -> str:
        """Get username from available claims."""
        if self.preferred_username:
            return self.preferred_username
        # Fall back to email prefix
        return self.email.split("@")[0]


class ProviderPresets:
    """Preset configurations for popular OIDC providers.

    These presets provide default configurations that can be used as a starting
    point for configuring SSO with popular identity providers.
    """

    @staticmethod
    def okta(
        domain: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        authorization_server_id: str = "default",
    ) -> dict[str, Any]:
        """Get Okta preset configuration.

        Args:
            domain: Okta domain (e.g., "your-company.okta.com")
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 callback URL
            authorization_server_id: Okta authorization server ID

        Returns:
            Configuration dict for SSOConfig
        """
        base_url = f"https://{domain}/oauth2/{authorization_server_id}"
        return {
            "provider": OIDCProvider.OKTA,
            "client_id": client_id,
            "client_secret": client_secret,
            "issuer_url": base_url,
            "redirect_uri": redirect_uri,
            "scopes": ["openid", "profile", "email", "groups"],
            "groups_claim": "groups",
        }

    @staticmethod
    def auth0(
        domain: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Get Auth0 preset configuration.

        Args:
            domain: Auth0 domain (e.g., "your-company.auth0.com")
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 callback URL

        Returns:
            Configuration dict for SSOConfig
        """
        return {
            "provider": OIDCProvider.AUTH0,
            "client_id": client_id,
            "client_secret": client_secret,
            "issuer_url": f"https://{domain}",
            "redirect_uri": redirect_uri,
            "scopes": ["openid", "profile", "email"],
            # Auth0 uses a custom claim for groups via rules/actions
            "groups_claim": "https://your-namespace/groups",
        }

    @staticmethod
    def azure_ad(
        tenant_id: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Get Azure AD (Microsoft Entra ID) preset configuration.

        Args:
            tenant_id: Azure AD tenant ID or domain
            client_id: OAuth2 client ID (Application ID)
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 callback URL

        Returns:
            Configuration dict for SSOConfig
        """
        return {
            "provider": OIDCProvider.AZURE_AD,
            "client_id": client_id,
            "client_secret": client_secret,
            "issuer_url": f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            "redirect_uri": redirect_uri,
            "scopes": ["openid", "profile", "email"],
            # Azure AD returns groups as GUIDs by default
            "groups_claim": "groups",
        }

    @staticmethod
    def google(
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        hd: str | None = None,
    ) -> dict[str, Any]:
        """Get Google Workspace preset configuration.

        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 callback URL
            hd: Hosted domain to restrict sign-ins (optional)

        Returns:
            Configuration dict for SSOConfig
        """
        config: dict[str, Any] = {
            "provider": OIDCProvider.GOOGLE,
            "client_id": client_id,
            "client_secret": client_secret,
            "issuer_url": "https://accounts.google.com",
            "redirect_uri": redirect_uri,
            "scopes": ["openid", "profile", "email"],
            # Google doesn't provide groups in OIDC, requires Directory API
            "groups_claim": "groups",
        }
        if hd:
            # Add hosted domain restriction
            config["hd"] = hd
        return config


class OIDCDiscoveryDocument(BaseModel):
    """OIDC Discovery document (well-known configuration)."""

    model_config = ConfigDict(extra="allow")

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str | None = None
    jwks_uri: str
    scopes_supported: list[str] = Field(default_factory=list)
    response_types_supported: list[str] = Field(default_factory=list)
    id_token_signing_alg_values_supported: list[str] = Field(default_factory=list)


class TokenResponse(BaseModel):
    """OAuth2 token response."""

    model_config = ConfigDict(extra="allow")

    access_token: str
    token_type: str
    expires_in: int | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    scope: str | None = None


class SSOError(Exception):
    """Base exception for SSO errors."""

    pass


class ConfigurationError(SSOError):
    """SSO configuration error."""

    pass


class TokenValidationError(SSOError):
    """Token validation error."""

    pass


class UserProvisioningError(SSOError):
    """User provisioning error."""

    pass


class SSOManager:
    """Manager for OIDC-based SSO operations.

    Handles the complete OIDC flow including:
    - Authorization URL generation
    - Token exchange and validation
    - User info extraction
    - JIT user provisioning
    - Group-to-role mapping
    """

    def __init__(self, config: SSOConfig) -> None:
        """Initialize SSO manager.

        Args:
            config: SSO configuration
        """
        self.config = config
        self._discovery: OIDCDiscoveryDocument | None = None
        self._jwks: dict[str, Any] | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def _fetch_discovery_document(self) -> OIDCDiscoveryDocument:
        """Fetch OIDC discovery document.

        Returns:
            OIDC discovery document

        Raises:
            ConfigurationError: If discovery document cannot be fetched
        """
        if self._discovery is not None:
            return self._discovery

        discovery_url = f"{self.config.issuer_url}/.well-known/openid-configuration"

        client = await self._get_http_client()
        try:
            response = await client.get(discovery_url)
            response.raise_for_status()
            data = response.json()
            self._discovery = OIDCDiscoveryDocument(**data)
            return self._discovery
        except httpx.HTTPError as e:
            raise ConfigurationError(
                f"Failed to fetch OIDC discovery document from {discovery_url}: {e}"
            ) from e

    async def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch JSON Web Key Set for token validation.

        Returns:
            JWKS data

        Raises:
            ConfigurationError: If JWKS cannot be fetched
        """
        if self._jwks is not None:
            return self._jwks

        discovery = await self._fetch_discovery_document()
        jwks_uri = self.config.jwks_uri or discovery.jwks_uri

        client = await self._get_http_client()
        try:
            response = await client.get(jwks_uri)
            response.raise_for_status()
            self._jwks = response.json()
            return self._jwks
        except httpx.HTTPError as e:
            raise ConfigurationError(
                f"Failed to fetch JWKS from {jwks_uri}: {e}"
            ) from e

    def generate_state(self) -> str:
        """Generate a cryptographically secure state parameter.

        Returns:
            Random state string for CSRF protection
        """
        return secrets.token_urlsafe(32)

    def generate_nonce(self) -> str:
        """Generate a cryptographically secure nonce.

        Returns:
            Random nonce for replay protection
        """
        return secrets.token_urlsafe(32)

    async def get_authorization_url(
        self,
        state: str,
        nonce: str | None = None,
        extra_params: dict[str, str] | None = None,
    ) -> str:
        """Generate authorization URL for OIDC login.

        Args:
            state: State parameter for CSRF protection
            nonce: Nonce for replay protection (generated if not provided)
            extra_params: Additional parameters to include in the URL

        Returns:
            Authorization URL to redirect the user to
        """
        discovery = await self._fetch_discovery_document()
        auth_endpoint = (
            self.config.authorization_endpoint or discovery.authorization_endpoint
        )

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
            "nonce": nonce or self.generate_nonce(),
        }

        if extra_params:
            params.update(extra_params)

        # Build query string
        query_parts = [f"{k}={v}" for k, v in params.items()]
        query_string = "&".join(query_parts)

        return f"{auth_endpoint}?{query_string}"

    async def exchange_code(
        self,
        code: str,
        expected_state: str | None = None,
        received_state: str | None = None,
    ) -> TokenResponse:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            expected_state: Expected state value (for validation)
            received_state: Received state value from callback

        Returns:
            Token response with access_token and id_token

        Raises:
            TokenValidationError: If state validation fails or token exchange fails
        """
        # Validate state if provided
        if expected_state is not None and received_state is not None:
            if expected_state != received_state:
                raise TokenValidationError("State mismatch - possible CSRF attack")

        discovery = await self._fetch_discovery_document()
        token_endpoint = self.config.token_endpoint or discovery.token_endpoint

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "redirect_uri": self.config.redirect_uri,
        }

        client = await self._get_http_client()
        try:
            response = await client.post(
                token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return TokenResponse(**response.json())
        except httpx.HTTPError as e:
            raise TokenValidationError(
                f"Failed to exchange authorization code: {e}"
            ) from e

    async def validate_id_token(
        self,
        id_token: str,
        nonce: str | None = None,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Validate and decode ID token.

        Args:
            id_token: JWT ID token to validate
            nonce: Expected nonce value (for replay protection)
            access_token: Access token for at_hash validation

        Returns:
            Decoded token claims

        Raises:
            TokenValidationError: If token validation fails
        """
        jwks_data = await self._fetch_jwks()

        try:
            # Create JWK set
            jwks = JsonWebKey.import_key_set(jwks_data)

            # Create JWT decoder
            jwt = JsonWebToken(["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"])

            # Decode and validate token
            claims = jwt.decode(
                id_token,
                jwks,
                claims_options={
                    "iss": {"essential": True, "value": self.config.issuer_url},
                    "aud": {"essential": True, "value": self.config.client_id},
                    "exp": {"essential": True},
                    "iat": {"essential": True},
                },
            )

            # Validate claims
            claims.validate()

            # Validate nonce if provided
            if nonce is not None:
                token_nonce = claims.get("nonce")
                if token_nonce != nonce:
                    raise TokenValidationError(
                        "Nonce mismatch - possible replay attack"
                    )

            return dict(claims)

        except ExpiredTokenError as e:
            raise TokenValidationError("ID token has expired") from e
        except InvalidClaimError as e:
            raise TokenValidationError(f"Invalid token claim: {e}") from e
        except DecodeError as e:
            raise TokenValidationError(f"Failed to decode ID token: {e}") from e
        except Exception as e:
            raise TokenValidationError(f"Token validation failed: {e}") from e

    async def get_userinfo(self, access_token: str) -> dict[str, Any]:
        """Fetch user info from the userinfo endpoint.

        Args:
            access_token: OAuth2 access token

        Returns:
            User info claims

        Raises:
            TokenValidationError: If userinfo request fails
        """
        discovery = await self._fetch_discovery_document()
        userinfo_endpoint = self.config.userinfo_endpoint or discovery.userinfo_endpoint

        if not userinfo_endpoint:
            raise ConfigurationError("Userinfo endpoint not available")

        client = await self._get_http_client()
        try:
            response = await client.get(
                userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise TokenValidationError(f"Failed to fetch userinfo: {e}") from e

    def extract_user_info(
        self,
        id_token_claims: dict[str, Any],
        userinfo_claims: dict[str, Any] | None = None,
    ) -> SSOUserInfo:
        """Extract user information from token claims.

        Combines claims from ID token and userinfo endpoint.

        Args:
            id_token_claims: Claims from the ID token
            userinfo_claims: Claims from the userinfo endpoint (optional)

        Returns:
            Structured user information
        """
        # Merge claims (userinfo takes precedence)
        claims = {**id_token_claims}
        if userinfo_claims:
            claims.update(userinfo_claims)

        # Extract email
        email = claims.get(self.config.email_claim)
        if not email:
            raise TokenValidationError(
                f"Email claim '{self.config.email_claim}' not found"
            )

        # Extract groups
        groups = claims.get(self.config.groups_claim, [])
        if isinstance(groups, str):
            groups = [groups]

        # Build user info
        return SSOUserInfo(
            sub=claims["sub"],
            email=email,
            email_verified=claims.get("email_verified", False),
            name=claims.get(self.config.name_claim),
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            preferred_username=claims.get(self.config.username_claim)
            if self.config.username_claim
            else claims.get("preferred_username"),
            picture=claims.get("picture"),
            groups=groups,
        )

    def map_groups_to_roles(self, groups: list[str]) -> list[str]:
        """Map IdP groups to ATP roles.

        Args:
            groups: Groups from the identity provider

        Returns:
            List of ATP role names
        """
        roles: set[str] = set()

        for mapping in self.config.group_role_mappings:
            if mapping.idp_group in groups:
                roles.add(mapping.atp_role)

        # Add default role if no mappings matched
        if not roles:
            roles.add(self.config.default_role)

        return list(roles)

    async def authenticate(
        self,
        code: str,
        state: str | None = None,
        expected_state: str | None = None,
        nonce: str | None = None,
    ) -> tuple[SSOUserInfo, list[str]]:
        """Complete authentication flow.

        Exchanges code for tokens, validates them, and extracts user info.

        Args:
            code: Authorization code from callback
            state: State received in callback
            expected_state: Expected state for validation
            nonce: Expected nonce for replay protection

        Returns:
            Tuple of (user_info, roles)
        """
        # Exchange code for tokens
        tokens = await self.exchange_code(
            code=code,
            expected_state=expected_state,
            received_state=state,
        )

        if not tokens.id_token:
            raise TokenValidationError("No ID token in response")

        # Validate ID token
        id_token_claims = await self.validate_id_token(
            id_token=tokens.id_token,
            nonce=nonce,
            access_token=tokens.access_token,
        )

        # Optionally fetch userinfo
        userinfo_claims = None
        try:
            userinfo_claims = await self.get_userinfo(tokens.access_token)
        except (ConfigurationError, TokenValidationError):
            # Userinfo endpoint might not be available
            pass

        # Extract user info
        user_info = self.extract_user_info(id_token_claims, userinfo_claims)

        # Map groups to roles
        roles = self.map_groups_to_roles(user_info.groups)

        return user_info, roles


async def provision_sso_user(
    session: AsyncSession,
    user_info: SSOUserInfo,
    tenant_id: str = "default",
) -> User:
    """Provision or update a user from SSO.

    Implements Just-In-Time (JIT) user provisioning:
    - Creates new users if they don't exist
    - Updates existing users with latest info from IdP

    Args:
        session: Database session
        user_info: User information from SSO
        tenant_id: Tenant ID for the user

    Returns:
        Created or updated User

    Raises:
        UserProvisioningError: If user provisioning fails
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
                username=user_info.username,
                email=user_info.email,
                # SSO users don't have a local password
                hashed_password="SSO_USER_NO_LOCAL_PASSWORD",
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
            if user.username != user_info.username:
                user.username = user_info.username

        return user

    except Exception as e:
        raise UserProvisioningError(f"Failed to provision user: {e}") from e


async def assign_sso_roles(
    session: AsyncSession,
    user: User,
    role_names: list[str],
    assigned_by_id: int | None = None,
) -> None:
    """Assign roles to an SSO user based on group mappings.

    Args:
        session: Database session
        user: User to assign roles to
        role_names: List of role names to assign
        assigned_by_id: ID of user making the assignment (None for system)

    Raises:
        UserProvisioningError: If role assignment fails
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
        raise UserProvisioningError(f"Failed to assign roles: {e}") from e
