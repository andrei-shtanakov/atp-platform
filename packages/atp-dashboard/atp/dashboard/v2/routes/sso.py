"""SSO (Single Sign-On) routes for OIDC integration.

This module provides SSO endpoints for OIDC-based authentication:
- Authorization URL generation (redirect to IdP)
- OAuth2 callback handling
- SSO configuration management
"""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
)
from atp.dashboard.auth.sso import (
    GroupRoleMapping,
    OIDCProvider,
    ProviderPresets,
    SSOConfig,
    SSOManager,
)
from atp.dashboard.auth.sso.oidc import (
    ConfigurationError,
    SSOError,
    TokenValidationError,
    UserProvisioningError,
    assign_sso_roles,
    provision_sso_user,
)
from atp.dashboard.models import DEFAULT_TENANT_ID
from atp.dashboard.schemas import Token
from atp.dashboard.tenancy.models import Tenant, TenantSettings
from atp.dashboard.v2.dependencies import AdminUser, DBSession

router = APIRouter(prefix="/sso", tags=["sso"])

# In-memory session store for SSO state/nonce
# In production, use Redis or database-backed sessions
_sso_sessions: dict[str, dict[str, Any]] = {}


class SSOInitRequest(BaseModel):
    """Request to initiate SSO login."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(default=DEFAULT_TENANT_ID, description="Tenant ID for SSO")
    return_url: str | None = Field(
        default=None, description="URL to redirect after successful login"
    )


class SSOInitResponse(BaseModel):
    """Response with authorization URL."""

    authorization_url: str = Field(..., description="URL to redirect to for SSO login")
    state: str = Field(..., description="State parameter for CSRF protection")


class SSOCallbackParams(BaseModel):
    """Parameters received in SSO callback."""

    code: str = Field(..., description="Authorization code")
    state: str = Field(..., description="State parameter")
    error: str | None = Field(
        default=None, description="Error code if authorization failed"
    )
    error_description: str | None = Field(default=None, description="Error description")


class SSOConfigCreate(BaseModel):
    """Request to configure SSO for a tenant."""

    model_config = ConfigDict(extra="forbid")

    provider: OIDCProvider = Field(..., description="OIDC provider type")
    client_id: str = Field(..., min_length=1, description="OAuth2 client ID")
    client_secret: str = Field(..., min_length=1, description="OAuth2 client secret")
    issuer_url: str = Field(..., min_length=1, description="OIDC issuer URL")
    redirect_uri: str = Field(..., min_length=1, description="OAuth2 callback URL")
    scopes: list[str] = Field(
        default_factory=lambda: ["openid", "profile", "email", "groups"],
        description="OAuth2 scopes",
    )
    group_role_mappings: list[GroupRoleMapping] = Field(
        default_factory=list, description="Group to role mappings"
    )
    default_role: str = Field(
        default="viewer", description="Default role for new users"
    )
    groups_claim: str = Field(default="groups", description="Claim name for groups")


class SSOConfigResponse(BaseModel):
    """SSO configuration response (sensitive fields redacted)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    provider: OIDCProvider | None = None
    issuer_url: str | None = None
    scopes: list[str] = Field(default_factory=list)
    group_role_mappings: list[GroupRoleMapping] = Field(default_factory=list)
    default_role: str = "viewer"


class ProviderPresetResponse(BaseModel):
    """Response with provider preset information."""

    provider: OIDCProvider
    name: str
    description: str
    required_fields: list[str]
    optional_fields: list[str]
    documentation_url: str | None = None


async def _get_tenant_sso_config(
    session: DBSession, tenant_id: str
) -> tuple[Tenant | None, SSOConfig | None]:
    """Get tenant and SSO configuration.

    Args:
        session: Database session
        tenant_id: Tenant ID

    Returns:
        Tuple of (tenant, sso_config) or (None, None) if not found/configured
    """
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        return None, None

    settings = tenant.settings
    if not settings.sso_enabled or not settings.sso_config:
        return tenant, None

    try:
        sso_config = SSOConfig(**settings.sso_config)
        return tenant, sso_config
    except Exception:
        return tenant, None


@router.post("/init", response_model=SSOInitResponse)
async def initiate_sso(
    session: DBSession,
    request: SSOInitRequest,
) -> SSOInitResponse:
    """Initiate SSO login flow.

    Generates an authorization URL for the configured OIDC provider.
    The user should be redirected to this URL to authenticate.

    Args:
        session: Database session
        request: SSO initiation request

    Returns:
        Authorization URL and state parameter

    Raises:
        HTTPException: If SSO is not configured for the tenant
    """
    tenant, sso_config = await _get_tenant_sso_config(session, request.tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{request.tenant_id}' not found",
        )

    if sso_config is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SSO not configured for tenant '{request.tenant_id}'",
        )

    # Create SSO manager
    sso_manager = SSOManager(sso_config)

    try:
        # Generate state and nonce
        state = sso_manager.generate_state()
        nonce = sso_manager.generate_nonce()

        # Store in session for callback validation
        _sso_sessions[state] = {
            "tenant_id": request.tenant_id,
            "nonce": nonce,
            "return_url": request.return_url,
        }

        # Generate authorization URL
        auth_url = await sso_manager.get_authorization_url(
            state=state,
            nonce=nonce,
        )

        return SSOInitResponse(
            authorization_url=auth_url,
            state=state,
        )

    except ConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SSO configuration error: {e}",
        )
    finally:
        await sso_manager.close()


@router.get("/callback")
async def sso_callback(
    session: DBSession,
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State parameter"),
    error: str | None = Query(default=None, description="Error code"),
    error_description: str | None = Query(
        default=None, description="Error description"
    ),
) -> Token:
    """Handle SSO callback from identity provider.

    Exchanges the authorization code for tokens, validates them,
    provisions the user (JIT), and returns an ATP access token.

    Args:
        session: Database session
        code: Authorization code from IdP
        state: State parameter for CSRF validation
        error: Error code if authorization failed
        error_description: Error description

    Returns:
        ATP access token

    Raises:
        HTTPException: If callback validation or user provisioning fails
    """
    # Check for errors from IdP
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SSO authorization failed: {error} - {error_description}",
        )

    # Validate state
    session_data = _sso_sessions.pop(state, None)
    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter",
        )

    tenant_id = session_data["tenant_id"]
    nonce = session_data["nonce"]

    # Get SSO config
    tenant, sso_config = await _get_tenant_sso_config(session, tenant_id)

    if tenant is None or sso_config is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SSO configuration not found",
        )

    # Create SSO manager
    sso_manager = SSOManager(sso_config)

    try:
        # Complete authentication
        user_info, roles = await sso_manager.authenticate(
            code=code,
            state=state,
            expected_state=state,  # Already validated by pop
            nonce=nonce,
        )

        # Provision user (JIT)
        user = await provision_sso_user(
            session=session,
            user_info=user_info,
            tenant_id=tenant_id,
        )

        # Assign roles based on group mappings
        await assign_sso_roles(
            session=session,
            user=user,
            role_names=roles,
        )

        await session.commit()

        # Create ATP access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=access_token_expires,
        )

        return Token(access_token=access_token)

    except TokenValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {e}",
        )
    except UserProvisioningError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User provisioning failed: {e}",
        )
    except SSOError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SSO error: {e}",
        )
    finally:
        await sso_manager.close()


@router.get("/config", response_model=SSOConfigResponse)
async def get_sso_config(
    session: DBSession,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> SSOConfigResponse:
    """Get SSO configuration for a tenant.

    Returns the current SSO configuration with sensitive fields redacted.

    Args:
        session: Database session
        tenant_id: Tenant ID

    Returns:
        SSO configuration (redacted)
    """
    tenant, sso_config = await _get_tenant_sso_config(session, tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    if sso_config is None:
        return SSOConfigResponse(enabled=False)

    return SSOConfigResponse(
        enabled=True,
        provider=sso_config.provider,
        issuer_url=sso_config.issuer_url,
        scopes=sso_config.scopes,
        group_role_mappings=sso_config.group_role_mappings,
        default_role=sso_config.default_role,
    )


@router.put("/config")
async def configure_sso(
    session: DBSession,
    admin_user: AdminUser,
    config: SSOConfigCreate,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> SSOConfigResponse:
    """Configure SSO for a tenant.

    Requires admin privileges. Sets up OIDC configuration for the tenant.

    Args:
        session: Database session
        admin_user: Authenticated admin user
        config: SSO configuration
        tenant_id: Tenant ID

    Returns:
        Updated SSO configuration
    """
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    # Build SSO config
    sso_config = SSOConfig(
        provider=config.provider,
        client_id=config.client_id,
        client_secret=config.client_secret,
        issuer_url=config.issuer_url,
        redirect_uri=config.redirect_uri,
        scopes=config.scopes,
        group_role_mappings=config.group_role_mappings,
        default_role=config.default_role,
        groups_claim=config.groups_claim,
    )

    # Validate by attempting to fetch discovery document
    sso_manager = SSOManager(sso_config)
    try:
        await sso_manager._fetch_discovery_document()
    except ConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid SSO configuration: {e}",
        )
    finally:
        await sso_manager.close()

    # Update tenant settings
    settings = tenant.settings
    new_settings = TenantSettings(
        default_timeout_seconds=settings.default_timeout_seconds,
        allow_external_agents=settings.allow_external_agents,
        require_mfa=settings.require_mfa,
        sso_enabled=True,
        sso_provider=config.provider.value,
        sso_config=sso_config.model_dump(),
        custom_branding=settings.custom_branding,
        notification_channels=settings.notification_channels,
        retention_days=settings.retention_days,
    )
    tenant.settings = new_settings

    await session.commit()

    return SSOConfigResponse(
        enabled=True,
        provider=sso_config.provider,
        issuer_url=sso_config.issuer_url,
        scopes=sso_config.scopes,
        group_role_mappings=sso_config.group_role_mappings,
        default_role=sso_config.default_role,
    )


@router.delete("/config")
async def disable_sso(
    session: DBSession,
    admin_user: AdminUser,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> dict[str, str]:
    """Disable SSO for a tenant.

    Requires admin privileges. Removes SSO configuration from the tenant.

    Args:
        session: Database session
        admin_user: Authenticated admin user
        tenant_id: Tenant ID

    Returns:
        Success message
    """
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    # Update tenant settings
    settings = tenant.settings
    new_settings = TenantSettings(
        default_timeout_seconds=settings.default_timeout_seconds,
        allow_external_agents=settings.allow_external_agents,
        require_mfa=settings.require_mfa,
        sso_enabled=False,
        sso_provider=None,
        sso_config={},
        custom_branding=settings.custom_branding,
        notification_channels=settings.notification_channels,
        retention_days=settings.retention_days,
    )
    tenant.settings = new_settings

    await session.commit()

    return {"message": f"SSO disabled for tenant '{tenant_id}'"}


@router.get("/providers", response_model=list[ProviderPresetResponse])
async def list_providers() -> list[ProviderPresetResponse]:
    """List available OIDC provider presets.

    Returns information about supported identity providers
    and their configuration requirements.

    Returns:
        List of provider presets
    """
    return [
        ProviderPresetResponse(
            provider=OIDCProvider.OKTA,
            name="Okta",
            description="Enterprise identity management with OIDC",
            required_fields=["domain", "client_id", "client_secret", "redirect_uri"],
            optional_fields=["authorization_server_id"],
            documentation_url="https://developer.okta.com/docs/guides/implement-grant-type/authcode/main/",
        ),
        ProviderPresetResponse(
            provider=OIDCProvider.AUTH0,
            name="Auth0",
            description="Flexible identity platform for developers",
            required_fields=["domain", "client_id", "client_secret", "redirect_uri"],
            optional_fields=[],
            documentation_url="https://auth0.com/docs/authenticate/protocols/openid-connect-protocol",
        ),
        ProviderPresetResponse(
            provider=OIDCProvider.AZURE_AD,
            name="Azure AD (Microsoft Entra ID)",
            description="Microsoft identity platform for enterprise",
            required_fields=["tenant_id", "client_id", "client_secret", "redirect_uri"],
            optional_fields=[],
            documentation_url="https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols-oidc",
        ),
        ProviderPresetResponse(
            provider=OIDCProvider.GOOGLE,
            name="Google Workspace",
            description="Google identity for organization users",
            required_fields=["client_id", "client_secret", "redirect_uri"],
            optional_fields=["hd"],
            documentation_url="https://developers.google.com/identity/openid-connect/openid-connect",
        ),
        ProviderPresetResponse(
            provider=OIDCProvider.GENERIC,
            name="Generic OIDC",
            description="Any OIDC-compliant identity provider",
            required_fields=[
                "client_id",
                "client_secret",
                "issuer_url",
                "redirect_uri",
            ],
            optional_fields=[
                "authorization_endpoint",
                "token_endpoint",
                "userinfo_endpoint",
                "jwks_uri",
            ],
            documentation_url=None,
        ),
    ]


@router.post("/providers/{provider}/preset")
async def get_provider_preset(
    provider: OIDCProvider,
    domain: str | None = Query(default=None, description="Provider domain"),
    tenant_id: str | None = Query(default=None, description="Azure AD tenant ID"),
) -> dict[str, Any]:
    """Get preset configuration for a provider.

    Returns a configuration template that can be used as a starting
    point for SSO setup.

    Args:
        provider: OIDC provider type
        domain: Provider domain (for Okta, Auth0)
        tenant_id: Azure AD tenant ID

    Returns:
        Configuration template
    """
    if provider == OIDCProvider.OKTA:
        if not domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Domain required for Okta",
            )
        return ProviderPresets.okta(
            domain=domain,
            client_id="<your-client-id>",
            client_secret="<your-client-secret>",
            redirect_uri="<your-redirect-uri>",
        )
    elif provider == OIDCProvider.AUTH0:
        if not domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Domain required for Auth0",
            )
        return ProviderPresets.auth0(
            domain=domain,
            client_id="<your-client-id>",
            client_secret="<your-client-secret>",
            redirect_uri="<your-redirect-uri>",
        )
    elif provider == OIDCProvider.AZURE_AD:
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID required for Azure AD",
            )
        return ProviderPresets.azure_ad(
            tenant_id=tenant_id,
            client_id="<your-client-id>",
            client_secret="<your-client-secret>",
            redirect_uri="<your-redirect-uri>",
        )
    elif provider == OIDCProvider.GOOGLE:
        return ProviderPresets.google(
            client_id="<your-client-id>",
            client_secret="<your-client-secret>",
            redirect_uri="<your-redirect-uri>",
        )
    else:
        return {
            "provider": "generic",
            "client_id": "<your-client-id>",
            "client_secret": "<your-client-secret>",
            "issuer_url": "<your-issuer-url>",
            "redirect_uri": "<your-redirect-uri>",
        }
