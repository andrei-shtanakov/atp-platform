"""SAML 2.0 SSO routes for authentication.

This module provides SAML SP endpoints for authentication:
- SP metadata endpoint (for IdP configuration)
- AuthnRequest initiation (redirect to IdP)
- Assertion Consumer Service (ACS) callback
- Single Logout Service (SLS)
- SAML configuration management
"""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
)
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
    SAMLUserProvisioningError,
    SAMLValidationError,
    assign_saml_roles,
    parse_idp_metadata,
    parse_idp_metadata_url,
    provision_saml_user,
)
from atp.dashboard.models import DEFAULT_TENANT_ID
from atp.dashboard.schemas import Token
from atp.dashboard.tenancy.models import Tenant, TenantSettings
from atp.dashboard.v2.dependencies import AdminUser, DBSession

router = APIRouter(prefix="/saml", tags=["saml"])

# In-memory session store for SAML relay state
# In production, use Redis or database-backed sessions
_saml_sessions: dict[str, dict[str, Any]] = {}


class SAMLInitRequest(BaseModel):
    """Request to initiate SAML login."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(default=DEFAULT_TENANT_ID, description="Tenant ID for SAML")
    return_url: str | None = Field(
        default=None, description="URL to redirect after successful login"
    )
    force_authn: bool = Field(
        default=False, description="Force re-authentication at IdP"
    )


class SAMLInitResponse(BaseModel):
    """Response with SAML AuthnRequest redirect URL."""

    redirect_url: str = Field(..., description="URL to redirect to for SAML login")
    relay_state: str = Field(..., description="Relay state for CSRF protection")


class SAMLConfigCreate(BaseModel):
    """Request to configure SAML for a tenant."""

    model_config = ConfigDict(extra="forbid")

    provider: SAMLProvider = Field(
        default=SAMLProvider.GENERIC, description="SAML provider type"
    )
    sp_entity_id: str = Field(
        ..., min_length=1, description="Service Provider Entity ID"
    )
    sp_acs_url: str = Field(
        ..., min_length=1, description="Assertion Consumer Service URL"
    )
    sp_sls_url: str | None = Field(
        default=None, description="Single Logout Service URL"
    )
    sp_name_id_format: SAMLNameIDFormat = Field(
        default=SAMLNameIDFormat.EMAIL, description="NameID format"
    )
    idp_entity_id: str = Field(
        ..., min_length=1, description="Identity Provider Entity ID"
    )
    idp_sso_url: str = Field(..., min_length=1, description="IdP SSO URL")
    idp_slo_url: str | None = Field(default=None, description="IdP SLO URL")
    idp_x509_cert: str = Field(..., min_length=1, description="IdP X.509 certificate")
    attribute_mapping: SAMLAttributeMapping = Field(
        default_factory=SAMLAttributeMapping, description="Attribute mapping"
    )
    group_role_mappings: list[SAMLGroupRoleMapping] = Field(
        default_factory=list, description="Group to role mappings"
    )
    default_role: str = Field(
        default="viewer", description="Default role for new users"
    )
    want_assertions_signed: bool = Field(
        default=True, description="Require signed assertions"
    )


class SAMLConfigResponse(BaseModel):
    """SAML configuration response (sensitive fields redacted)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    provider: SAMLProvider | None = None
    sp_entity_id: str | None = None
    sp_acs_url: str | None = None
    idp_entity_id: str | None = None
    idp_sso_url: str | None = None
    attribute_mapping: SAMLAttributeMapping | None = None
    group_role_mappings: list[SAMLGroupRoleMapping] = Field(default_factory=list)
    default_role: str = "viewer"


class SAMLProviderPresetResponse(BaseModel):
    """Response with provider preset information."""

    provider: SAMLProvider
    name: str
    description: str
    attribute_mapping: SAMLAttributeMapping
    documentation_url: str | None = None


class SAMLIdPMetadataRequest(BaseModel):
    """Request to parse IdP metadata."""

    model_config = ConfigDict(extra="forbid")

    metadata_xml: str | None = Field(default=None, description="IdP metadata XML")
    metadata_url: str | None = Field(default=None, description="URL to IdP metadata")


class SAMLIdPMetadataResponse(BaseModel):
    """Response with parsed IdP metadata."""

    entity_id: str | None = None
    sso_url: str | None = None
    slo_url: str | None = None
    x509_cert: str | None = None
    name_id_format: str | None = None


async def _get_tenant_saml_config(
    session: DBSession, tenant_id: str
) -> tuple[Tenant | None, SAMLConfig | None]:
    """Get tenant and SAML configuration.

    Args:
        session: Database session
        tenant_id: Tenant ID

    Returns:
        Tuple of (tenant, saml_config) or (None, None) if not found/configured
    """
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        return None, None

    settings = tenant.settings
    if not settings.sso_enabled or settings.sso_provider != "saml":
        return tenant, None

    if not settings.sso_config:
        return tenant, None

    try:
        saml_config = SAMLConfig(**settings.sso_config)
        return tenant, saml_config
    except Exception:
        return tenant, None


@router.post("/init", response_model=SAMLInitResponse)
async def initiate_saml(
    request: Request,
    session: DBSession,
    body: SAMLInitRequest,
) -> SAMLInitResponse:
    """Initiate SAML login flow.

    Generates a SAML AuthnRequest and returns the redirect URL.
    The user should be redirected to this URL to authenticate.

    Args:
        request: FastAPI request
        session: Database session
        body: SAML initiation request

    Returns:
        Redirect URL and relay state

    Raises:
        HTTPException: If SAML is not configured for the tenant
    """
    tenant, saml_config = await _get_tenant_saml_config(session, body.tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{body.tenant_id}' not found",
        )

    if saml_config is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SAML not configured for tenant '{body.tenant_id}'",
        )

    try:
        saml_manager = SAMLManager(saml_config)
        relay_state = saml_manager.generate_relay_state()

        # Store session data for callback
        _saml_sessions[relay_state] = {
            "tenant_id": body.tenant_id,
            "return_url": body.return_url,
        }

        # Get current request URL for building AuthnRequest
        request_url = str(request.url)

        redirect_url = saml_manager.get_authn_request_url(
            request_url=request_url,
            relay_state=relay_state,
            force_authn=body.force_authn,
        )

        return SAMLInitResponse(
            redirect_url=redirect_url,
            relay_state=relay_state,
        )

    except SAMLConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SAML configuration error: {e}",
        )


@router.post("/acs")
async def assertion_consumer_service(
    request: Request,
    session: DBSession,
    saml_response: str = Form(..., alias="SAMLResponse"),
    relay_state: str = Form(None, alias="RelayState"),
) -> Token:
    """SAML Assertion Consumer Service (ACS) endpoint.

    Processes the SAML response from the IdP, validates the assertion,
    provisions the user (JIT), and returns an ATP access token.

    Args:
        request: FastAPI request
        session: Database session
        saml_response: Base64-encoded SAML response
        relay_state: Relay state parameter

    Returns:
        ATP access token

    Raises:
        HTTPException: If assertion validation or user provisioning fails
    """
    # Validate relay state
    session_data = _saml_sessions.pop(relay_state, None) if relay_state else None

    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired relay state parameter",
        )

    tenant_id = session_data["tenant_id"]

    # Get SAML config
    tenant, saml_config = await _get_tenant_saml_config(session, tenant_id)

    if tenant is None or saml_config is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SAML configuration not found",
        )

    try:
        saml_manager = SAMLManager(saml_config)
        request_url = str(request.url)

        # Process SAML response
        user_info, roles = saml_manager.process_response(
            request_url=request_url,
            saml_response=saml_response,
            relay_state=relay_state,
        )

        # Provision user (JIT)
        user = await provision_saml_user(
            session=session,
            user_info=user_info,
            tenant_id=tenant_id,
        )

        # Assign roles based on group mappings
        await assign_saml_roles(
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

    except SAMLValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"SAML assertion validation failed: {e}",
        )
    except SAMLUserProvisioningError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User provisioning failed: {e}",
        )
    except SAMLError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SAML error: {e}",
        )


@router.get("/metadata")
async def get_sp_metadata(
    session: DBSession,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> Response:
    """Get SAML Service Provider metadata.

    Returns the SP metadata XML that can be used to configure the IdP.

    Args:
        session: Database session
        tenant_id: Tenant ID

    Returns:
        SP metadata XML

    Raises:
        HTTPException: If SAML is not configured
    """
    tenant, saml_config = await _get_tenant_saml_config(session, tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    if saml_config is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SAML not configured for tenant '{tenant_id}'",
        )

    try:
        saml_manager = SAMLManager(saml_config)
        metadata = saml_manager.get_sp_metadata()

        return Response(
            content=metadata,
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=sp_metadata.xml"},
        )

    except SAMLConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SP metadata: {e}",
        )


@router.post("/sls")
async def single_logout_service(
    request: Request,
    session: DBSession,
    saml_response: str | None = Form(None, alias="SAMLResponse"),
    relay_state: str | None = Form(None, alias="RelayState"),
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> dict[str, str]:
    """SAML Single Logout Service (SLS) endpoint.

    Processes logout response from IdP.

    Args:
        request: FastAPI request
        session: Database session
        saml_response: SAML logout response
        relay_state: Relay state
        tenant_id: Tenant ID

    Returns:
        Success message
    """
    tenant, saml_config = await _get_tenant_saml_config(session, tenant_id)

    if tenant is None or saml_config is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SAML configuration not found",
        )

    try:
        saml_manager = SAMLManager(saml_config)
        request_url = str(request.url)

        success = saml_manager.process_logout_response(
            request_url=request_url,
            saml_response=saml_response,
        )

        if success:
            return {"message": "Logout successful"}
        else:
            return {"message": "Logout processed"}

    except SAMLValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Logout validation failed: {e}",
        )


@router.get("/config", response_model=SAMLConfigResponse)
async def get_saml_config(
    session: DBSession,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> SAMLConfigResponse:
    """Get SAML configuration for a tenant.

    Returns the current SAML configuration with sensitive fields redacted.

    Args:
        session: Database session
        tenant_id: Tenant ID

    Returns:
        SAML configuration (redacted)
    """
    tenant, saml_config = await _get_tenant_saml_config(session, tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    if saml_config is None:
        return SAMLConfigResponse(enabled=False)

    return SAMLConfigResponse(
        enabled=True,
        provider=saml_config.provider,
        sp_entity_id=saml_config.sp_entity_id,
        sp_acs_url=saml_config.sp_acs_url,
        idp_entity_id=saml_config.idp_entity_id,
        idp_sso_url=saml_config.idp_sso_url,
        attribute_mapping=saml_config.attribute_mapping,
        group_role_mappings=saml_config.group_role_mappings,
        default_role=saml_config.default_role,
    )


@router.put("/config")
async def configure_saml(
    session: DBSession,
    admin_user: AdminUser,
    config: SAMLConfigCreate,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> SAMLConfigResponse:
    """Configure SAML for a tenant.

    Requires admin privileges. Sets up SAML configuration for the tenant.

    Args:
        session: Database session
        admin_user: Authenticated admin user
        config: SAML configuration
        tenant_id: Tenant ID

    Returns:
        Updated SAML configuration
    """
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    # Build SAML config
    saml_config = SAMLConfig(
        provider=config.provider,
        sp_entity_id=config.sp_entity_id,
        sp_acs_url=config.sp_acs_url,
        sp_sls_url=config.sp_sls_url,
        sp_name_id_format=config.sp_name_id_format,
        idp_entity_id=config.idp_entity_id,
        idp_sso_url=config.idp_sso_url,
        idp_slo_url=config.idp_slo_url,
        idp_x509_cert=config.idp_x509_cert,
        attribute_mapping=config.attribute_mapping,
        group_role_mappings=config.group_role_mappings,
        default_role=config.default_role,
        want_assertions_signed=config.want_assertions_signed,
    )

    # Validate by creating settings
    try:
        saml_manager = SAMLManager(saml_config)
        saml_manager.get_settings()
    except SAMLConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid SAML configuration: {e}",
        )

    # Update tenant settings
    settings = tenant.settings
    new_settings = TenantSettings(
        default_timeout_seconds=settings.default_timeout_seconds,
        allow_external_agents=settings.allow_external_agents,
        require_mfa=settings.require_mfa,
        sso_enabled=True,
        sso_provider="saml",
        sso_config=saml_config.model_dump(),
        custom_branding=settings.custom_branding,
        notification_channels=settings.notification_channels,
        retention_days=settings.retention_days,
    )
    tenant.settings = new_settings

    await session.commit()

    return SAMLConfigResponse(
        enabled=True,
        provider=saml_config.provider,
        sp_entity_id=saml_config.sp_entity_id,
        sp_acs_url=saml_config.sp_acs_url,
        idp_entity_id=saml_config.idp_entity_id,
        idp_sso_url=saml_config.idp_sso_url,
        attribute_mapping=saml_config.attribute_mapping,
        group_role_mappings=saml_config.group_role_mappings,
        default_role=saml_config.default_role,
    )


@router.delete("/config")
async def disable_saml(
    session: DBSession,
    admin_user: AdminUser,
    tenant_id: str = Query(default=DEFAULT_TENANT_ID, description="Tenant ID"),
) -> dict[str, str]:
    """Disable SAML for a tenant.

    Requires admin privileges. Removes SAML configuration from the tenant.

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

    return {"message": f"SAML disabled for tenant '{tenant_id}'"}


@router.get("/providers", response_model=list[SAMLProviderPresetResponse])
async def list_providers() -> list[SAMLProviderPresetResponse]:
    """List available SAML provider presets.

    Returns information about supported identity providers
    and their attribute mappings.

    Returns:
        List of provider presets
    """
    return [
        SAMLProviderPresetResponse(
            provider=SAMLProvider.OKTA,
            name="Okta",
            description="Enterprise identity management with SAML 2.0",
            attribute_mapping=SAMLProviderPresets.okta(),
            documentation_url="https://help.okta.com/en-us/content/topics/apps/apps_app_integration_wizard_saml.htm",
        ),
        SAMLProviderPresetResponse(
            provider=SAMLProvider.AZURE_AD,
            name="Azure AD (Microsoft Entra ID)",
            description="Microsoft identity platform for enterprise",
            attribute_mapping=SAMLProviderPresets.azure_ad(),
            documentation_url="https://learn.microsoft.com/en-us/entra/identity-platform/single-sign-on-saml-protocol",
        ),
        SAMLProviderPresetResponse(
            provider=SAMLProvider.ADFS,
            name="Active Directory Federation Services",
            description="On-premises Microsoft identity federation",
            attribute_mapping=SAMLProviderPresets.adfs(),
            documentation_url="https://learn.microsoft.com/en-us/windows-server/identity/ad-fs/operations/add-an-application-group",
        ),
        SAMLProviderPresetResponse(
            provider=SAMLProvider.GOOGLE,
            name="Google Workspace",
            description="Google identity for organization users",
            attribute_mapping=SAMLProviderPresets.google(),
            documentation_url="https://support.google.com/a/answer/6087519",
        ),
        SAMLProviderPresetResponse(
            provider=SAMLProvider.PING_IDENTITY,
            name="Ping Identity",
            description="Enterprise identity platform",
            attribute_mapping=SAMLProviderPresets.ping_identity(),
            documentation_url="https://docs.pingidentity.com/",
        ),
        SAMLProviderPresetResponse(
            provider=SAMLProvider.ONELOGIN,
            name="OneLogin",
            description="Cloud-based identity and access management",
            attribute_mapping=SAMLProviderPresets.onelogin(),
            documentation_url="https://developers.onelogin.com/saml",
        ),
        SAMLProviderPresetResponse(
            provider=SAMLProvider.GENERIC,
            name="Generic SAML 2.0",
            description="Any SAML 2.0 compliant identity provider",
            attribute_mapping=SAMLAttributeMapping(),
            documentation_url=None,
        ),
    ]


@router.post("/providers/{provider}/preset", response_model=SAMLProviderPresetResponse)
async def get_provider_preset(
    provider: SAMLProvider,
) -> SAMLProviderPresetResponse:
    """Get attribute mapping preset for a provider.

    Returns the default attribute mapping that can be used as a starting
    point for SAML setup.

    Args:
        provider: SAML provider type

    Returns:
        Provider preset with attribute mapping
    """
    presets = await list_providers()
    for preset in presets:
        if preset.provider == provider:
            return preset

    # Fallback to generic
    return SAMLProviderPresetResponse(
        provider=SAMLProvider.GENERIC,
        name="Generic SAML 2.0",
        description="Any SAML 2.0 compliant identity provider",
        attribute_mapping=SAMLAttributeMapping(),
        documentation_url=None,
    )


@router.post("/parse-idp-metadata", response_model=SAMLIdPMetadataResponse)
async def parse_idp_metadata_endpoint(
    request_body: SAMLIdPMetadataRequest,
) -> SAMLIdPMetadataResponse:
    """Parse IdP metadata and extract configuration.

    Parses IdP metadata from XML or URL and returns the extracted
    configuration values.

    Args:
        request_body: Metadata XML or URL

    Returns:
        Extracted IdP configuration

    Raises:
        HTTPException: If metadata parsing fails
    """
    if not request_body.metadata_xml and not request_body.metadata_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either metadata_xml or metadata_url is required",
        )

    try:
        if request_body.metadata_xml:
            idp_data = parse_idp_metadata(request_body.metadata_xml)
        else:
            idp_data = parse_idp_metadata_url(request_body.metadata_url)  # type: ignore[arg-type]

        # Extract SSO binding URL
        sso_url = None
        sso_service = idp_data.get("singleSignOnService", {})
        if isinstance(sso_service, dict):
            sso_url = sso_service.get("url")

        # Extract SLO binding URL
        slo_url = None
        slo_service = idp_data.get("singleLogoutService", {})
        if isinstance(slo_service, dict):
            slo_url = slo_service.get("url")

        return SAMLIdPMetadataResponse(
            entity_id=idp_data.get("entityId"),
            sso_url=sso_url,
            slo_url=slo_url,
            x509_cert=idp_data.get("x509cert"),
            name_id_format=idp_data.get("NameIDFormat"),
        )

    except SAMLConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
