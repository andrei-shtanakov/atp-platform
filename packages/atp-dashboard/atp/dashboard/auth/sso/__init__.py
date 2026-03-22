"""SSO (Single Sign-On) module for ATP Dashboard.

This module provides SSO support with:
- OIDC (OpenID Connect) for Okta, Auth0, Azure AD, Google Workspace
- SAML 2.0 for enterprise identity providers
- Token/assertion validation
- Just-In-Time (JIT) user provisioning
- Group-to-role mapping

Usage (OIDC):
    from atp.dashboard.auth.sso import SSOManager, SSOConfig

    config = SSOConfig(
        provider="okta",
        client_id="your-client-id",
        client_secret="your-client-secret",
        issuer_url="https://your-domain.okta.com",
    )

    sso_manager = SSOManager(config)
    auth_url = await sso_manager.get_authorization_url(state="random-state")

Usage (SAML):
    from atp.dashboard.auth.sso import SAMLManager, SAMLConfig

    config = SAMLConfig(
        sp_entity_id="https://your-app.com/saml/metadata",
        sp_acs_url="https://your-app.com/saml/acs",
        idp_entity_id="https://idp.example.com",
        idp_sso_url="https://idp.example.com/saml/sso",
        idp_x509_cert="...",
    )

    saml_manager = SAMLManager(config)
    auth_url = saml_manager.get_authn_request_url(request_url)
"""

from atp.dashboard.auth.sso.oidc import (
    GroupRoleMapping,
    OIDCProvider,
    ProviderPresets,
    SSOConfig,
    SSOManager,
    SSOUserInfo,
)
from atp.dashboard.auth.sso.saml import (
    SAMLAttributeMapping,
    SAMLConfig,
    SAMLGroupRoleMapping,
    SAMLManager,
    SAMLNameIDFormat,
    SAMLProvider,
    SAMLProviderPresets,
    SAMLUserInfo,
)

__all__ = [
    # OIDC exports
    "OIDCProvider",
    "SSOConfig",
    "SSOManager",
    "SSOUserInfo",
    "GroupRoleMapping",
    "ProviderPresets",
    # SAML exports
    "SAMLProvider",
    "SAMLConfig",
    "SAMLManager",
    "SAMLUserInfo",
    "SAMLGroupRoleMapping",
    "SAMLAttributeMapping",
    "SAMLNameIDFormat",
    "SAMLProviderPresets",
]
