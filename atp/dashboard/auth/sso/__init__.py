"""SSO (Single Sign-On) module for ATP Dashboard.

This module provides OIDC-based SSO support with:
- Popular provider configurations (Okta, Auth0, Azure AD, Google Workspace)
- Token validation and user claims extraction
- Just-In-Time (JIT) user provisioning
- Group-to-role mapping

Usage:
    from atp.dashboard.auth.sso import SSOManager, SSOConfig

    # Configure SSO for a tenant
    config = SSOConfig(
        provider="okta",
        client_id="your-client-id",
        client_secret="your-client-secret",
        issuer_url="https://your-domain.okta.com",
    )

    sso_manager = SSOManager(config)
    auth_url = await sso_manager.get_authorization_url(state="random-state")
"""

from atp.dashboard.auth.sso.oidc import (
    GroupRoleMapping,
    OIDCProvider,
    ProviderPresets,
    SSOConfig,
    SSOManager,
    SSOUserInfo,
)

__all__ = [
    "OIDCProvider",
    "SSOConfig",
    "SSOManager",
    "SSOUserInfo",
    "GroupRoleMapping",
    "ProviderPresets",
]
