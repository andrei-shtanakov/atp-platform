"""Tests for SSO routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atp.dashboard.auth.sso.oidc import (
    OIDCProvider,
    SSOConfig,
    SSOUserInfo,
)
from atp.dashboard.v2.routes.sso import (
    ProviderPresetResponse,
    SSOConfigCreate,
    SSOConfigResponse,
    SSOInitRequest,
    SSOInitResponse,
    router,
)


@pytest.fixture
def app() -> FastAPI:
    """Create test app with SSO routes."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestSSOInitRequest:
    """Tests for SSOInitRequest model."""

    def test_default_values(self) -> None:
        """Test default values."""
        request = SSOInitRequest()
        assert request.tenant_id == "default"
        assert request.return_url is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        request = SSOInitRequest(
            tenant_id="custom-tenant",
            return_url="https://example.com/dashboard",
        )
        assert request.tenant_id == "custom-tenant"
        assert request.return_url == "https://example.com/dashboard"


class TestSSOInitResponse:
    """Tests for SSOInitResponse model."""

    def test_create_response(self) -> None:
        """Test creating response."""
        response = SSOInitResponse(
            authorization_url="https://idp.example.com/authorize?...",
            state="random-state",
        )
        assert "authorize" in response.authorization_url
        assert response.state == "random-state"


class TestSSOConfigCreate:
    """Tests for SSOConfigCreate model."""

    def test_create_config(self) -> None:
        """Test creating SSO config."""
        config = SSOConfigCreate(
            provider=OIDCProvider.OKTA,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://example.okta.com",
            redirect_uri="https://app.example.com/callback",
        )
        assert config.provider == OIDCProvider.OKTA
        assert config.client_id == "test-client"
        assert config.default_role == "viewer"

    def test_default_scopes(self) -> None:
        """Test default scopes."""
        config = SSOConfigCreate(
            provider=OIDCProvider.GENERIC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            redirect_uri="https://app.example.com/callback",
        )
        assert "openid" in config.scopes
        assert "profile" in config.scopes
        assert "email" in config.scopes


class TestSSOConfigResponse:
    """Tests for SSOConfigResponse model."""

    def test_disabled_config(self) -> None:
        """Test disabled SSO config response."""
        response = SSOConfigResponse(enabled=False)
        assert response.enabled is False
        assert response.provider is None
        assert response.issuer_url is None

    def test_enabled_config(self) -> None:
        """Test enabled SSO config response."""
        response = SSOConfigResponse(
            enabled=True,
            provider=OIDCProvider.OKTA,
            issuer_url="https://example.okta.com",
            scopes=["openid", "profile", "email"],
            default_role="viewer",
        )
        assert response.enabled is True
        assert response.provider == OIDCProvider.OKTA


class TestProviderPresetResponse:
    """Tests for ProviderPresetResponse model."""

    def test_create_preset_response(self) -> None:
        """Test creating preset response."""
        response = ProviderPresetResponse(
            provider=OIDCProvider.OKTA,
            name="Okta",
            description="Enterprise identity management",
            required_fields=["domain", "client_id", "client_secret"],
            optional_fields=["authorization_server_id"],
            documentation_url="https://developer.okta.com/docs",
        )
        assert response.provider == OIDCProvider.OKTA
        assert response.name == "Okta"
        assert "domain" in response.required_fields


class TestListProvidersEndpoint:
    """Tests for list providers endpoint."""

    def test_list_providers(self, client: TestClient) -> None:
        """Test listing available providers."""
        response = client.get("/sso/providers")
        assert response.status_code == 200
        data = response.json()

        assert len(data) == 5  # Okta, Auth0, Azure AD, Google, Generic

        provider_names = [p["provider"] for p in data]
        assert "okta" in provider_names
        assert "auth0" in provider_names
        assert "azure_ad" in provider_names
        assert "google" in provider_names
        assert "generic" in provider_names


class TestProviderPresetEndpoint:
    """Tests for provider preset endpoint."""

    def test_okta_preset(self, client: TestClient) -> None:
        """Test getting Okta preset."""
        response = client.post(
            "/sso/providers/okta/preset",
            params={"domain": "example.okta.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "okta"
        assert "oauth2" in data["issuer_url"]

    def test_okta_preset_missing_domain(self, client: TestClient) -> None:
        """Test Okta preset without domain."""
        response = client.post("/sso/providers/okta/preset")
        assert response.status_code == 400
        assert "Domain required" in response.json()["detail"]

    def test_auth0_preset(self, client: TestClient) -> None:
        """Test getting Auth0 preset."""
        response = client.post(
            "/sso/providers/auth0/preset",
            params={"domain": "example.auth0.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "auth0"

    def test_azure_ad_preset(self, client: TestClient) -> None:
        """Test getting Azure AD preset."""
        response = client.post(
            "/sso/providers/azure_ad/preset",
            params={"tenant_id": "test-tenant-id"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "azure_ad"
        assert "test-tenant-id" in data["issuer_url"]

    def test_azure_ad_preset_missing_tenant_id(self, client: TestClient) -> None:
        """Test Azure AD preset without tenant ID."""
        response = client.post("/sso/providers/azure_ad/preset")
        assert response.status_code == 400
        assert "Tenant ID required" in response.json()["detail"]

    def test_google_preset(self, client: TestClient) -> None:
        """Test getting Google preset."""
        response = client.post("/sso/providers/google/preset")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "google"
        assert "accounts.google.com" in data["issuer_url"]

    def test_generic_preset(self, client: TestClient) -> None:
        """Test getting generic preset."""
        response = client.post("/sso/providers/generic/preset")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "generic"


class TestSSOFlowIntegration:
    """Integration-style tests for SSO flow."""

    @pytest.fixture
    def mock_tenant(self) -> MagicMock:
        """Create mock tenant."""
        tenant = MagicMock()
        tenant.id = "test-tenant"
        tenant.settings = MagicMock()
        tenant.settings.sso_enabled = True
        tenant.settings.sso_config = {
            "provider": "generic",
            "client_id": "test-client",
            "client_secret": "test-secret",
            "issuer_url": "https://idp.example.com",
            "redirect_uri": "https://app.example.com/callback",
        }
        return tenant

    @pytest.fixture
    def mock_sso_config(self) -> SSOConfig:
        """Create mock SSO config."""
        return SSOConfig(
            provider=OIDCProvider.GENERIC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            redirect_uri="https://app.example.com/callback",
        )

    @pytest.fixture
    def mock_user_info(self) -> SSOUserInfo:
        """Create mock user info."""
        return SSOUserInfo(
            sub="user-123",
            email="user@example.com",
            email_verified=True,
            name="Test User",
            groups=["developers"],
        )

    def test_sso_init_tenant_not_found(
        self,
        client: TestClient,
    ) -> None:
        """Test SSO init with non-existent tenant."""
        with patch(
            "atp.dashboard.v2.routes.sso._get_tenant_sso_config",
            new_callable=AsyncMock,
            return_value=(None, None),
        ):
            response = client.post(
                "/sso/init",
                json={"tenant_id": "nonexistent"},
            )
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    def test_sso_init_sso_not_configured(
        self,
        client: TestClient,
        mock_tenant: MagicMock,
    ) -> None:
        """Test SSO init when SSO is not configured."""
        mock_tenant.settings.sso_enabled = False
        with patch(
            "atp.dashboard.v2.routes.sso._get_tenant_sso_config",
            new_callable=AsyncMock,
            return_value=(mock_tenant, None),
        ):
            response = client.post(
                "/sso/init",
                json={"tenant_id": "test-tenant"},
            )
            assert response.status_code == 400
            assert "not configured" in response.json()["detail"]

    def test_sso_callback_invalid_state(
        self,
        client: TestClient,
    ) -> None:
        """Test SSO callback with invalid state."""
        response = client.get(
            "/sso/callback",
            params={"code": "auth-code", "state": "invalid-state"},
        )
        assert response.status_code == 400
        assert "Invalid or expired state" in response.json()["detail"]

    def test_sso_callback_error_from_idp(
        self,
        client: TestClient,
    ) -> None:
        """Test SSO callback with error from IdP."""
        response = client.get(
            "/sso/callback",
            params={
                "code": "auth-code",
                "state": "state",
                "error": "access_denied",
                "error_description": "User cancelled",
            },
        )
        assert response.status_code == 400
        assert "access_denied" in response.json()["detail"]

    def test_get_sso_config_tenant_not_found(
        self,
        client: TestClient,
    ) -> None:
        """Test getting SSO config for non-existent tenant."""
        with patch(
            "atp.dashboard.v2.routes.sso._get_tenant_sso_config",
            new_callable=AsyncMock,
            return_value=(None, None),
        ):
            response = client.get("/sso/config", params={"tenant_id": "nonexistent"})
            assert response.status_code == 404

    def test_get_sso_config_sso_disabled(
        self,
        client: TestClient,
        mock_tenant: MagicMock,
    ) -> None:
        """Test getting SSO config when SSO is disabled."""
        with patch(
            "atp.dashboard.v2.routes.sso._get_tenant_sso_config",
            new_callable=AsyncMock,
            return_value=(mock_tenant, None),
        ):
            response = client.get("/sso/config", params={"tenant_id": "test-tenant"})
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is False

    def test_get_sso_config_enabled(
        self,
        client: TestClient,
        mock_tenant: MagicMock,
        mock_sso_config: SSOConfig,
    ) -> None:
        """Test getting SSO config when SSO is enabled."""
        with patch(
            "atp.dashboard.v2.routes.sso._get_tenant_sso_config",
            new_callable=AsyncMock,
            return_value=(mock_tenant, mock_sso_config),
        ):
            response = client.get("/sso/config", params={"tenant_id": "test-tenant"})
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert data["provider"] == "generic"
            assert data["issuer_url"] == "https://idp.example.com"
