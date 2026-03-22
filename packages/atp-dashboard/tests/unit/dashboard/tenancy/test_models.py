"""Tests for tenant models."""

import pytest

from atp.dashboard.tenancy.models import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    DEFAULT_TENANT_SCHEMA,
    Tenant,
    TenantQuotas,
    TenantSettings,
)


class TestTenantQuotas:
    """Tests for TenantQuotas model."""

    def test_default_values(self) -> None:
        """Test default quota values."""
        quotas = TenantQuotas()
        assert quotas.max_tests_per_day == 100
        assert quotas.max_parallel_runs == 5
        assert quotas.max_storage_gb == 10.0
        assert quotas.max_agents == 10
        assert quotas.llm_budget_monthly == 100.00
        assert quotas.max_users == 10
        assert quotas.max_suites == 50

    def test_custom_values(self) -> None:
        """Test custom quota values."""
        quotas = TenantQuotas(
            max_tests_per_day=500,
            max_parallel_runs=20,
            max_storage_gb=50.0,
            max_agents=50,
            llm_budget_monthly=500.00,
            max_users=100,
            max_suites=200,
        )
        assert quotas.max_tests_per_day == 500
        assert quotas.max_parallel_runs == 20
        assert quotas.max_storage_gb == 50.0
        assert quotas.max_agents == 50
        assert quotas.llm_budget_monthly == 500.00

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        quotas = TenantQuotas(max_tests_per_day=200)
        data = quotas.model_dump()
        assert data["max_tests_per_day"] == 200
        assert "max_parallel_runs" in data

    def test_validation_min_values(self) -> None:
        """Test validation of minimum values."""
        # max_parallel_runs must be >= 1
        with pytest.raises(ValueError):
            TenantQuotas(max_parallel_runs=0)

        # max_tests_per_day can be 0 (effectively disabled)
        quotas = TenantQuotas(max_tests_per_day=0)
        assert quotas.max_tests_per_day == 0


class TestTenantSettings:
    """Tests for TenantSettings model."""

    def test_default_values(self) -> None:
        """Test default settings values."""
        settings = TenantSettings()
        assert settings.default_timeout_seconds == 300
        assert settings.allow_external_agents is True
        assert settings.require_mfa is False
        assert settings.sso_enabled is False
        assert settings.sso_provider is None
        assert settings.sso_config == {}
        assert settings.custom_branding == {}
        assert settings.notification_channels == ["email"]
        assert settings.retention_days == 90

    def test_custom_values(self) -> None:
        """Test custom settings values."""
        settings = TenantSettings(
            default_timeout_seconds=600,
            require_mfa=True,
            sso_enabled=True,
            sso_provider="okta",
            sso_config={"client_id": "test"},
            retention_days=365,
        )
        assert settings.default_timeout_seconds == 600
        assert settings.require_mfa is True
        assert settings.sso_enabled is True
        assert settings.sso_provider == "okta"
        assert settings.sso_config == {"client_id": "test"}

    def test_model_allows_extra_fields(self) -> None:
        """Test that extra fields are allowed for extensibility."""
        settings = TenantSettings(custom_field="value")
        assert settings.custom_field == "value"


class TestTenant:
    """Tests for Tenant model."""

    def test_create_tenant(self) -> None:
        """Test creating a tenant."""
        tenant = Tenant(
            id="acme-corp",
            name="Acme Corporation",
            plan="enterprise",
            description="Test tenant",
            quotas_json=TenantQuotas().model_dump(),
            settings_json=TenantSettings().model_dump(),
            schema_name="tenant_acme_corp",
            is_active=True,
        )
        assert tenant.id == "acme-corp"
        assert tenant.name == "Acme Corporation"
        assert tenant.plan == "enterprise"
        assert tenant.schema_name == "tenant_acme_corp"
        assert tenant.is_active is True

    def test_tenant_quotas_property(self) -> None:
        """Test quotas property getter."""
        tenant = Tenant(
            id="test",
            name="Test",
            schema_name="public",
            quotas_json={"max_tests_per_day": 500},
            settings_json={},
        )
        quotas = tenant.quotas
        assert isinstance(quotas, TenantQuotas)
        assert quotas.max_tests_per_day == 500

    def test_tenant_quotas_setter(self) -> None:
        """Test quotas property setter."""
        tenant = Tenant(
            id="test",
            name="Test",
            schema_name="public",
            quotas_json={},
            settings_json={},
        )
        new_quotas = TenantQuotas(max_tests_per_day=1000)
        tenant.quotas = new_quotas
        assert tenant.quotas_json["max_tests_per_day"] == 1000

    def test_tenant_settings_property(self) -> None:
        """Test settings property getter."""
        tenant = Tenant(
            id="test",
            name="Test",
            schema_name="public",
            quotas_json={},
            settings_json={"require_mfa": True},
        )
        settings = tenant.settings
        assert isinstance(settings, TenantSettings)
        assert settings.require_mfa is True

    def test_tenant_settings_setter(self) -> None:
        """Test settings property setter."""
        tenant = Tenant(
            id="test",
            name="Test",
            schema_name="public",
            quotas_json={},
            settings_json={},
        )
        new_settings = TenantSettings(require_mfa=True)
        tenant.settings = new_settings
        assert tenant.settings_json["require_mfa"] is True

    def test_tenant_repr(self) -> None:
        """Test tenant string representation."""
        tenant = Tenant(
            id="test",
            name="Test Tenant",
            plan="pro",
            schema_name="public",
        )
        repr_str = repr(tenant)
        assert "test" in repr_str
        assert "Test Tenant" in repr_str
        assert "pro" in repr_str


class TestConstants:
    """Tests for module constants."""

    def test_default_tenant_id(self) -> None:
        """Test default tenant ID constant."""
        assert DEFAULT_TENANT_ID == "default"

    def test_default_tenant_name(self) -> None:
        """Test default tenant name constant."""
        assert DEFAULT_TENANT_NAME == "Default Tenant"

    def test_default_tenant_schema(self) -> None:
        """Test default tenant schema constant."""
        assert DEFAULT_TENANT_SCHEMA == "public"
