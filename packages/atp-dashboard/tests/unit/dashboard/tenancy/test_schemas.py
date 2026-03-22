"""Tests for tenant Pydantic schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from atp.dashboard.tenancy.models import TenantQuotas, TenantSettings
from atp.dashboard.tenancy.schemas import (
    TenantCreate,
    TenantList,
    TenantQuotaCheck,
    TenantQuotaStatus,
    TenantResponse,
    TenantSummary,
    TenantUpdate,
    TenantUsage,
)


class TestTenantCreate:
    """Tests for TenantCreate schema."""

    def test_valid_creation(self) -> None:
        """Test creating with valid data."""
        tenant = TenantCreate(
            id="acme-corp",
            name="Acme Corporation",
            plan="enterprise",
            description="Test tenant",
        )
        assert tenant.id == "acme-corp"
        assert tenant.name == "Acme Corporation"
        assert tenant.plan == "enterprise"

    def test_minimal_creation(self) -> None:
        """Test creating with minimal data."""
        tenant = TenantCreate(id="acme", name="Acme")
        assert tenant.id == "acme"
        assert tenant.plan == "free"  # default
        assert tenant.quotas is not None
        assert tenant.settings is not None

    def test_custom_quotas(self) -> None:
        """Test creating with custom quotas."""
        quotas = TenantQuotas(max_tests_per_day=500)
        tenant = TenantCreate(id="acme", name="Acme", quotas=quotas)
        assert tenant.quotas.max_tests_per_day == 500

    def test_custom_settings(self) -> None:
        """Test creating with custom settings."""
        settings = TenantSettings(require_mfa=True)
        tenant = TenantCreate(id="acme", name="Acme", settings=settings)
        assert tenant.settings.require_mfa is True

    def test_invalid_id_empty(self) -> None:
        """Test validation rejects empty ID."""
        with pytest.raises(ValidationError):
            TenantCreate(id="", name="Acme")

    def test_invalid_id_uppercase(self) -> None:
        """Test validation rejects uppercase ID."""
        with pytest.raises(ValidationError):
            TenantCreate(id="Acme", name="Acme")

    def test_invalid_id_special_chars(self) -> None:
        """Test validation rejects special characters."""
        with pytest.raises(ValidationError):
            TenantCreate(id="acme@corp", name="Acme")

    def test_invalid_plan(self) -> None:
        """Test validation rejects invalid plan."""
        with pytest.raises(ValidationError):
            TenantCreate(id="acme", name="Acme", plan="invalid")

    def test_invalid_name_empty(self) -> None:
        """Test validation rejects empty name."""
        with pytest.raises(ValidationError):
            TenantCreate(id="acme", name="")


class TestTenantUpdate:
    """Tests for TenantUpdate schema."""

    def test_partial_update(self) -> None:
        """Test partial update."""
        update = TenantUpdate(name="New Name")
        assert update.name == "New Name"
        assert update.plan is None
        assert update.is_active is None

    def test_full_update(self) -> None:
        """Test full update."""
        update = TenantUpdate(
            name="New Name",
            plan="enterprise",
            description="Updated",
            is_active=False,
        )
        assert update.name == "New Name"
        assert update.plan == "enterprise"
        assert update.is_active is False

    def test_update_quotas(self) -> None:
        """Test updating quotas."""
        quotas = TenantQuotas(max_tests_per_day=1000)
        update = TenantUpdate(quotas=quotas)
        assert update.quotas is not None
        assert update.quotas.max_tests_per_day == 1000

    def test_invalid_plan(self) -> None:
        """Test validation rejects invalid plan."""
        with pytest.raises(ValidationError):
            TenantUpdate(plan="invalid")


class TestTenantResponse:
    """Tests for TenantResponse schema."""

    def test_from_orm(self) -> None:
        """Test creating from ORM model."""
        from atp.dashboard.tenancy.models import Tenant

        tenant = Tenant(
            id="acme",
            name="Acme",
            plan="enterprise",
            description="Test",
            quotas_json=TenantQuotas().model_dump(),
            settings_json=TenantSettings().model_dump(),
            schema_name="tenant_acme",
            is_active=True,
            contact_email="admin@acme.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        response = TenantResponse.model_validate(tenant)
        assert response.id == "acme"
        assert response.name == "Acme"
        assert response.quotas.max_tests_per_day == 100
        assert response.settings.require_mfa is False


class TestTenantSummary:
    """Tests for TenantSummary schema."""

    def test_from_orm(self) -> None:
        """Test creating from ORM model."""
        from atp.dashboard.tenancy.models import Tenant

        tenant = Tenant(
            id="acme",
            name="Acme",
            plan="pro",
            schema_name="tenant_acme",
            is_active=True,
            created_at=datetime.now(),
        )

        summary = TenantSummary.model_validate(tenant)
        assert summary.id == "acme"
        assert summary.name == "Acme"
        assert summary.plan == "pro"
        assert summary.is_active is True


class TestTenantList:
    """Tests for TenantList schema."""

    def test_empty_list(self) -> None:
        """Test empty list."""
        tenant_list = TenantList(
            total=0,
            items=[],
            limit=10,
            offset=0,
        )
        assert tenant_list.total == 0
        assert len(tenant_list.items) == 0

    def test_with_items(self) -> None:
        """Test list with items."""
        items = [
            TenantSummary(
                id="acme1",
                name="Acme 1",
                plan="free",
                is_active=True,
                created_at=datetime.now(),
            ),
            TenantSummary(
                id="acme2",
                name="Acme 2",
                plan="pro",
                is_active=True,
                created_at=datetime.now(),
            ),
        ]
        tenant_list = TenantList(
            total=2,
            items=items,
            limit=10,
            offset=0,
        )
        assert tenant_list.total == 2
        assert len(tenant_list.items) == 2


class TestTenantUsage:
    """Tests for TenantUsage schema."""

    def test_usage_creation(self) -> None:
        """Test creating usage statistics."""
        usage = TenantUsage(
            tenant_id="acme",
            user_count=5,
            agent_count=10,
            suite_count=3,
            execution_count=100,
            storage_gb=2.5,
            tests_today=50,
            llm_cost_this_month=25.50,
        )
        assert usage.tenant_id == "acme"
        assert usage.user_count == 5
        assert usage.llm_cost_this_month == 25.50


class TestTenantQuotaStatus:
    """Tests for quota status schemas."""

    def test_quota_check(self) -> None:
        """Test quota check result."""
        check = TenantQuotaCheck(
            quota_name="max_tests_per_day",
            current_value=80,
            limit_value=100,
            percentage_used=80.0,
            is_exceeded=False,
        )
        assert check.quota_name == "max_tests_per_day"
        assert check.percentage_used == 80.0
        assert check.is_exceeded is False

    def test_quota_status(self) -> None:
        """Test quota status."""
        checks = [
            TenantQuotaCheck(
                quota_name="max_tests_per_day",
                current_value=80,
                limit_value=100,
                percentage_used=80.0,
                is_exceeded=False,
            ),
            TenantQuotaCheck(
                quota_name="max_agents",
                current_value=15,
                limit_value=10,
                percentage_used=150.0,
                is_exceeded=True,
            ),
        ]
        status = TenantQuotaStatus(
            tenant_id="acme",
            checks=checks,
            any_exceeded=True,
        )
        assert status.tenant_id == "acme"
        assert status.any_exceeded is True
        assert len(status.checks) == 2
