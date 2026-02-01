"""Pydantic schemas for tenant API request/response models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from atp.dashboard.tenancy.models import TenantQuotas, TenantSettings


class TenantCreate(BaseModel):
    """Schema for creating a tenant."""

    id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$",
        description="Unique tenant ID (lowercase letters, numbers, hyphens)",
    )
    name: str = Field(
        ..., min_length=1, max_length=255, description="Display name for the tenant"
    )
    plan: str = Field(
        default="free",
        pattern=r"^(free|pro|enterprise)$",
        description="Subscription plan",
    )
    description: str | None = Field(None, max_length=2000, description="Description")
    quotas: TenantQuotas = Field(
        default_factory=TenantQuotas, description="Resource quotas"
    )
    settings: TenantSettings = Field(
        default_factory=TenantSettings, description="Tenant settings"
    )
    contact_email: str | None = Field(None, max_length=255, description="Contact email")


class TenantUpdate(BaseModel):
    """Schema for updating a tenant."""

    name: str | None = Field(None, min_length=1, max_length=255)
    plan: str | None = Field(None, pattern=r"^(free|pro|enterprise)$")
    description: str | None = Field(None, max_length=2000)
    quotas: TenantQuotas | None = None
    settings: TenantSettings | None = None
    contact_email: str | None = None
    is_active: bool | None = None


class TenantResponse(BaseModel):
    """Schema for tenant response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    plan: str
    description: str | None
    quotas: TenantQuotas
    settings: TenantSettings
    schema_name: str
    is_active: bool
    contact_email: str | None
    created_at: datetime
    updated_at: datetime


class TenantSummary(BaseModel):
    """Summary of a tenant for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    plan: str
    is_active: bool
    created_at: datetime


class TenantList(BaseModel):
    """Paginated list of tenants."""

    total: int
    items: list[TenantSummary]
    limit: int
    offset: int


class TenantUsage(BaseModel):
    """Current usage statistics for a tenant."""

    tenant_id: str
    user_count: int
    agent_count: int
    suite_count: int
    execution_count: int
    storage_gb: float
    tests_today: int
    llm_cost_this_month: float


class TenantUsageResponse(BaseModel):
    """Tenant with usage statistics."""

    tenant: TenantResponse
    usage: TenantUsage
    quotas_exceeded: list[str] = Field(
        default_factory=list, description="List of exceeded quota names"
    )


class TenantQuotaCheck(BaseModel):
    """Result of a quota check."""

    quota_name: str
    current_value: float | int
    limit_value: float | int
    percentage_used: float
    is_exceeded: bool


class TenantQuotaStatus(BaseModel):
    """Status of all quotas for a tenant."""

    tenant_id: str
    checks: list[TenantQuotaCheck]
    any_exceeded: bool
