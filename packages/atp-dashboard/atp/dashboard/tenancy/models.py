"""Tenant models for multi-tenancy support.

This module defines the Tenant model and associated Pydantic schemas
for tenant quotas and settings.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from atp.dashboard.models import Base


class TenantQuotas(BaseModel):
    """Resource quotas for a tenant.

    Defines limits on various resources that a tenant can use.
    """

    model_config = ConfigDict(extra="forbid")

    max_tests_per_day: int = Field(
        default=100, ge=0, description="Maximum tests that can be run per day"
    )
    max_parallel_runs: int = Field(
        default=5, ge=1, description="Maximum parallel test runs"
    )
    max_storage_gb: float = Field(
        default=10.0, ge=0, description="Maximum storage in GB"
    )
    max_agents: int = Field(default=10, ge=1, description="Maximum number of agents")
    llm_budget_monthly: float = Field(
        default=100.00,
        ge=0,
        description="Monthly LLM budget in USD",
    )
    max_users: int = Field(default=10, ge=1, description="Maximum number of users")
    max_suites: int = Field(
        default=50, ge=1, description="Maximum number of test suites"
    )


class TenantSettings(BaseModel):
    """Tenant-specific settings and configuration.

    Contains customizable settings for the tenant's environment.
    """

    model_config = ConfigDict(extra="allow")

    default_timeout_seconds: int = Field(
        default=300, ge=1, description="Default timeout for tests"
    )
    allow_external_agents: bool = Field(
        default=True, description="Allow connecting external agents"
    )
    require_mfa: bool = Field(default=False, description="Require MFA for all users")
    sso_enabled: bool = Field(default=False, description="Enable SSO integration")
    sso_provider: str | None = Field(default=None, description="SSO provider name")
    sso_config: dict[str, Any] = Field(
        default_factory=dict, description="SSO configuration"
    )
    custom_branding: dict[str, Any] = Field(
        default_factory=dict, description="Custom branding settings"
    )
    notification_channels: list[str] = Field(
        default_factory=lambda: ["email"], description="Enabled notification channels"
    )
    retention_days: int = Field(
        default=90, ge=1, description="Data retention period in days"
    )


class Tenant(Base):
    """Tenant model for multi-tenancy support.

    Each tenant represents an isolated organization with its own
    users, agents, test suites, and results.
    """

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(
        String(50), default="free"
    )  # free, pro, enterprise
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Quotas and settings stored as JSON
    quotas_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    settings_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Schema name for this tenant (for schema-per-tenant isolation)
    schema_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Status and lifecycle
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    # Optional contact info
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_tenant_name", "name"),
        Index("idx_tenant_plan", "plan"),
        Index("idx_tenant_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"Tenant(id={self.id!r}, name={self.name!r}, plan={self.plan!r})"

    @property
    def quotas(self) -> TenantQuotas:
        """Get tenant quotas as a Pydantic model."""
        return TenantQuotas(**(self.quotas_json or {}))

    @quotas.setter
    def quotas(self, value: TenantQuotas) -> None:
        """Set tenant quotas from a Pydantic model."""
        self.quotas_json = value.model_dump()

    @property
    def settings(self) -> TenantSettings:
        """Get tenant settings as a Pydantic model."""
        return TenantSettings(**(self.settings_json or {}))

    @settings.setter
    def settings(self, value: TenantSettings) -> None:
        """Set tenant settings from a Pydantic model."""
        self.settings_json = value.model_dump()


# Default tenant ID for single-tenant deployments and migration
DEFAULT_TENANT_ID = "default"
DEFAULT_TENANT_NAME = "Default Tenant"
DEFAULT_TENANT_SCHEMA = "public"  # Use public schema for default tenant
