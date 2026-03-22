"""Multi-tenancy support for ATP Dashboard.

This module provides tenant isolation using schema-per-tenant architecture.
Each tenant gets a dedicated database schema with all tables, providing
strong data isolation while keeping operational costs manageable.

Key Components:
- Tenant: SQLAlchemy model for tenant metadata
- TenantQuotas: Pydantic model for tenant resource limits
- TenantSettings: Pydantic model for tenant configuration
- TenantManager: Manages tenant lifecycle (create, delete, migrate)
- TenantAwareSession: Session wrapper that scopes queries to a tenant

Usage:
    # Create a new tenant
    manager = TenantManager(engine)
    await manager.create_tenant("acme-corp", "Acme Corporation", plan="enterprise")

    # Get a tenant-scoped session
    async with TenantAwareSession(engine, tenant_id="acme-corp") as session:
        users = await session.execute(select(User))  # Only returns tenant's users

    # Delete a tenant (with safety checks)
    await manager.delete_tenant("acme-corp", confirm=True)

    # Run migration for existing deployments
    from atp.dashboard.tenancy import run_tenant_migration
    await run_tenant_migration(engine)
"""

from atp.dashboard.tenancy.manager import (
    TenantDeleteError,
    TenantError,
    TenantExistsError,
    TenantManager,
    TenantNotFoundError,
)
from atp.dashboard.tenancy.migration import (
    create_default_tenant,
    migrate_existing_data_to_default_tenant,
    run_tenant_migration,
)
from atp.dashboard.tenancy.models import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    DEFAULT_TENANT_SCHEMA,
    Tenant,
    TenantQuotas,
    TenantSettings,
)
from atp.dashboard.tenancy.quotas import (
    QuotaChecker,
    QuotaCheckResult,
    QuotaExceededError,
    QuotaType,
    QuotaUsage,
    QuotaUsageTracker,
    QuotaViolation,
)
from atp.dashboard.tenancy.session import (
    TenantAwareSession,
    TenantSessionFactory,
    get_tenant_factory,
    get_tenant_session_factory,
    set_tenant_factory,
)

__all__ = [
    # Models
    "Tenant",
    "TenantQuotas",
    "TenantSettings",
    # Constants
    "DEFAULT_TENANT_ID",
    "DEFAULT_TENANT_NAME",
    "DEFAULT_TENANT_SCHEMA",
    # Manager
    "TenantManager",
    "TenantError",
    "TenantExistsError",
    "TenantNotFoundError",
    "TenantDeleteError",
    # Session
    "TenantAwareSession",
    "TenantSessionFactory",
    "get_tenant_factory",
    "get_tenant_session_factory",
    "set_tenant_factory",
    # Migration
    "create_default_tenant",
    "migrate_existing_data_to_default_tenant",
    "run_tenant_migration",
    # Quotas
    "QuotaType",
    "QuotaViolation",
    "QuotaCheckResult",
    "QuotaUsage",
    "QuotaExceededError",
    "QuotaChecker",
    "QuotaUsageTracker",
]
