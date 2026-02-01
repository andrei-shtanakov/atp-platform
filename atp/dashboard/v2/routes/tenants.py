"""Tenant management routes.

This module provides CRUD operations for managing tenants
in the ATP Dashboard. All endpoints require admin privileges.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from atp.dashboard.database import get_database
from atp.dashboard.tenancy.manager import (
    TenantDeleteError,
    TenantError,
    TenantExistsError,
    TenantManager,
    TenantNotFoundError,
)
from atp.dashboard.tenancy.models import Tenant, TenantQuotas, TenantSettings
from atp.dashboard.tenancy.quotas import QuotaChecker
from atp.dashboard.tenancy.schemas import (
    TenantCreate,
    TenantList,
    TenantQuotaCheck,
    TenantQuotaStatus,
    TenantResponse,
    TenantSummary,
    TenantUpdate,
    TenantUsage,
    TenantUsageResponse,
)
from atp.dashboard.v2.dependencies import AdminUser, DBSession

router = APIRouter(prefix="/tenants", tags=["tenants"])


async def get_tenant_manager() -> TenantManager:
    """Get tenant manager for dependency injection.

    Returns:
        TenantManager instance.
    """
    db = get_database()
    engine: AsyncEngine = db.engine
    return TenantManager(engine)


TenantManagerDep = Annotated[TenantManager, Depends(get_tenant_manager)]


def _tenant_to_response(tenant: Tenant) -> TenantResponse:
    """Convert a Tenant ORM model to a TenantResponse schema.

    Args:
        tenant: The tenant ORM model.

    Returns:
        TenantResponse schema.
    """
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        plan=tenant.plan,
        description=tenant.description,
        quotas=tenant.quotas,
        settings=tenant.settings,
        schema_name=tenant.schema_name,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


def _tenant_to_summary(tenant: Tenant) -> TenantSummary:
    """Convert a Tenant ORM model to a TenantSummary schema.

    Args:
        tenant: The tenant ORM model.

    Returns:
        TenantSummary schema.
    """
    return TenantSummary(
        id=tenant.id,
        name=tenant.name,
        plan=tenant.plan,
        is_active=tenant.is_active,
        created_at=tenant.created_at,
    )


@router.get("", response_model=TenantList)
async def list_tenants(
    session: DBSession,
    user: AdminUser,
    active_only: bool = Query(True, description="Only return active tenants"),
    plan: str | None = Query(
        None, description="Filter by plan (free, pro, enterprise)"
    ),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> TenantList:
    """List all tenants with optional filtering.

    Requires admin privileges.

    Args:
        session: Database session.
        user: Admin user.
        active_only: Only return active tenants.
        plan: Filter by subscription plan.
        limit: Maximum number of tenants to return.
        offset: Number of tenants to skip.

    Returns:
        Paginated list of tenants.
    """
    # Build query for items
    query = select(Tenant)
    count_query = select(func.count(Tenant.id))

    if active_only:
        query = query.where(Tenant.is_active.is_(True))
        count_query = count_query.where(Tenant.is_active.is_(True))

    if plan:
        query = query.where(Tenant.plan == plan)
        count_query = count_query.where(Tenant.plan == plan)

    # Get total count
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated items
    query = query.order_by(Tenant.created_at.desc())
    query = query.limit(limit).offset(offset)

    result = await session.execute(query)
    tenants = result.scalars().all()

    return TenantList(
        total=total,
        items=[_tenant_to_summary(t) for t in tenants],
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_tenant(
    tenant_data: TenantCreate,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantResponse:
    """Create a new tenant.

    Requires admin privileges.

    Args:
        tenant_data: Tenant creation data.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The created tenant.

    Raises:
        HTTPException: If tenant already exists or validation fails.
    """
    try:
        tenant = await manager.create_tenant(
            tenant_id=tenant_data.id,
            name=tenant_data.name,
            plan=tenant_data.plan,
            description=tenant_data.description,
            quotas=tenant_data.quotas,
            settings=tenant_data.settings,
            contact_email=tenant_data.contact_email,
        )
        return _tenant_to_response(tenant)
    except TenantExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except TenantError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tenant: {e}",
        ) from e


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantResponse:
    """Get tenant by ID.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The requested tenant.

    Raises:
        HTTPException: If tenant not found.
    """
    tenant = await manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )
    return _tenant_to_response(tenant)


@router.put("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: str,
    tenant_data: TenantUpdate,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantResponse:
    """Update a tenant.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        tenant_data: Tenant update data.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The updated tenant.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await manager.update_tenant(
            tenant_id=tenant_id,
            name=tenant_data.name,
            plan=tenant_data.plan,
            description=tenant_data.description,
            quotas=tenant_data.quotas,
            settings=tenant_data.settings,
            contact_email=tenant_data.contact_email,
            is_active=tenant_data.is_active,
        )
        return _tenant_to_response(tenant)
    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(
    tenant_id: str,
    manager: TenantManagerDep,
    user: AdminUser,
    hard_delete: bool = Query(
        False, description="Permanently delete tenant and all data"
    ),
) -> None:
    """Delete a tenant.

    Requires admin privileges. By default, performs a soft delete
    (marks tenant as inactive). Use hard_delete=true to permanently
    remove the tenant and all its data.

    Args:
        tenant_id: Tenant ID.
        manager: Tenant manager.
        user: Admin user.
        hard_delete: If true, permanently delete tenant and schema.

    Raises:
        HTTPException: If tenant not found or deletion fails.
    """
    try:
        await manager.delete_tenant(
            tenant_id=tenant_id,
            confirm=True,
            hard_delete=hard_delete,
        )
    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except TenantDeleteError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/{tenant_id}/quotas", response_model=TenantQuotas)
async def get_tenant_quotas(
    tenant_id: str,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantQuotas:
    """Get tenant quota configuration.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The tenant's quotas.

    Raises:
        HTTPException: If tenant not found.
    """
    tenant = await manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )
    return tenant.quotas


@router.put("/{tenant_id}/quotas", response_model=TenantQuotas)
async def update_tenant_quotas(
    tenant_id: str,
    quotas: TenantQuotas,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantQuotas:
    """Update tenant quota configuration.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        quotas: New quota configuration.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The updated quotas.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await manager.update_tenant(
            tenant_id=tenant_id,
            quotas=quotas,
        )
        return tenant.quotas
    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.get("/{tenant_id}/settings", response_model=TenantSettings)
async def get_tenant_settings(
    tenant_id: str,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantSettings:
    """Get tenant settings configuration.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The tenant's settings.

    Raises:
        HTTPException: If tenant not found.
    """
    tenant = await manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )
    return tenant.settings


@router.put("/{tenant_id}/settings", response_model=TenantSettings)
async def update_tenant_settings(
    tenant_id: str,
    settings: TenantSettings,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantSettings:
    """Update tenant settings configuration.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        settings: New settings configuration.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The updated settings.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await manager.update_tenant(
            tenant_id=tenant_id,
            settings=settings,
        )
        return tenant.settings
    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.get("/{tenant_id}/usage", response_model=TenantUsageResponse)
async def get_tenant_usage(
    tenant_id: str,
    session: DBSession,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantUsageResponse:
    """Get tenant usage statistics.

    Returns current resource usage for the tenant along with
    a list of any exceeded quotas.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        session: Database session.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        Tenant usage statistics.

    Raises:
        HTTPException: If tenant not found.
    """
    tenant = await manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    # Use QuotaChecker to get real usage statistics
    checker = QuotaChecker(session)
    quota_usage = await checker.get_usage(tenant_id)

    # Convert to TenantUsage schema
    usage = TenantUsage(
        tenant_id=tenant_id,
        user_count=quota_usage.users,
        agent_count=quota_usage.agents,
        suite_count=quota_usage.suites,
        execution_count=0,  # Not tracked in quotas
        storage_gb=quota_usage.storage_gb,
        tests_today=quota_usage.tests_today,
        llm_cost_this_month=quota_usage.llm_cost_this_month,
    )

    # Check which quotas are exceeded
    check_result = await checker.check_all_quotas(tenant_id)
    quotas_exceeded = [v.quota_type.value for v in check_result.violations]

    return TenantUsageResponse(
        tenant=_tenant_to_response(tenant),
        usage=usage,
        quotas_exceeded=quotas_exceeded,
    )


@router.get("/{tenant_id}/quota-status", response_model=TenantQuotaStatus)
async def get_tenant_quota_status(
    tenant_id: str,
    session: DBSession,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantQuotaStatus:
    """Get detailed quota status for a tenant.

    Returns the current usage percentage for each quota.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        session: Database session.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        Detailed quota status.

    Raises:
        HTTPException: If tenant not found.
    """
    tenant = await manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )

    # Use QuotaChecker to get real usage and quota status
    checker = QuotaChecker(session)
    quota_usage = await checker.get_usage(tenant_id)
    quotas = tenant.quotas

    def _calc_percentage(current: float | int, limit: float | int) -> float:
        """Calculate percentage used."""
        if limit == 0:
            return 100.0 if current > 0 else 0.0
        return min(100.0, (float(current) / float(limit)) * 100)

    checks = [
        TenantQuotaCheck(
            quota_name="max_users",
            current_value=quota_usage.users,
            limit_value=quotas.max_users,
            percentage_used=_calc_percentage(quota_usage.users, quotas.max_users),
            is_exceeded=quota_usage.users > quotas.max_users,
        ),
        TenantQuotaCheck(
            quota_name="max_agents",
            current_value=quota_usage.agents,
            limit_value=quotas.max_agents,
            percentage_used=_calc_percentage(quota_usage.agents, quotas.max_agents),
            is_exceeded=quota_usage.agents > quotas.max_agents,
        ),
        TenantQuotaCheck(
            quota_name="max_suites",
            current_value=quota_usage.suites,
            limit_value=quotas.max_suites,
            percentage_used=_calc_percentage(quota_usage.suites, quotas.max_suites),
            is_exceeded=quota_usage.suites > quotas.max_suites,
        ),
        TenantQuotaCheck(
            quota_name="max_storage_gb",
            current_value=quota_usage.storage_gb,
            limit_value=quotas.max_storage_gb,
            percentage_used=_calc_percentage(
                quota_usage.storage_gb, quotas.max_storage_gb
            ),
            is_exceeded=quota_usage.storage_gb > quotas.max_storage_gb,
        ),
        TenantQuotaCheck(
            quota_name="max_tests_per_day",
            current_value=quota_usage.tests_today,
            limit_value=quotas.max_tests_per_day,
            percentage_used=_calc_percentage(
                quota_usage.tests_today, quotas.max_tests_per_day
            ),
            is_exceeded=quota_usage.tests_today > quotas.max_tests_per_day,
        ),
        TenantQuotaCheck(
            quota_name="llm_budget_monthly",
            current_value=quota_usage.llm_cost_this_month,
            limit_value=quotas.llm_budget_monthly,
            percentage_used=_calc_percentage(
                quota_usage.llm_cost_this_month, quotas.llm_budget_monthly
            ),
            is_exceeded=quota_usage.llm_cost_this_month > quotas.llm_budget_monthly,
        ),
        TenantQuotaCheck(
            quota_name="max_parallel_runs",
            current_value=quota_usage.parallel_runs,
            limit_value=quotas.max_parallel_runs,
            percentage_used=_calc_percentage(
                quota_usage.parallel_runs, quotas.max_parallel_runs
            ),
            is_exceeded=quota_usage.parallel_runs > quotas.max_parallel_runs,
        ),
    ]

    return TenantQuotaStatus(
        tenant_id=tenant_id,
        checks=checks,
        any_exceeded=any(check.is_exceeded for check in checks),
    )


@router.post("/{tenant_id}/activate", response_model=TenantResponse)
async def activate_tenant(
    tenant_id: str,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantResponse:
    """Activate a deactivated tenant.

    Requires admin privileges.

    Args:
        tenant_id: Tenant ID.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The activated tenant.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await manager.update_tenant(
            tenant_id=tenant_id,
            is_active=True,
        )
        return _tenant_to_response(tenant)
    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/{tenant_id}/deactivate", response_model=TenantResponse)
async def deactivate_tenant(
    tenant_id: str,
    manager: TenantManagerDep,
    user: AdminUser,
) -> TenantResponse:
    """Deactivate a tenant (soft delete).

    Requires admin privileges. This is a reversible operation
    that disables the tenant without deleting data.

    Args:
        tenant_id: Tenant ID.
        manager: Tenant manager.
        user: Admin user.

    Returns:
        The deactivated tenant.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await manager.update_tenant(
            tenant_id=tenant_id,
            is_active=False,
        )
        return _tenant_to_response(tenant)
    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
