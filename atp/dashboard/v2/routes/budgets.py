"""Budget management routes.

This module provides endpoints for creating, updating, and monitoring
cost budgets with alert thresholds.

Permissions:
    - GET /budgets: BUDGETS_READ
    - GET /budgets/{id}: BUDGETS_READ
    - POST /budgets: BUDGETS_WRITE
    - PUT /budgets/{id}: BUDGETS_WRITE
    - DELETE /budgets/{id}: BUDGETS_DELETE
    - GET /budgets/{id}/usage: BUDGETS_READ
    - GET /budgets/status/all: BUDGETS_READ
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from atp.analytics.database import AnalyticsDatabase
from atp.analytics.repository import CostRepository
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    BudgetCreate,
    BudgetList,
    BudgetResponse,
    BudgetUpdate,
    BudgetUsageResponse,
    BudgetWithUsageResponse,
)

router = APIRouter(prefix="/budgets", tags=["budgets"])


def _budget_to_response(budget) -> BudgetResponse:
    """Convert a CostBudget model to BudgetResponse."""
    return BudgetResponse(
        id=budget.id,
        name=budget.name,
        period=budget.period,
        limit_usd=float(budget.limit_usd),
        alert_threshold=budget.alert_threshold,
        scope=budget.scope_json,
        alert_channels=budget.alert_channels_json,
        description=budget.description,
        is_active=budget.is_active,
        created_at=budget.created_at,
        updated_at=budget.updated_at,
    )


def _usage_to_response(usage: dict) -> BudgetUsageResponse:
    """Convert a budget usage dict to BudgetUsageResponse."""
    return BudgetUsageResponse(
        budget_id=usage["budget_id"],
        budget_name=usage["budget_name"],
        period=usage["period"],
        period_start=usage["period_start"],
        spent=float(usage["spent"]),
        limit=float(usage["limit"]),
        remaining=float(usage["remaining"]),
        percentage=usage["percentage"],
        is_over_threshold=usage["is_over_threshold"],
        is_over_limit=usage["is_over_limit"],
    )


@router.get("", response_model=BudgetList)
async def list_budgets(
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_READ))],
    period: str | None = Query(
        None, pattern="^(daily|weekly|monthly)$", description="Filter by period"
    ),
    is_active: bool | None = Query(None, description="Filter by active status"),
    include_usage: bool = Query(
        True, description="Include current usage for each budget"
    ),
) -> BudgetList:
    """List all budgets with optional filtering.

    Requires BUDGETS_READ permission.

    Args:
        period: Optional period filter (daily, weekly, monthly).
        is_active: Optional active status filter.
        include_usage: Whether to include current usage (default True).

    Returns:
        BudgetList with budget items.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        budgets = await repo.list_budgets(
            period=period,
            is_active=is_active,
        )

        items = []
        for budget in budgets:
            response = BudgetWithUsageResponse(
                id=budget.id,
                name=budget.name,
                period=budget.period,
                limit_usd=float(budget.limit_usd),
                alert_threshold=budget.alert_threshold,
                scope=budget.scope_json,
                alert_channels=budget.alert_channels_json,
                description=budget.description,
                is_active=budget.is_active,
                created_at=budget.created_at,
                updated_at=budget.updated_at,
                usage=None,
            )

            if include_usage:
                usage = await repo.get_budget_usage(budget)
                response.usage = _usage_to_response(usage)

            items.append(response)

        return BudgetList(
            items=items,
            total=len(items),
        )


@router.get("/status/all", response_model=list[BudgetUsageResponse])
async def check_all_budgets(
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_READ))],
    reference_date: datetime | None = Query(
        None, description="Reference date for usage calculation"
    ),
) -> list[BudgetUsageResponse]:
    """Check usage for all active budgets.

    Requires BUDGETS_READ permission.

    Args:
        reference_date: Optional reference date for calculation.

    Returns:
        List of BudgetUsageResponse for all active budgets.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        usages = await repo.check_all_budgets(reference_date)

        return [_usage_to_response(usage) for usage in usages]


@router.get("/{budget_id}", response_model=BudgetWithUsageResponse)
async def get_budget(
    budget_id: int,
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_READ))],
    include_usage: bool = Query(
        True, description="Include current usage for the budget"
    ),
) -> BudgetWithUsageResponse:
    """Get a budget by ID.

    Requires BUDGETS_READ permission.

    Args:
        budget_id: Budget ID.
        include_usage: Whether to include current usage (default True).

    Returns:
        BudgetWithUsageResponse for the budget.

    Raises:
        HTTPException: If budget not found.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        budget = await repo.get_budget(budget_id)
        if budget is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with ID {budget_id} not found",
            )

        response = BudgetWithUsageResponse(
            id=budget.id,
            name=budget.name,
            period=budget.period,
            limit_usd=float(budget.limit_usd),
            alert_threshold=budget.alert_threshold,
            scope=budget.scope_json,
            alert_channels=budget.alert_channels_json,
            description=budget.description,
            is_active=budget.is_active,
            created_at=budget.created_at,
            updated_at=budget.updated_at,
            usage=None,
        )

        if include_usage:
            usage = await repo.get_budget_usage(budget)
            response.usage = _usage_to_response(usage)

        return response


@router.post("", response_model=BudgetResponse, status_code=status.HTTP_201_CREATED)
async def create_budget(
    data: BudgetCreate,
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_WRITE))],
) -> BudgetResponse:
    """Create a new budget.

    Requires BUDGETS_WRITE permission.

    Args:
        data: Budget creation data.

    Returns:
        Created BudgetResponse.

    Raises:
        HTTPException: If budget with same name exists.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        # Check if budget with same name exists
        existing = await repo.get_budget_by_name(data.name)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Budget with name '{data.name}' already exists",
            )

        budget = await repo.create_budget(
            name=data.name,
            period=data.period,
            limit_usd=Decimal(str(data.limit_usd)),
            alert_threshold=data.alert_threshold,
            scope=data.scope,
            alert_channels=data.alert_channels,
            description=data.description,
        )

        await session.commit()

        return _budget_to_response(budget)


@router.put("/{budget_id}", response_model=BudgetResponse)
async def update_budget(
    budget_id: int,
    data: BudgetUpdate,
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_WRITE))],
) -> BudgetResponse:
    """Update a budget.

    Requires BUDGETS_WRITE permission.

    Args:
        budget_id: Budget ID.
        data: Budget update data.

    Returns:
        Updated BudgetResponse.

    Raises:
        HTTPException: If budget not found or name conflict.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        budget = await repo.get_budget(budget_id)
        if budget is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with ID {budget_id} not found",
            )

        # Check name conflict if name is being changed
        if data.name is not None and data.name != budget.name:
            existing = await repo.get_budget_by_name(data.name)
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Budget with name '{data.name}' already exists",
                )

        # Prepare update kwargs
        update_kwargs: dict[str, str | Decimal | float | dict | list | bool | None] = {}
        if data.name is not None:
            update_kwargs["name"] = data.name
        if data.period is not None:
            update_kwargs["period"] = data.period
        if data.limit_usd is not None:
            update_kwargs["limit_usd"] = Decimal(str(data.limit_usd))
        if data.alert_threshold is not None:
            update_kwargs["alert_threshold"] = data.alert_threshold
        if data.scope is not None:
            update_kwargs["scope"] = data.scope
        if data.alert_channels is not None:
            update_kwargs["alert_channels"] = data.alert_channels
        if data.description is not None:
            update_kwargs["description"] = data.description
        if data.is_active is not None:
            update_kwargs["is_active"] = data.is_active

        budget = await repo.update_budget(budget, **update_kwargs)  # type: ignore[arg-type]

        await session.commit()

        return _budget_to_response(budget)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_budget(
    budget_id: int,
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_DELETE))],
) -> None:
    """Delete a budget.

    Requires BUDGETS_DELETE permission.

    Args:
        budget_id: Budget ID.

    Raises:
        HTTPException: If budget not found.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        deleted = await repo.delete_budget(budget_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with ID {budget_id} not found",
            )

        await session.commit()


@router.get("/{budget_id}/usage", response_model=BudgetUsageResponse)
async def get_budget_usage(
    budget_id: int,
    _: Annotated[None, Depends(require_permission(Permission.BUDGETS_READ))],
    reference_date: datetime | None = Query(
        None, description="Reference date for usage calculation"
    ),
) -> BudgetUsageResponse:
    """Get current usage for a budget.

    Requires BUDGETS_READ permission.

    Args:
        budget_id: Budget ID.
        reference_date: Optional reference date for calculation.

    Returns:
        BudgetUsageResponse with current usage.

    Raises:
        HTTPException: If budget not found.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        budget = await repo.get_budget(budget_id)
        if budget is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with ID {budget_id} not found",
            )

        usage = await repo.get_budget_usage(budget, reference_date)

        return _usage_to_response(usage)
