"""Cost analytics routes.

This module provides endpoints for viewing cost data, breakdowns,
and trends for LLM operations.

Permissions:
    - GET /costs: ANALYTICS_READ (view cost summary)
    - GET /costs/records: ANALYTICS_READ
    - GET /costs/by-provider: ANALYTICS_READ
    - GET /costs/by-model: ANALYTICS_READ
    - GET /costs/by-agent: ANALYTICS_READ
    - GET /costs/trend: ANALYTICS_READ
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from atp.analytics.database import AnalyticsDatabase
from atp.analytics.repository import CostRepository
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    CostBreakdownItem,
    CostRecordList,
    CostRecordResponse,
    CostSummaryResponse,
    CostTrendPoint,
)

router = APIRouter(prefix="/costs", tags=["costs"])


async def get_analytics_session():
    """Get an analytics database session."""
    db = AnalyticsDatabase()
    await db.create_tables()
    async with db.session() as session:
        yield session


def _decimal_to_float(value: Decimal | None) -> float:
    """Convert Decimal to float, handling None."""
    if value is None:
        return 0.0
    return float(value)


def _build_breakdown_items(
    rows: list[dict[str, Any]], name_key: str, total_cost: Decimal
) -> list[CostBreakdownItem]:
    """Build breakdown items from aggregation rows."""
    items = []
    for row in rows:
        row_cost = _decimal_to_float(row.get("total_cost"))
        percentage = (row_cost / float(total_cost) * 100) if total_cost > 0 else 0.0
        items.append(
            CostBreakdownItem(
                name=row.get(name_key) or "unknown",
                total_cost=row_cost,
                total_input_tokens=row.get("total_input_tokens") or 0,
                total_output_tokens=row.get("total_output_tokens") or 0,
                record_count=row.get("record_count") or 0,
                percentage=round(percentage, 2),
            )
        )
    return items


@router.get("", response_model=CostSummaryResponse)
async def get_cost_summary(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    start_date: datetime | None = Query(None, description="Start date for filtering"),
    end_date: datetime | None = Query(None, description="End date for filtering"),
    provider: str | None = Query(None, description="Filter by provider"),
    model: str | None = Query(None, description="Filter by model"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
) -> CostSummaryResponse:
    """Get cost summary with breakdowns by provider, model, and agent.

    Requires ANALYTICS_READ permission.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        provider: Optional provider filter.
        model: Optional model filter.
        agent_name: Optional agent name filter.

    Returns:
        CostSummaryResponse with total costs and breakdowns.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        # Get total cost
        total_cost = await repo.get_total_cost(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            model=model,
            agent_name=agent_name,
        )

        # Get breakdowns
        by_provider_raw = await repo.get_costs_by_provider(
            start_date=start_date,
            end_date=end_date,
        )
        by_model_raw = await repo.get_costs_by_model(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
        )
        by_agent_raw = await repo.get_costs_by_agent(
            start_date=start_date,
            end_date=end_date,
        )

        # Get daily trend
        daily_trend_raw = await repo.get_costs_by_day(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            model=model,
            agent_name=agent_name,
        )

        # Calculate totals
        total_input_tokens = sum(
            row.get("total_input_tokens") or 0 for row in by_provider_raw
        )
        total_output_tokens = sum(
            row.get("total_output_tokens") or 0 for row in by_provider_raw
        )
        total_records = sum(row.get("record_count") or 0 for row in by_provider_raw)

        # Build response
        return CostSummaryResponse(
            total_cost=_decimal_to_float(total_cost),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_records=total_records,
            by_provider=_build_breakdown_items(by_provider_raw, "provider", total_cost),
            by_model=_build_breakdown_items(by_model_raw, "model", total_cost),
            by_agent=_build_breakdown_items(by_agent_raw, "agent_name", total_cost),
            daily_trend=[
                CostTrendPoint(
                    date=str(row["date"]),
                    total_cost=_decimal_to_float(row.get("total_cost")),
                    total_tokens=row.get("total_tokens") or 0,
                    record_count=row.get("record_count") or 0,
                )
                for row in daily_trend_raw
            ],
        )


@router.get("/records", response_model=CostRecordList)
async def list_cost_records(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    start_date: datetime | None = Query(None, description="Start date for filtering"),
    end_date: datetime | None = Query(None, description="End date for filtering"),
    provider: str | None = Query(None, description="Filter by provider"),
    model: str | None = Query(None, description="Filter by model"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
    suite_id: str | None = Query(None, description="Filter by suite ID"),
    test_id: str | None = Query(None, description="Filter by test ID"),
    limit: int = Query(default=50, le=100, ge=1, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
) -> CostRecordList:
    """List cost records with optional filtering.

    Requires ANALYTICS_READ permission.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        provider: Optional provider filter.
        model: Optional model filter.
        agent_name: Optional agent name filter.
        suite_id: Optional suite ID filter.
        test_id: Optional test ID filter.
        limit: Maximum number of records to return.
        offset: Number of records to skip.

    Returns:
        CostRecordList with paginated cost records.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        records = await repo.list_cost_records(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            model=model,
            agent_name=agent_name,
            suite_id=suite_id,
            test_id=test_id,
            limit=limit,
            offset=offset,
        )

        # Get total count (approximate - for now use the current page)
        # For proper pagination, we would need a count query
        total = len(records) + offset
        if len(records) == limit:
            # There might be more records
            total = offset + limit + 1

        items = [
            CostRecordResponse(
                id=record.id,
                timestamp=record.timestamp,
                provider=record.provider,
                model=record.model,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cost_usd=float(record.cost_usd),
                test_id=record.test_id,
                suite_id=record.suite_id,
                agent_name=record.agent_name,
                metadata=record.metadata_json,
            )
            for record in records
        ]

        return CostRecordList(
            total=total,
            items=items,
            limit=limit,
            offset=offset,
        )


@router.get("/by-provider", response_model=list[CostBreakdownItem])
async def get_costs_by_provider(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    start_date: datetime | None = Query(None, description="Start date for filtering"),
    end_date: datetime | None = Query(None, description="End date for filtering"),
) -> list[CostBreakdownItem]:
    """Get cost breakdown by provider.

    Requires ANALYTICS_READ permission.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.

    Returns:
        List of CostBreakdownItem for each provider.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        total_cost = await repo.get_total_cost(
            start_date=start_date,
            end_date=end_date,
        )

        by_provider_raw = await repo.get_costs_by_provider(
            start_date=start_date,
            end_date=end_date,
        )

        return _build_breakdown_items(by_provider_raw, "provider", total_cost)


@router.get("/by-model", response_model=list[CostBreakdownItem])
async def get_costs_by_model(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    start_date: datetime | None = Query(None, description="Start date for filtering"),
    end_date: datetime | None = Query(None, description="End date for filtering"),
    provider: str | None = Query(None, description="Filter by provider"),
) -> list[CostBreakdownItem]:
    """Get cost breakdown by model.

    Requires ANALYTICS_READ permission.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        provider: Optional provider filter.

    Returns:
        List of CostBreakdownItem for each model.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        total_cost = await repo.get_total_cost(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
        )

        by_model_raw = await repo.get_costs_by_model(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
        )

        return _build_breakdown_items(by_model_raw, "model", total_cost)


@router.get("/by-agent", response_model=list[CostBreakdownItem])
async def get_costs_by_agent(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    start_date: datetime | None = Query(None, description="Start date for filtering"),
    end_date: datetime | None = Query(None, description="End date for filtering"),
) -> list[CostBreakdownItem]:
    """Get cost breakdown by agent.

    Requires ANALYTICS_READ permission.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.

    Returns:
        List of CostBreakdownItem for each agent.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        total_cost = await repo.get_total_cost(
            start_date=start_date,
            end_date=end_date,
        )

        by_agent_raw = await repo.get_costs_by_agent(
            start_date=start_date,
            end_date=end_date,
        )

        return _build_breakdown_items(by_agent_raw, "agent_name", total_cost)


@router.get("/trend", response_model=list[CostTrendPoint])
async def get_cost_trend(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    start_date: datetime | None = Query(None, description="Start date for filtering"),
    end_date: datetime | None = Query(None, description="End date for filtering"),
    provider: str | None = Query(None, description="Filter by provider"),
    model: str | None = Query(None, description="Filter by model"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
) -> list[CostTrendPoint]:
    """Get daily cost trend.

    Requires ANALYTICS_READ permission.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        provider: Optional provider filter.
        model: Optional model filter.
        agent_name: Optional agent name filter.

    Returns:
        List of CostTrendPoint for daily cost trend.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = CostRepository(session)

        daily_trend_raw = await repo.get_costs_by_day(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            model=model,
            agent_name=agent_name,
        )

        return [
            CostTrendPoint(
                date=str(row["date"]),
                total_cost=_decimal_to_float(row.get("total_cost")),
                total_tokens=row.get("total_tokens") or 0,
                record_count=row.get("record_count") or 0,
            )
            for row in daily_trend_raw
        ]
