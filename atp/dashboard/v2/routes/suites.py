"""Suite execution routes.

This module provides endpoints for querying and viewing
suite execution results.
"""

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import SuiteExecution
from atp.dashboard.schemas import (
    SuiteExecutionDetail,
    SuiteExecutionList,
    SuiteExecutionSummary,
    TestExecutionSummary,
)
from atp.dashboard.v2.dependencies import CurrentUser, DBSession

router = APIRouter(prefix="/suites", tags=["suites"])


@router.get("", response_model=SuiteExecutionList)
async def list_suite_executions(
    session: DBSession,
    user: CurrentUser,
    suite_name: str | None = None,
    agent_id: int | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> SuiteExecutionList:
    """List suite executions with optional filtering.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Filter by suite name.
        agent_id: Filter by agent ID.
        limit: Maximum number of results (default 50, max 100).
        offset: Offset for pagination.

    Returns:
        Paginated list of suite executions.
    """
    # Build query
    stmt = select(SuiteExecution).options(selectinload(SuiteExecution.agent))
    if suite_name:
        stmt = stmt.where(SuiteExecution.suite_name == suite_name)
    if agent_id:
        stmt = stmt.where(SuiteExecution.agent_id == agent_id)

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    # Get paginated results
    stmt = stmt.order_by(SuiteExecution.started_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    executions = result.scalars().all()

    items = []
    for exec in executions:
        summary = SuiteExecutionSummary.model_validate(exec)
        summary.agent_name = exec.agent.name if exec.agent else None
        items.append(summary)

    return SuiteExecutionList(
        total=total,
        items=items,
        limit=limit,
        offset=offset,
    )


@router.get("/names/list", response_model=list[str])
async def list_suite_names(session: DBSession, user: CurrentUser) -> list[str]:
    """List unique suite names.

    Args:
        session: Database session.
        user: Current user (optional auth).

    Returns:
        List of unique suite names ordered alphabetically.
    """
    stmt = (
        select(SuiteExecution.suite_name).distinct().order_by(SuiteExecution.suite_name)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.get("/{execution_id}", response_model=SuiteExecutionDetail)
async def get_suite_execution(
    session: DBSession, execution_id: int, user: CurrentUser
) -> SuiteExecutionDetail:
    """Get suite execution details.

    Args:
        session: Database session.
        execution_id: Suite execution ID.
        user: Current user (optional auth).

    Returns:
        Detailed suite execution information including test executions.

    Raises:
        HTTPException: If execution not found.
    """
    stmt = (
        select(SuiteExecution)
        .where(SuiteExecution.id == execution_id)
        .options(
            selectinload(SuiteExecution.agent),
            selectinload(SuiteExecution.test_executions),
        )
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite execution {execution_id} not found",
        )

    detail = SuiteExecutionDetail.model_validate(execution)
    detail.agent_name = execution.agent.name if execution.agent else None
    detail.tests = [
        TestExecutionSummary.model_validate(t) for t in execution.test_executions
    ]
    return detail
