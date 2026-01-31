"""Test execution routes.

This module provides endpoints for querying and viewing
test execution results.
"""

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import TestExecution
from atp.dashboard.schemas import (
    EvaluationResultResponse,
    RunResultSummary,
    ScoreComponentResponse,
    TestExecutionDetail,
    TestExecutionList,
    TestExecutionSummary,
)
from atp.dashboard.v2.dependencies import CurrentUser, DBSession

router = APIRouter(prefix="/tests", tags=["tests"])


@router.get("", response_model=TestExecutionList)
async def list_test_executions(
    session: DBSession,
    user: CurrentUser,
    suite_execution_id: int | None = None,
    test_id: str | None = None,
    success: bool | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> TestExecutionList:
    """List test executions with optional filtering.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_execution_id: Filter by suite execution ID.
        test_id: Filter by test ID.
        success: Filter by success status.
        limit: Maximum number of results (default 50, max 100).
        offset: Offset for pagination.

    Returns:
        Paginated list of test executions.
    """
    stmt = select(TestExecution)
    if suite_execution_id:
        stmt = stmt.where(TestExecution.suite_execution_id == suite_execution_id)
    if test_id:
        stmt = stmt.where(TestExecution.test_id == test_id)
    if success is not None:
        stmt = stmt.where(TestExecution.success == success)

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    # Get paginated results
    stmt = stmt.order_by(TestExecution.started_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    executions = result.scalars().all()

    return TestExecutionList(
        total=total,
        items=[TestExecutionSummary.model_validate(e) for e in executions],
        limit=limit,
        offset=offset,
    )


@router.get("/{execution_id}", response_model=TestExecutionDetail)
async def get_test_execution(
    session: DBSession, execution_id: int, user: CurrentUser
) -> TestExecutionDetail:
    """Get test execution details.

    Args:
        session: Database session.
        execution_id: Test execution ID.
        user: Current user (optional auth).

    Returns:
        Detailed test execution information including runs,
        evaluations, and score components.

    Raises:
        HTTPException: If execution not found.
    """
    stmt = (
        select(TestExecution)
        .where(TestExecution.id == execution_id)
        .options(
            selectinload(TestExecution.run_results),
            selectinload(TestExecution.evaluation_results),
            selectinload(TestExecution.score_components),
        )
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test execution {execution_id} not found",
        )

    detail = TestExecutionDetail.model_validate(execution)
    detail.runs = [RunResultSummary.model_validate(r) for r in execution.run_results]
    detail.evaluations = [
        EvaluationResultResponse.model_validate(e) for e in execution.evaluation_results
    ]
    detail.score_components = [
        ScoreComponentResponse.model_validate(s) for s in execution.score_components
    ]
    return detail
