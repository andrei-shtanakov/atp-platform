"""Trend analysis routes.

This module provides endpoints for analyzing historical trends
in test and suite performance over time.
"""

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import Agent, SuiteExecution, TestExecution
from atp.dashboard.schemas import (
    SuiteTrend,
    TestTrend,
    TrendDataPoint,
    TrendResponse,
)
from atp.dashboard.v2.dependencies import CurrentUser, DBSession

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("/suite", response_model=TrendResponse)
async def get_suite_trends(
    session: DBSession,
    user: CurrentUser,
    suite_name: str,
    agent_name: str | None = None,
    metric: str = Query(
        default="success_rate", pattern="^(success_rate|score|duration)$"
    ),
    limit: int = Query(default=30, le=100),
) -> TrendResponse:
    """Get historical trends for a suite.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Name of the suite to analyze.
        agent_name: Optional agent name to filter by.
        metric: Metric to track (success_rate, score, or duration).
        limit: Maximum number of data points (default 30, max 100).

    Returns:
        TrendResponse with historical data points.
    """
    # Build query
    stmt = (
        select(SuiteExecution)
        .where(SuiteExecution.suite_name == suite_name)
        .options(selectinload(SuiteExecution.agent))
        .order_by(SuiteExecution.started_at.desc())
        .limit(limit)
    )
    if agent_name:
        stmt = stmt.join(Agent).where(Agent.name == agent_name)

    result = await session.execute(stmt)
    executions = result.scalars().all()

    # Build trend data
    data_points: list[TrendDataPoint] = []
    for exec in reversed(executions):
        if metric == "success_rate":
            value = exec.success_rate
        elif metric == "score":
            # Calculate average score from tests
            if exec.test_executions:
                scores = [t.score for t in exec.test_executions if t.score is not None]
                value = sum(scores) / len(scores) if scores else 0.0
            else:
                value = 0.0
        else:  # duration
            value = exec.duration_seconds or 0.0

        data_points.append(
            TrendDataPoint(
                timestamp=exec.started_at,
                value=value,
                execution_id=exec.id,
            )
        )

    agent_name_display = agent_name or "all"
    return TrendResponse(
        suite_trends=[
            SuiteTrend(
                suite_name=suite_name,
                agent_name=agent_name_display,
                data_points=data_points,
                metric=metric,
            )
        ]
    )


@router.get("/test", response_model=TrendResponse)
async def get_test_trends(
    session: DBSession,
    user: CurrentUser,
    suite_name: str,
    test_id: str,
    agent_name: str | None = None,
    metric: str = Query(default="score", pattern="^(score|duration|success_rate)$"),
    limit: int = Query(default=30, le=100),
) -> TrendResponse:
    """Get historical trends for a specific test.

    Args:
        session: Database session.
        user: Current user (optional auth).
        suite_name: Name of the suite containing the test.
        test_id: ID of the test to analyze.
        agent_name: Optional agent name to filter by.
        metric: Metric to track (score, duration, or success_rate).
        limit: Maximum number of data points (default 30, max 100).

    Returns:
        TrendResponse with historical data points for the test.
    """
    # Build query
    stmt = (
        select(TestExecution)
        .join(SuiteExecution)
        .where(
            SuiteExecution.suite_name == suite_name,
            TestExecution.test_id == test_id,
        )
        .order_by(TestExecution.started_at.desc())
        .limit(limit)
    )
    if agent_name:
        stmt = stmt.join(Agent).where(Agent.name == agent_name)

    result = await session.execute(stmt)
    executions = result.scalars().all()

    # Get test name from first execution
    test_name = executions[0].test_name if executions else test_id

    # Build trend data
    data_points: list[TrendDataPoint] = []
    for exec in reversed(executions):
        if metric == "score":
            value = exec.score or 0.0
        elif metric == "duration":
            value = exec.duration_seconds or 0.0
        else:  # success_rate
            value = 1.0 if exec.success else 0.0

        data_points.append(
            TrendDataPoint(
                timestamp=exec.started_at,
                value=value,
                execution_id=exec.id,
            )
        )

    return TrendResponse(
        test_trends=[
            TestTrend(
                test_id=test_id,
                test_name=test_name,
                data_points=data_points,
                metric=metric,
            )
        ]
    )
