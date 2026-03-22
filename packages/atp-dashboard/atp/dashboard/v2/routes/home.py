"""Home page and dashboard summary routes.

This module provides the main dashboard endpoint that returns
summary statistics for the platform.

Permissions:
    - GET /dashboard/summary: SUITES_READ (requires ability to view results)
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import Agent, SuiteExecution
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import DashboardSummary, SuiteExecutionSummary
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.SUITES_READ))],
) -> DashboardSummary:
    """Get dashboard summary statistics.

    Requires SUITES_READ permission.

    Returns aggregate statistics for the platform including:
    - Total number of agents
    - Total number of unique suites
    - Total number of suite executions
    - Recent success rate and average score
    - List of recent executions

    Args:
        session: Database session.

    Returns:
        DashboardSummary with platform statistics.
    """
    # Count agents
    agent_count = (await session.execute(select(func.count(Agent.id)))).scalar() or 0

    # Count unique suites
    suite_count = (
        await session.execute(
            select(func.count(func.distinct(SuiteExecution.suite_name)))
        )
    ).scalar() or 0

    # Count total executions
    exec_count = (
        await session.execute(select(func.count(SuiteExecution.id)))
    ).scalar() or 0

    # Get recent executions (last 10)
    stmt = (
        select(SuiteExecution)
        .options(
            selectinload(SuiteExecution.agent),
            selectinload(SuiteExecution.test_executions),
        )
        .order_by(SuiteExecution.started_at.desc())
        .limit(10)
    )
    result = await session.execute(stmt)
    recent_execs = result.scalars().all()

    # Calculate recent success rate and score
    if recent_execs:
        recent_success_rate = sum(e.success_rate for e in recent_execs) / len(
            recent_execs
        )

        # Get scores from test executions
        all_scores: list[float] = []
        for exec in recent_execs:
            if hasattr(exec, "test_executions"):
                for test in exec.test_executions:
                    if test.score is not None:
                        all_scores.append(test.score)
        recent_avg_score = sum(all_scores) / len(all_scores) if all_scores else None
    else:
        recent_success_rate = 0.0
        recent_avg_score = None

    recent_summaries = []
    for exec in recent_execs:
        summary = SuiteExecutionSummary.model_validate(exec)
        summary.agent_name = exec.agent.name if exec.agent else None
        recent_summaries.append(summary)

    return DashboardSummary(
        total_agents=agent_count,
        total_suites=suite_count,
        total_executions=exec_count,
        recent_success_rate=recent_success_rate,
        recent_avg_score=recent_avg_score,
        recent_executions=recent_summaries,
    )
