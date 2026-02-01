"""Comparison routes.

This module provides endpoints for comparing multiple agents
on tests and suites, including side-by-side execution views.

Permissions:
    - GET /compare/agents: RESULTS_READ
    - GET /compare/side-by-side: RESULTS_READ
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import Agent, RunResult, SuiteExecution, TestExecution
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    AgentComparisonMetrics,
    AgentComparisonResponse,
    AgentExecutionDetail,
    EventSummary,
    SideBySideComparisonResponse,
    TestComparisonMetrics,
)
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/compare", tags=["compare"])


def _format_event_summary(event: dict) -> EventSummary:
    """Format a raw event dict into an EventSummary.

    Args:
        event: Raw event dictionary from events_json.

    Returns:
        EventSummary with formatted data.
    """
    event_type = event.get("event_type", "unknown")
    payload = event.get("payload", {})

    # Generate summary based on event type
    if event_type == "tool_call":
        tool = payload.get("tool", "unknown")
        event_status = payload.get("status", "")
        summary = f"Tool call: {tool} ({event_status})"
    elif event_type == "llm_request":
        model = payload.get("model", "unknown")
        tokens = payload.get("input_tokens", 0) + payload.get("output_tokens", 0)
        summary = f"LLM request: {model} ({tokens} tokens)"
    elif event_type == "reasoning":
        thought = payload.get("thought", "")
        step = payload.get("step", "")
        summary = thought[:50] + "..." if len(thought) > 50 else thought
        if not summary and step:
            summary = step
        if not summary:
            summary = "Reasoning step"
    elif event_type == "error":
        error_type = payload.get("error_type", "Error")
        message = payload.get("message", "")[:50]
        summary = f"{error_type}: {message}"
    elif event_type == "progress":
        percentage = payload.get("percentage", 0)
        message = payload.get("message", "")
        if message:
            summary = f"Progress: {percentage}% - {message}"
        else:
            summary = f"Progress: {percentage}%"
    else:
        summary = f"Event: {event_type}"

    # Parse timestamp
    timestamp_str = event.get("timestamp", "")
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    return EventSummary(
        sequence=event.get("sequence", 0),
        timestamp=timestamp,
        event_type=event_type,
        summary=summary,
        data=payload,
    )


def _calculate_metrics_from_run(
    run: RunResult,
) -> tuple[int | None, int | None, int | None, int | None, float | None]:
    """Calculate metrics from a run result.

    Args:
        run: RunResult instance.

    Returns:
        Tuple of (total_tokens, total_steps, tool_calls, llm_calls, cost_usd).
    """
    return (
        run.total_tokens,
        run.total_steps,
        run.tool_calls,
        run.llm_calls,
        run.cost_usd,
    )


@router.get("/agents", response_model=AgentComparisonResponse)
async def compare_agents(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    suite_name: str,
    agents: list[str] = Query(...),
    limit_per_agent: int = Query(default=10, le=50),
) -> AgentComparisonResponse:
    """Compare multiple agents on a suite.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        suite_name: Name of the test suite to compare.
        agents: List of agent names to compare.
        limit_per_agent: Max executions per agent to consider (default 10, max 50).

    Returns:
        AgentComparisonResponse with aggregate and per-test metrics.
    """
    agent_metrics: list[AgentComparisonMetrics] = []
    test_metrics_map: dict[str, dict[str, Any]] = {}

    for agent_name in agents:
        # Get agent's executions
        stmt = (
            select(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                Agent.name == agent_name,
            )
            .options(selectinload(SuiteExecution.test_executions))
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit_per_agent)
        )
        result = await session.execute(stmt)
        executions = list(result.scalars().all())

        if not executions:
            continue

        # Calculate aggregate metrics
        total_executions = len(executions)
        avg_success_rate = sum(e.success_rate for e in executions) / total_executions

        # Calculate average score
        all_scores: list[float] = []
        for exec in executions:
            for test in exec.test_executions:
                if test.score is not None:
                    all_scores.append(test.score)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else None

        # Calculate average duration
        durations = [e.duration_seconds for e in executions if e.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else None

        # Latest execution metrics
        latest = executions[0]
        latest_scores = [t.score for t in latest.test_executions if t.score is not None]
        latest_score = (
            sum(latest_scores) / len(latest_scores) if latest_scores else None
        )

        agent_metrics.append(
            AgentComparisonMetrics(
                agent_name=agent_name,
                total_executions=total_executions,
                avg_success_rate=avg_success_rate,
                avg_score=avg_score,
                avg_duration_seconds=avg_duration,
                latest_success_rate=latest.success_rate,
                latest_score=latest_score,
            )
        )

        # Collect per-test metrics
        for exec in executions:
            for test in exec.test_executions:
                if test.test_id not in test_metrics_map:
                    test_metrics_map[test.test_id] = {"_name": test.test_name}
                if agent_name not in test_metrics_map[test.test_id]:
                    test_metrics_map[test.test_id][agent_name] = {
                        "scores": [],
                        "durations": [],
                        "successes": [],
                    }
                data = test_metrics_map[test.test_id][agent_name]
                if isinstance(data, dict) and "scores" in data:
                    if test.score is not None:
                        data["scores"].append(test.score)
                    if test.duration_seconds is not None:
                        data["durations"].append(test.duration_seconds)
                    data["successes"].append(1 if test.success else 0)

    # Build test comparison metrics
    test_comparisons: list[TestComparisonMetrics] = []
    for test_id, agent_data in test_metrics_map.items():
        test_name = agent_data.pop("_name", test_id)
        metrics_by_agent: dict[str, AgentComparisonMetrics] = {}

        for agent_name_key, data in agent_data.items():
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
                durations = data["durations"]
                successes = data["successes"]

                metrics_by_agent[agent_name_key] = AgentComparisonMetrics(
                    agent_name=agent_name_key,
                    total_executions=len(successes),
                    avg_success_rate=sum(successes) / len(successes)
                    if successes
                    else 0,
                    avg_score=sum(scores) / len(scores) if scores else None,
                    avg_duration_seconds=sum(durations) / len(durations)
                    if durations
                    else None,
                    latest_success_rate=successes[-1] if successes else None,
                    latest_score=scores[-1] if scores else None,
                )

        test_comparisons.append(
            TestComparisonMetrics(
                test_id=test_id,
                test_name=test_name if isinstance(test_name, str) else test_id,
                metrics_by_agent=metrics_by_agent,
            )
        )

    return AgentComparisonResponse(
        suite_name=suite_name,
        agents=agent_metrics,
        tests=test_comparisons,
    )


@router.get(
    "/side-by-side",
    response_model=SideBySideComparisonResponse,
)
async def get_side_by_side_comparison(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    suite_name: str,
    test_id: str,
    agents: list[str] = Query(..., min_length=2, max_length=3),
) -> SideBySideComparisonResponse:
    """Get detailed side-by-side comparison of agents on a specific test.

    Requires RESULTS_READ permission.

    This endpoint returns the latest test execution for each agent on the
    specified test, including formatted events for timeline visualization
    and metrics for comparison.

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        test_id: ID of the specific test.
        agents: List of agent names to compare (2-3 agents).

    Returns:
        SideBySideComparisonResponse with execution details for each agent.

    Raises:
        HTTPException: If no executions found for any agent.
    """
    agent_details: list[AgentExecutionDetail] = []
    test_name: str | None = None

    for agent_name in agents:
        # Query latest test execution for this agent on this test
        stmt = (
            select(TestExecution)
            .join(SuiteExecution)
            .join(Agent)
            .where(
                SuiteExecution.suite_name == suite_name,
                TestExecution.test_id == test_id,
                Agent.name == agent_name,
            )
            .options(selectinload(TestExecution.run_results))
            .order_by(TestExecution.started_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            # Agent has no execution for this test - skip but don't fail
            continue

        # Get test name from first found execution
        if test_name is None:
            test_name = execution.test_name

        # Get the latest run result (or first if only one)
        run_results = sorted(execution.run_results, key=lambda r: r.run_number)
        latest_run = run_results[-1] if run_results else None

        # Extract and format events
        events: list[EventSummary] = []
        if latest_run and latest_run.events_json:
            for event in latest_run.events_json:
                events.append(_format_event_summary(event))
            # Sort by sequence
            events.sort(key=lambda e: e.sequence)

        # Calculate metrics
        total_tokens: int | None = None
        total_steps: int | None = None
        tool_calls: int | None = None
        llm_calls: int | None = None
        cost_usd: float | None = None

        if latest_run:
            total_tokens, total_steps, tool_calls, llm_calls, cost_usd = (
                _calculate_metrics_from_run(latest_run)
            )

        agent_details.append(
            AgentExecutionDetail(
                agent_name=agent_name,
                test_execution_id=execution.id,
                score=execution.score,
                success=execution.success,
                duration_seconds=execution.duration_seconds,
                total_tokens=total_tokens,
                total_steps=total_steps,
                tool_calls=tool_calls,
                llm_calls=llm_calls,
                cost_usd=cost_usd,
                events=events,
            )
        )

    if not agent_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No executions found for test '{test_id}' in suite '{suite_name}' "
            f"for any of the specified agents",
        )

    return SideBySideComparisonResponse(
        suite_name=suite_name,
        test_id=test_id,
        test_name=test_name or test_id,
        agents=agent_details,
    )
