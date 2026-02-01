"""Timeline visualization routes.

This module provides endpoints for viewing execution timelines
with event sequences and multi-agent comparison.

Permissions:
    - GET /timeline/events: RESULTS_READ
    - GET /timeline/compare: RESULTS_READ
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import Agent, SuiteExecution, TestExecution
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    AgentTimeline,
    MultiTimelineResponse,
    TimelineEvent,
    TimelineEventsResponse,
)
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/timeline", tags=["timeline"])


def _format_timeline_event(
    event: dict, first_timestamp: datetime | None
) -> TimelineEvent:
    """Format a raw event dict into a TimelineEvent with relative timing.

    Args:
        event: Raw event dictionary from events_json.
        first_timestamp: Timestamp of the first event for relative time calculation.

    Returns:
        TimelineEvent with formatted data and relative timing.
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

    # Calculate relative time from first event
    relative_time_ms = 0.0
    if first_timestamp is not None:
        delta = timestamp - first_timestamp
        relative_time_ms = delta.total_seconds() * 1000

    # Extract duration from payload if available
    duration_ms: float | None = None
    if "duration_ms" in payload:
        duration_ms = float(payload["duration_ms"])

    return TimelineEvent(
        sequence=event.get("sequence", 0),
        timestamp=timestamp,
        event_type=event_type,
        summary=summary,
        data=payload,
        relative_time_ms=relative_time_ms,
        duration_ms=duration_ms,
    )


@router.get(
    "/events",
    response_model=TimelineEventsResponse,
)
async def get_timeline_events(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    suite_name: str,
    test_id: str,
    agent_name: str,
    event_types: list[str] | None = Query(None),
    limit: int = Query(default=1000, le=1000),
    offset: int = Query(default=0, ge=0),
) -> TimelineEventsResponse:
    """Get timeline events for a specific test execution.

    Requires RESULTS_READ permission.

    Returns events from the latest test execution for the specified agent,
    with relative timing information for timeline visualization.

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        test_id: ID of the specific test.
        agent_name: Name of the agent.
        event_types: Optional list of event types to filter by
            (tool_call, llm_request, reasoning, error, progress).
        limit: Maximum number of events to return (default 1000, max 1000).
        offset: Offset for pagination.

    Returns:
        TimelineEventsResponse with events and timing information.

    Raises:
        HTTPException: If no execution found for the specified parameters.
    """
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No execution found for test '{test_id}' in suite '{suite_name}' "
            f"by agent '{agent_name}'",
        )

    # Get the latest run result
    run_results = sorted(execution.run_results, key=lambda r: r.run_number)
    latest_run = run_results[-1] if run_results else None

    if latest_run is None or not latest_run.events_json:
        # Return empty response if no events
        return TimelineEventsResponse(
            suite_name=suite_name,
            test_id=test_id,
            test_name=execution.test_name,
            agent_name=agent_name,
            total_events=0,
            events=[],
            total_duration_ms=None,
            execution_id=execution.id,
        )

    # Get all events from the run
    raw_events = latest_run.events_json

    # Sort by sequence to ensure correct ordering
    raw_events = sorted(raw_events, key=lambda e: e.get("sequence", 0))

    # Apply event type filtering if specified
    if event_types:
        valid_types = set(event_types)
        raw_events = [e for e in raw_events if e.get("event_type") in valid_types]

    # Get total count before pagination
    total_events = len(raw_events)

    # Apply pagination
    paginated_events = raw_events[offset : offset + limit]

    # Parse first timestamp for relative time calculation
    first_timestamp: datetime | None = None
    if raw_events:
        first_ts_str = raw_events[0].get("timestamp", "")
        if first_ts_str:
            try:
                first_timestamp = datetime.fromisoformat(
                    first_ts_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

    # Format events with relative timing
    timeline_events: list[TimelineEvent] = []
    for event in paginated_events:
        timeline_events.append(_format_timeline_event(event, first_timestamp))

    # Calculate total duration from first to last event
    total_duration_ms: float | None = None
    if raw_events and len(raw_events) > 1:
        last_ts_str = raw_events[-1].get("timestamp", "")
        if first_timestamp and last_ts_str:
            try:
                last_timestamp = datetime.fromisoformat(
                    last_ts_str.replace("Z", "+00:00")
                )
                delta = last_timestamp - first_timestamp
                total_duration_ms = delta.total_seconds() * 1000
            except ValueError:
                pass

    return TimelineEventsResponse(
        suite_name=suite_name,
        test_id=test_id,
        test_name=execution.test_name,
        agent_name=agent_name,
        total_events=total_events,
        events=timeline_events,
        total_duration_ms=total_duration_ms,
        execution_id=execution.id,
    )


@router.get(
    "/compare",
    response_model=MultiTimelineResponse,
)
async def get_multi_timeline(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    suite_name: str,
    test_id: str,
    agents: list[str] = Query(..., min_length=2, max_length=3),
    event_types: list[str] | None = Query(None),
) -> MultiTimelineResponse:
    """Get aligned timelines for multiple agents on the same test.

    Requires RESULTS_READ permission.

    Returns timelines for 2-3 agents aligned by start time, enabling
    visual comparison of execution strategies and timing.

    Args:
        session: Database session.
        suite_name: Name of the test suite.
        test_id: ID of the specific test.
        agents: List of agent names to compare (2-3 agents).
        event_types: Optional list of event types to filter by
            (tool_call, llm_request, reasoning, error, progress).

    Returns:
        MultiTimelineResponse with aligned timelines for each agent.

    Raises:
        HTTPException: If no executions found for any agent.
    """
    timelines: list[AgentTimeline] = []
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

        # Get the latest run result
        run_results = sorted(execution.run_results, key=lambda r: r.run_number)
        latest_run = run_results[-1] if run_results else None

        if latest_run is None or not latest_run.events_json:
            # No events - create an empty timeline
            timelines.append(
                AgentTimeline(
                    agent_name=agent_name,
                    test_execution_id=execution.id,
                    start_time=execution.started_at,
                    total_duration_ms=0.0,
                    events=[],
                )
            )
            continue

        # Get all events from the run
        raw_events = latest_run.events_json

        # Sort by sequence to ensure correct ordering
        raw_events = sorted(raw_events, key=lambda e: e.get("sequence", 0))

        # Apply event type filtering if specified
        if event_types:
            valid_types = set(event_types)
            raw_events = [e for e in raw_events if e.get("event_type") in valid_types]

        # Parse first timestamp for relative time calculation
        first_timestamp: datetime | None = None
        if raw_events:
            first_ts_str = raw_events[0].get("timestamp", "")
            if first_ts_str:
                try:
                    first_timestamp = datetime.fromisoformat(
                        first_ts_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Format events with relative timing
        timeline_events: list[TimelineEvent] = []
        for event in raw_events:
            timeline_events.append(_format_timeline_event(event, first_timestamp))

        # Calculate total duration from first to last event
        total_duration_ms = 0.0
        if raw_events and len(raw_events) > 1:
            last_ts_str = raw_events[-1].get("timestamp", "")
            if first_timestamp and last_ts_str:
                try:
                    last_timestamp = datetime.fromisoformat(
                        last_ts_str.replace("Z", "+00:00")
                    )
                    delta = last_timestamp - first_timestamp
                    total_duration_ms = delta.total_seconds() * 1000
                except ValueError:
                    pass

        # Use first timestamp or execution start time
        start_time = first_timestamp or execution.started_at

        timelines.append(
            AgentTimeline(
                agent_name=agent_name,
                test_execution_id=execution.id,
                start_time=start_time,
                total_duration_ms=total_duration_ms,
                events=timeline_events,
            )
        )

    if not timelines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No executions found for test '{test_id}' in suite '{suite_name}' "
            f"for any of the specified agents",
        )

    return MultiTimelineResponse(
        suite_name=suite_name,
        test_id=test_id,
        test_name=test_name or test_id,
        timelines=timelines,
    )
