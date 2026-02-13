"""Dashboard API routes for agent execution traces.

Provides endpoints for listing and viewing recorded execution
traces stored via the tracing module.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/traces", tags=["agent-traces"])


class TraceStepResponse(BaseModel):
    """API response for a single trace step."""

    sequence: int
    timestamp: str
    event_type: str
    task_id: str
    payload: dict[str, Any]
    duration_ms: float | None = None


class TraceSummaryResponse(BaseModel):
    """API response for trace summary."""

    trace_id: str
    test_id: str
    test_name: str
    status: str
    started_at: str
    completed_at: str | None
    total_events: int
    agent_name: str | None = None
    suite_name: str | None = None


class TraceDetailResponse(BaseModel):
    """API response for full trace details."""

    trace_id: str
    test_id: str
    test_name: str
    status: str
    started_at: str
    completed_at: str | None
    total_events: int
    error: str | None = None
    agent_name: str | None = None
    suite_name: str | None = None
    steps: list[TraceStepResponse] = Field(default_factory=list)
    event_type_counts: dict[str, int] = Field(default_factory=dict)
    duration_seconds: float | None = None


class TraceListResponse(BaseModel):
    """API response for trace listing."""

    total: int
    traces: list[TraceSummaryResponse]


@router.get(
    "/agent-traces",
    response_model=TraceListResponse,
)
async def list_agent_traces(
    test_id: str | None = Query(default=None, description="Filter by test ID"),
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> TraceListResponse:
    """List recorded agent execution traces.

    Args:
        test_id: Optional test ID filter.
        status: Optional status filter.
        limit: Maximum results to return.

    Returns:
        TraceListResponse with matching traces.
    """
    from atp.tracing import get_default_storage

    storage = get_default_storage()
    summaries = storage.list_traces(test_id=test_id, status=status, limit=limit)

    items = [
        TraceSummaryResponse(
            trace_id=s.trace_id,
            test_id=s.test_id,
            test_name=s.test_name,
            status=s.status,
            started_at=s.started_at.isoformat(),
            completed_at=(s.completed_at.isoformat() if s.completed_at else None),
            total_events=s.total_events,
            agent_name=s.metadata.agent_name,
            suite_name=s.metadata.suite_name,
        )
        for s in summaries
    ]

    return TraceListResponse(total=len(items), traces=items)


@router.get(
    "/agent-traces/{trace_id}",
    response_model=TraceDetailResponse,
)
async def get_agent_trace(trace_id: str) -> TraceDetailResponse:
    """Get full details of a specific trace.

    Args:
        trace_id: Unique trace identifier.

    Returns:
        TraceDetailResponse with trace details and steps.

    Raises:
        HTTPException: If trace is not found.
    """
    from atp.tracing import get_default_storage

    storage = get_default_storage()
    trace = storage.load(trace_id)

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trace not found: {trace_id}",
        )

    steps = [
        TraceStepResponse(
            sequence=step.sequence,
            timestamp=step.timestamp.isoformat(),
            event_type=step.event_type.value,
            task_id=step.task_id,
            payload=step.payload,
            duration_ms=step.duration_ms,
        )
        for step in trace.steps
    ]

    return TraceDetailResponse(
        trace_id=trace.trace_id,
        test_id=trace.test_id,
        test_name=trace.test_name,
        status=trace.status,
        started_at=trace.started_at.isoformat(),
        completed_at=(trace.completed_at.isoformat() if trace.completed_at else None),
        total_events=trace.total_events,
        error=trace.error,
        agent_name=trace.metadata.agent_name,
        suite_name=trace.metadata.suite_name,
        steps=steps,
        event_type_counts=trace.event_type_counts,
        duration_seconds=trace.duration_seconds,
    )
