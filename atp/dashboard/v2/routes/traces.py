"""Traces debug endpoint for OpenTelemetry spans.

This module provides a debug endpoint for viewing collected spans
when telemetry debug mode is enabled. This is intended for development
and debugging purposes only.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from atp.core.telemetry import get_debug_exporter, get_telemetry_settings

router = APIRouter(prefix="/debug", tags=["debug"])


class SpanResponse(BaseModel):
    """Response model for a single span."""

    trace_id: str = Field(..., description="Unique trace identifier")
    span_id: str = Field(..., description="Unique span identifier")
    parent_span_id: str | None = Field(None, description="Parent span ID if nested")
    name: str = Field(..., description="Span name")
    kind: str = Field(..., description="Span kind (INTERNAL, SERVER, CLIENT, etc.)")
    start_time: datetime = Field(..., description="Span start timestamp")
    end_time: datetime | None = Field(None, description="Span end timestamp")
    status: str = Field(..., description="Span status (OK, ERROR, UNSET)")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Span attributes"
    )
    events: list[dict[str, Any]] = Field(
        default_factory=list, description="Span events"
    )


class TracesResponse(BaseModel):
    """Response model for the traces endpoint."""

    enabled: bool = Field(..., description="Whether debug mode is enabled")
    span_count: int = Field(..., description="Number of spans returned")
    total_stored: int = Field(..., description="Total spans in storage")
    spans: list[SpanResponse] = Field(default_factory=list, description="List of spans")


class TelemetryStatusResponse(BaseModel):
    """Response model for telemetry status."""

    telemetry_enabled: bool = Field(..., description="Whether telemetry is enabled")
    debug_mode: bool = Field(
        ..., description="Whether debug mode (in-memory storage) is enabled"
    )
    otlp_endpoint: str | None = Field(
        None, description="OTLP exporter endpoint if configured"
    )
    service_name: str = Field(..., description="Service name")
    service_version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Deployment environment")
    sample_rate: float = Field(..., description="Trace sampling rate")
    max_debug_spans: int = Field(..., description="Maximum spans stored in debug mode")


@router.get("/traces", response_model=TracesResponse)
async def get_traces(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of spans to return",
    ),
    trace_id: str | None = Query(
        default=None,
        description="Filter by trace ID",
    ),
    name_filter: str | None = Query(
        default=None,
        description="Filter by span name substring",
    ),
) -> TracesResponse:
    """Get collected trace spans (debug mode only).

    This endpoint returns spans collected in memory when debug mode is enabled.
    It's intended for development and debugging purposes.

    Requires ATP_TELEMETRY_DEBUG_MODE=true to be set.

    Args:
        limit: Maximum number of spans to return (1-1000).
        trace_id: Optional filter by trace ID.
        name_filter: Optional filter by span name substring.

    Returns:
        TracesResponse with collected spans.

    Raises:
        HTTPException: If debug mode is not enabled.
    """
    settings = get_telemetry_settings()

    if not settings.debug_mode:
        raise HTTPException(
            status_code=404,
            detail=(
                "Debug traces endpoint is only available when "
                "ATP_TELEMETRY_DEBUG_MODE=true"
            ),
        )

    exporter = get_debug_exporter()
    if exporter is None:
        raise HTTPException(
            status_code=503,
            detail="Debug exporter not initialized. Ensure telemetry is configured.",
        )

    # Get spans from the in-memory exporter
    span_data = exporter.get_spans(
        limit=limit,
        trace_id=trace_id,
        name_filter=name_filter,
    )

    spans = [
        SpanResponse(
            trace_id=s.trace_id,
            span_id=s.span_id,
            parent_span_id=s.parent_span_id,
            name=s.name,
            kind=s.kind,
            start_time=s.start_time,
            end_time=s.end_time,
            status=s.status,
            attributes=s.attributes,
            events=s.events,
        )
        for s in span_data
    ]

    # Get total count
    all_spans = exporter.get_spans(limit=settings.max_debug_spans)
    total_stored = len(all_spans)

    return TracesResponse(
        enabled=True,
        span_count=len(spans),
        total_stored=total_stored,
        spans=spans,
    )


@router.get("/traces/{trace_id}", response_model=TracesResponse)
async def get_trace_by_id(
    trace_id: str,
) -> TracesResponse:
    """Get all spans for a specific trace.

    Args:
        trace_id: The trace ID to retrieve spans for.

    Returns:
        TracesResponse with spans for the given trace.

    Raises:
        HTTPException: If debug mode is not enabled or trace not found.
    """
    settings = get_telemetry_settings()

    if not settings.debug_mode:
        raise HTTPException(
            status_code=404,
            detail=(
                "Debug traces endpoint is only available when "
                "ATP_TELEMETRY_DEBUG_MODE=true"
            ),
        )

    exporter = get_debug_exporter()
    if exporter is None:
        raise HTTPException(
            status_code=503,
            detail="Debug exporter not initialized. Ensure telemetry is configured.",
        )

    # Get spans for the specific trace
    span_data = exporter.get_spans(trace_id=trace_id)

    if not span_data:
        raise HTTPException(
            status_code=404,
            detail=f"No spans found for trace ID: {trace_id}",
        )

    spans = [
        SpanResponse(
            trace_id=s.trace_id,
            span_id=s.span_id,
            parent_span_id=s.parent_span_id,
            name=s.name,
            kind=s.kind,
            start_time=s.start_time,
            end_time=s.end_time,
            status=s.status,
            attributes=s.attributes,
            events=s.events,
        )
        for s in span_data
    ]

    return TracesResponse(
        enabled=True,
        span_count=len(spans),
        total_stored=len(spans),
        spans=spans,
    )


@router.delete("/traces", status_code=204)
async def clear_traces() -> None:
    """Clear all stored trace spans (debug mode only).

    This endpoint clears the in-memory span storage.
    Requires ATP_TELEMETRY_DEBUG_MODE=true.

    Raises:
        HTTPException: If debug mode is not enabled.
    """
    settings = get_telemetry_settings()

    if not settings.debug_mode:
        raise HTTPException(
            status_code=404,
            detail=(
                "Debug traces endpoint is only available when "
                "ATP_TELEMETRY_DEBUG_MODE=true"
            ),
        )

    exporter = get_debug_exporter()
    if exporter is None:
        raise HTTPException(
            status_code=503,
            detail="Debug exporter not initialized. Ensure telemetry is configured.",
        )

    exporter.clear()


@router.get("/telemetry/status", response_model=TelemetryStatusResponse)
async def get_telemetry_status() -> TelemetryStatusResponse:
    """Get telemetry configuration status.

    Returns the current telemetry configuration including whether
    telemetry is enabled, debug mode status, and exporter configuration.

    Returns:
        TelemetryStatusResponse with configuration details.
    """
    settings = get_telemetry_settings()

    return TelemetryStatusResponse(
        telemetry_enabled=settings.enabled,
        debug_mode=settings.debug_mode,
        otlp_endpoint=settings.otlp_endpoint,
        service_name=settings.service_name,
        service_version=settings.service_version,
        environment=settings.environment,
        sample_rate=settings.sample_rate,
        max_debug_spans=settings.max_debug_spans,
    )
