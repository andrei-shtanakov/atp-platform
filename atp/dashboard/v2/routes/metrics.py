"""Prometheus metrics endpoint.

This module provides the /metrics endpoint for Prometheus scraping
and metrics status endpoints for monitoring the ATP platform.
"""

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from atp.core.metrics import (
    generate_metrics,
    get_metrics,
    get_metrics_settings,
)

router = APIRouter(tags=["metrics"])


class MetricsStatusResponse(BaseModel):
    """Response model for metrics status."""

    metrics_enabled: bool = Field(
        ..., description="Whether metrics collection is enabled"
    )
    prefix: str = Field(..., description="Metrics name prefix")
    multiprocess_mode: bool = Field(
        ..., description="Whether multiprocess mode is enabled"
    )


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    This endpoint is designed to be scraped by Prometheus.

    Returns:
        Response with metrics in Prometheus text format.
    """
    # Ensure metrics are initialized
    get_metrics()

    # Generate metrics output
    metrics_output = generate_metrics()

    return Response(
        content=metrics_output,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/status", response_model=MetricsStatusResponse)
async def get_metrics_status() -> MetricsStatusResponse:
    """Get metrics configuration status.

    Returns the current metrics configuration including whether
    metrics collection is enabled and the configuration settings.

    Returns:
        MetricsStatusResponse with configuration details.
    """
    settings = get_metrics_settings()

    return MetricsStatusResponse(
        metrics_enabled=settings.enabled,
        prefix=settings.prefix,
        multiprocess_mode=settings.multiprocess_mode,
    )
