"""Advanced analytics routes.

This module provides endpoints for trend analysis, anomaly detection,
correlation analysis, and data export functionality.

Permissions:
    - GET /analytics/trends: ANALYTICS_READ
    - GET /analytics/anomalies: ANALYTICS_READ
    - GET /analytics/correlations: ANALYTICS_READ
    - GET /analytics/export/csv: ANALYTICS_EXPORT
    - GET /analytics/export/excel: ANALYTICS_EXPORT
    - GET /analytics/reports: ANALYTICS_READ
    - GET /analytics/reports/{id}: ANALYTICS_READ
    - POST /analytics/reports: ANALYTICS_WRITE
    - PUT /analytics/reports/{id}: ANALYTICS_WRITE
    - DELETE /analytics/reports/{id}: ANALYTICS_DELETE
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from atp.analytics.advanced import (
    AdvancedAnalyticsService,
    ScheduledReportConfig,
    ScheduledReportFrequency,
    ScheduledReportsRepository,
)
from atp.analytics.database import AnalyticsDatabase
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    AnomalyDetectionResponseSchema,
    AnomalyResultResponse,
    CorrelationAnalysisResponseSchema,
    CorrelationResultResponse,
    ScheduledReportCreate,
    ScheduledReportList,
    ScheduledReportResponse,
    ScheduledReportUpdate,
    ScoreTrendsResponse,
    TrendAnalysisResponse,
    TrendDataPointResponse,
)
from atp.dashboard.v2.dependencies import DBSession, RequiredUser

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ==================== Trend Analysis ====================


@router.get("/trends", response_model=ScoreTrendsResponse)
async def get_score_trends(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    suite_name: str | None = Query(None, description="Filter by suite name"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
    test_id: str | None = Query(None, description="Filter by test ID"),
    start_date: datetime | None = Query(None, description="Start of analysis period"),
    end_date: datetime | None = Query(None, description="End of analysis period"),
    metrics: str = Query(
        default="score,success_rate,duration",
        description="Comma-separated metrics to analyze",
    ),
) -> ScoreTrendsResponse:
    """Analyze score trends over time.

    Requires ANALYTICS_READ permission.

    Returns trend analysis for specified metrics including:
    - Direction (improving, declining, stable)
    - Percent change over the period
    - Statistical measures (mean, standard deviation)
    - Data points for visualization

    Args:
        session: Database session.
        suite_name: Optional filter by suite name.
        agent_name: Optional filter by agent name.
        test_id: Optional filter by test ID.
        start_date: Start of analysis period (default: 30 days ago).
        end_date: End of analysis period (default: now).
        metrics: Comma-separated list of metrics to analyze.

    Returns:
        ScoreTrendsResponse with trend analysis for each metric.
    """
    service = AdvancedAnalyticsService(session)
    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]

    result = await service.analyze_score_trends(
        suite_name=suite_name,
        agent_name=agent_name,
        test_id=test_id,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics_list,
    )

    return ScoreTrendsResponse(
        suite_name=result.suite_name,
        agent_name=result.agent_name,
        trends=[
            TrendAnalysisResponse(
                metric=t.metric,
                direction=t.direction.value,
                change_percent=t.change_percent,
                start_value=t.start_value,
                end_value=t.end_value,
                average_value=t.average_value,
                std_deviation=t.std_deviation,
                data_points=[
                    TrendDataPointResponse(
                        date=dp.date,
                        value=dp.value,
                        count=dp.count,
                    )
                    for dp in t.data_points
                ],
                period_days=t.period_days,
                confidence=t.confidence,
            )
            for t in result.trends
        ],
        period_start=result.period_start,
        period_end=result.period_end,
    )


# ==================== Anomaly Detection ====================


@router.get("/anomalies", response_model=AnomalyDetectionResponseSchema)
async def detect_anomalies(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    suite_name: str | None = Query(None, description="Filter by suite name"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
    start_date: datetime | None = Query(None, description="Start of analysis period"),
    end_date: datetime | None = Query(None, description="End of analysis period"),
    metrics: str = Query(
        default="score,duration",
        description="Comma-separated metrics to analyze",
    ),
    sensitivity: str = Query(
        default="medium",
        pattern="^(low|medium|high)$",
        description="Detection sensitivity",
    ),
) -> AnomalyDetectionResponseSchema:
    """Detect anomalies in test results.

    Requires ANALYTICS_READ permission.

    Uses statistical analysis (z-score) to identify unusual results
    that deviate significantly from the baseline.

    Sensitivity levels:
    - low: Only detect severe anomalies (>3 standard deviations)
    - medium: Detect moderate anomalies (>2 standard deviations)
    - high: Detect mild anomalies (>1.5 standard deviations)

    Args:
        session: Database session.
        suite_name: Optional filter by suite name.
        agent_name: Optional filter by agent name.
        start_date: Start of analysis period (default: 30 days ago).
        end_date: End of analysis period (default: now).
        metrics: Comma-separated list of metrics to analyze.
        sensitivity: Detection sensitivity level.

    Returns:
        AnomalyDetectionResponseSchema with detected anomalies.
    """
    service = AdvancedAnalyticsService(session)
    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]

    result = await service.detect_anomalies(
        suite_name=suite_name,
        agent_name=agent_name,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics_list,
        sensitivity=sensitivity,
    )

    return AnomalyDetectionResponseSchema(
        anomalies=[
            AnomalyResultResponse(
                anomaly_type=a.anomaly_type.value,
                timestamp=a.timestamp,
                metric_name=a.metric_name,
                expected_value=a.expected_value,
                actual_value=a.actual_value,
                deviation_sigma=a.deviation_sigma,
                test_id=a.test_id,
                suite_id=a.suite_id,
                agent_name=a.agent_name,
                severity=a.severity,
                details=a.details,
            )
            for a in result.anomalies
        ],
        total_records_analyzed=result.total_records_analyzed,
        anomaly_rate=result.anomaly_rate,
        period_start=result.period_start,
        period_end=result.period_end,
    )


# ==================== Correlation Analysis ====================


@router.get("/correlations", response_model=CorrelationAnalysisResponseSchema)
async def analyze_correlations(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    suite_name: str | None = Query(None, description="Filter by suite name"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
    start_date: datetime | None = Query(None, description="Start of analysis period"),
    end_date: datetime | None = Query(None, description="End of analysis period"),
    factors: str = Query(
        default="duration,total_tokens,tool_calls,llm_calls",
        description="Comma-separated factors to correlate with scores",
    ),
) -> CorrelationAnalysisResponseSchema:
    """Analyze correlations between factors and test scores.

    Requires ANALYTICS_READ permission.

    Calculates Pearson correlation coefficients between factors
    (duration, token usage, tool calls, etc.) and test scores.

    Correlation strength classifications:
    - strong: |r| >= 0.7
    - moderate: 0.4 <= |r| < 0.7
    - weak: 0.2 <= |r| < 0.4
    - none: |r| < 0.2

    Args:
        session: Database session.
        suite_name: Optional filter by suite name.
        agent_name: Optional filter by agent name.
        start_date: Start of analysis period (default: 30 days ago).
        end_date: End of analysis period (default: now).
        factors: Comma-separated factors to analyze.

    Returns:
        CorrelationAnalysisResponseSchema with correlation results.
    """
    service = AdvancedAnalyticsService(session)
    factors_list = [f.strip() for f in factors.split(",") if f.strip()]

    result = await service.analyze_correlations(
        suite_name=suite_name,
        agent_name=agent_name,
        start_date=start_date,
        end_date=end_date,
        factors=factors_list,
    )

    return CorrelationAnalysisResponseSchema(
        correlations=[
            CorrelationResultResponse(
                factor_x=c.factor_x,
                factor_y=c.factor_y,
                correlation_coefficient=c.correlation_coefficient,
                strength=c.strength.value,
                sample_size=c.sample_size,
                p_value=c.p_value,
                details=c.details,
            )
            for c in result.correlations
        ],
        sample_size=result.sample_size,
        factors_analyzed=result.factors_analyzed,
    )


# ==================== Data Export ====================


@router.get("/export/csv")
async def export_to_csv(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_EXPORT))],
    suite_name: str | None = Query(None, description="Filter by suite name"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
    start_date: datetime | None = Query(None, description="Start date for export"),
    end_date: datetime | None = Query(None, description="End date for export"),
    include_runs: bool = Query(False, description="Include individual run details"),
) -> Response:
    """Export test results to CSV format.

    Requires ANALYTICS_EXPORT permission.

    Generates a CSV file containing test execution data with optional
    filters and run-level details.

    Args:
        session: Database session.
        suite_name: Optional filter by suite name.
        agent_name: Optional filter by agent name.
        start_date: Start date for export (default: 30 days ago).
        end_date: End date for export (default: now).
        include_runs: Whether to include individual run details.

    Returns:
        CSV file as downloadable response.
    """
    service = AdvancedAnalyticsService(session)

    csv_content = await service.export_to_csv(
        suite_name=suite_name,
        agent_name=agent_name,
        start_date=start_date,
        end_date=end_date,
        include_runs=include_runs,
    )

    # Generate filename
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{date_str}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/excel")
async def export_to_excel(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_EXPORT))],
    suite_name: str | None = Query(None, description="Filter by suite name"),
    agent_name: str | None = Query(None, description="Filter by agent name"),
    start_date: datetime | None = Query(None, description="Start date for export"),
    end_date: datetime | None = Query(None, description="End date for export"),
    include_trends: bool = Query(True, description="Include trends analysis sheet"),
    include_anomalies: bool = Query(True, description="Include anomalies sheet"),
) -> Response:
    """Export comprehensive analytics to Excel format.

    Requires ANALYTICS_EXPORT permission.

    Creates a multi-sheet Excel workbook with:
    - Test Results sheet
    - Trends sheet (optional)
    - Anomalies sheet (optional)

    Requires openpyxl package to be installed.

    Args:
        session: Database session.
        suite_name: Optional filter by suite name.
        agent_name: Optional filter by agent name.
        start_date: Start date for export (default: 30 days ago).
        end_date: End date for export (default: now).
        include_trends: Include trends analysis sheet.
        include_anomalies: Include anomalies sheet.

    Returns:
        Excel file as downloadable response.
    """
    service = AdvancedAnalyticsService(session)

    try:
        excel_content = await service.export_to_excel(
            suite_name=suite_name,
            agent_name=agent_name,
            start_date=start_date,
            end_date=end_date,
            include_trends=include_trends,
            include_anomalies=include_anomalies,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    # Generate filename
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analytics_report_{date_str}.xlsx"

    return Response(
        content=excel_content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ==================== Scheduled Reports ====================


@router.get("/reports", response_model=ScheduledReportList)
async def list_scheduled_reports(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    is_active: bool | None = Query(None, description="Filter by active status"),
) -> ScheduledReportList:
    """List scheduled analytics reports.

    Requires ANALYTICS_READ permission.

    Args:
        is_active: Optional filter by active status.

    Returns:
        List of scheduled reports.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = ScheduledReportsRepository(session)
        reports = await repo.list_reports(is_active=is_active)
        await session.commit()

        return ScheduledReportList(
            items=[
                ScheduledReportResponse(
                    id=r.id or 0,
                    name=r.name,
                    frequency=r.frequency.value,
                    recipients=r.recipients,
                    include_trends=r.include_trends,
                    include_anomalies=r.include_anomalies,
                    include_correlations=r.include_correlations,
                    filters=r.filters,
                    is_active=r.is_active,
                    last_run=r.last_run,
                    next_run=r.next_run,
                    created_at=r.created_at,
                )
                for r in reports
                if r.id is not None
            ],
            total=len(reports),
        )


@router.get("/reports/{report_id}", response_model=ScheduledReportResponse)
async def get_scheduled_report(
    report_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
) -> ScheduledReportResponse:
    """Get a scheduled report by ID.

    Requires ANALYTICS_READ permission.

    Args:
        report_id: Report ID.

    Returns:
        Scheduled report details.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = ScheduledReportsRepository(session)
        report = await repo.get_report(report_id)

        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")

        # ID is guaranteed to exist after retrieval from database
        assert report.id is not None
        return ScheduledReportResponse(
            id=report.id,
            name=report.name,
            frequency=report.frequency.value,
            recipients=report.recipients,
            include_trends=report.include_trends,
            include_anomalies=report.include_anomalies,
            include_correlations=report.include_correlations,
            filters=report.filters,
            is_active=report.is_active,
            last_run=report.last_run,
            next_run=report.next_run,
            created_at=report.created_at,
        )


@router.post("/reports", response_model=ScheduledReportResponse, status_code=201)
async def create_scheduled_report(
    report: ScheduledReportCreate,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ScheduledReportResponse:
    """Create a new scheduled report.

    Requires ANALYTICS_WRITE permission.

    Args:
        report: Report configuration.
        user: Authenticated user (required).

    Returns:
        Created report.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = ScheduledReportsRepository(session)

        config = ScheduledReportConfig(
            name=report.name,
            frequency=ScheduledReportFrequency(report.frequency),
            recipients=report.recipients,
            include_trends=report.include_trends,
            include_anomalies=report.include_anomalies,
            include_correlations=report.include_correlations,
            filters=report.filters,
        )

        created = await repo.create_report(config)
        await session.commit()

        # ID is guaranteed to exist after commit
        assert created.id is not None
        return ScheduledReportResponse(
            id=created.id,
            name=created.name,
            frequency=created.frequency.value,
            recipients=created.recipients,
            include_trends=created.include_trends,
            include_anomalies=created.include_anomalies,
            include_correlations=created.include_correlations,
            filters=created.filters,
            is_active=created.is_active,
            last_run=created.last_run,
            next_run=created.next_run,
            created_at=created.created_at,
        )


@router.put("/reports/{report_id}", response_model=ScheduledReportResponse)
async def update_scheduled_report(
    report_id: int,
    report: ScheduledReportUpdate,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ScheduledReportResponse:
    """Update a scheduled report.

    Requires ANALYTICS_WRITE permission.

    Args:
        report_id: Report ID.
        report: Updated configuration.
        user: Authenticated user (required).

    Returns:
        Updated report.
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = ScheduledReportsRepository(session)

        # Build updates dict, excluding None values
        updates: dict[str, Any] = {}
        if report.name is not None:
            updates["name"] = report.name
        if report.frequency is not None:
            updates["frequency"] = ScheduledReportFrequency(report.frequency)
        if report.recipients is not None:
            updates["recipients"] = report.recipients
        if report.include_trends is not None:
            updates["include_trends"] = report.include_trends
        if report.include_anomalies is not None:
            updates["include_anomalies"] = report.include_anomalies
        if report.include_correlations is not None:
            updates["include_correlations"] = report.include_correlations
        if report.filters is not None:
            updates["filters"] = report.filters
        if report.is_active is not None:
            updates["is_active"] = report.is_active

        updated = await repo.update_report(report_id, updates)
        await session.commit()

        if updated is None:
            raise HTTPException(status_code=404, detail="Report not found")

        # ID is guaranteed to exist after update
        assert updated.id is not None
        return ScheduledReportResponse(
            id=updated.id,
            name=updated.name,
            frequency=updated.frequency.value,
            recipients=updated.recipients,
            include_trends=updated.include_trends,
            include_anomalies=updated.include_anomalies,
            include_correlations=updated.include_correlations,
            filters=updated.filters,
            is_active=updated.is_active,
            last_run=updated.last_run,
            next_run=updated.next_run,
            created_at=updated.created_at,
        )


@router.delete("/reports/{report_id}", status_code=204)
async def delete_scheduled_report(
    report_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_DELETE))],
    user: RequiredUser,
) -> None:
    """Delete a scheduled report.

    Requires ANALYTICS_DELETE permission.

    Args:
        report_id: Report ID.
        user: Authenticated user (required).
    """
    db = AnalyticsDatabase()
    await db.create_tables()

    async with db.session() as session:
        repo = ScheduledReportsRepository(session)
        deleted = await repo.delete_report(report_id)
        await session.commit()

        if not deleted:
            raise HTTPException(status_code=404, detail="Report not found")
