"""Audit log routes for ATP Dashboard.

This module provides API endpoints for querying, filtering, and managing
audit logs in the ATP Dashboard.

Permissions:
    - GET /audit: AUDIT_READ
    - GET /audit/statistics: AUDIT_READ
    - GET /audit/{id}: AUDIT_READ
    - POST /audit/export: AUDIT_EXPORT
    - GET /audit/verify: AUDIT_READ
    - POST /audit/retention: AUDIT_MANAGE
    - GET /audit/retention: AUDIT_READ
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from atp.dashboard.audit import (
    AuditAction,
    AuditCategory,
    AuditChainVerificationResponse,
    AuditExportRequest,
    AuditLog,
    AuditLogFilter,
    AuditLogList,
    AuditLogResponse,
    AuditSeverity,
    AuditStatistics,
    RetentionExecutionResponse,
    RetentionPolicy,
    RetentionPolicyResponse,
    apply_retention_policy,
    get_audit_statistics,
    query_audit_logs,
    verify_audit_chain,
)
from atp.dashboard.models import DEFAULT_TENANT_ID
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.v2.dependencies import (
    DBSession,
    Pagination,
)

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=AuditLogList)
async def list_audit_logs(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
    pagination: Pagination,
    category: AuditCategory | None = None,
    action: AuditAction | None = None,
    severity: AuditSeverity | None = None,
    user_id: int | None = None,
    username: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    request_id: str | None = None,
    session_id: str | None = Query(None, alias="audit_session_id"),
) -> AuditLogList:
    """List audit logs with filtering and pagination.

    Requires AUDIT_READ permission.

    Args:
        session: Database session.
        pagination: Pagination parameters.
        category: Filter by event category.
        action: Filter by event action.
        severity: Filter by severity level.
        user_id: Filter by user ID.
        username: Filter by username (partial match).
        resource_type: Filter by resource type.
        resource_id: Filter by resource ID.
        start_date: Filter by start date (inclusive).
        end_date: Filter by end date (inclusive).
        request_id: Filter by request correlation ID.
        session_id: Filter by user session ID.

    Returns:
        Paginated list of audit log entries.
    """
    # Build filter
    filters = AuditLogFilter(
        category=category,
        action=action,
        severity=severity,
        user_id=user_id,
        username=username,
        resource_type=resource_type,
        resource_id=resource_id,
        start_date=start_date,
        end_date=end_date,
        request_id=request_id,
        session_id=session_id,
    )

    # Query audit logs
    entries, total = await query_audit_logs(
        session,
        tenant_id=DEFAULT_TENANT_ID,
        filters=filters,
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return AuditLogList(
        total=total,
        items=[AuditLogResponse.from_model(e) for e in entries],
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/statistics", response_model=AuditStatistics)
async def get_statistics(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
    days: int = Query(default=30, ge=1, le=365),
) -> AuditStatistics:
    """Get audit log statistics.

    Requires AUDIT_READ permission.

    Args:
        session: Database session.
        days: Number of days to include in statistics.

    Returns:
        Aggregated statistics about audit logs.
    """
    return await get_audit_statistics(
        session,
        tenant_id=DEFAULT_TENANT_ID,
        days=days,
    )


@router.get("/categories", response_model=list[dict[str, str]])
async def list_categories(
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
) -> list[dict[str, str]]:
    """List all audit categories.

    Requires AUDIT_READ permission.

    Returns:
        List of category names and descriptions.
    """
    return [{"value": cat.value, "name": cat.name} for cat in AuditCategory]


@router.get("/actions", response_model=list[dict[str, str]])
async def list_actions(
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
    category: AuditCategory | None = None,
) -> list[dict[str, str]]:
    """List all audit actions.

    Requires AUDIT_READ permission.

    Args:
        category: Filter actions by category.

    Returns:
        List of action names and descriptions.
    """
    actions = list(AuditAction)

    # Optionally filter by category
    if category:
        # Map categories to action prefixes
        category_prefixes: dict[AuditCategory, list[str]] = {
            AuditCategory.AUTHENTICATION: [
                "login",
                "logout",
                "password",
                "token",
                "sso",
            ],
            AuditCategory.AUTHORIZATION: ["permission", "role"],
            AuditCategory.DATA_ACCESS: [
                "data_read",
                "data_export",
                "data_search",
                "report",
            ],
            AuditCategory.DATA_MODIFICATION: [
                "data_create",
                "data_update",
                "data_delete",
                "bulk",
            ],
            AuditCategory.CONFIGURATION: [
                "config",
                "setting",
                "agent_config",
                "suite_config",
            ],
            AuditCategory.ADMIN_ACTION: ["user", "tenant", "quota", "budget"],
            AuditCategory.SECURITY: ["suspicious", "rate_limit", "invalid", "security"],
        }

        prefixes = category_prefixes.get(category, [])
        actions = [
            action
            for action in actions
            if any(action.value.startswith(prefix) for prefix in prefixes)
        ]

    return [{"value": action.value, "name": action.name} for action in actions]


@router.get("/severities", response_model=list[dict[str, str]])
async def list_severities(
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
) -> list[dict[str, str]]:
    """List all severity levels.

    Requires AUDIT_READ permission.

    Returns:
        List of severity levels and descriptions.
    """
    descriptions = {
        AuditSeverity.DEBUG: "Detailed debugging information",
        AuditSeverity.INFO: "Normal operations",
        AuditSeverity.WARNING: "Unusual but not harmful",
        AuditSeverity.ERROR: "Failed operations",
        AuditSeverity.CRITICAL: "Security incidents",
    }

    return [
        {"value": sev.value, "name": sev.name, "description": descriptions[sev]}
        for sev in AuditSeverity
    ]


@router.get("/verify", response_model=AuditChainVerificationResponse)
async def verify_chain(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
    start_id: int | None = None,
    end_id: int | None = None,
) -> AuditChainVerificationResponse:
    """Verify the integrity of the audit log chain.

    Checks that audit entries have not been tampered with by verifying
    the hash chain. This is important for compliance and security audits.

    Requires AUDIT_READ permission.

    Args:
        session: Database session.
        start_id: Starting entry ID (inclusive).
        end_id: Ending entry ID (inclusive).

    Returns:
        Verification result including any invalid entries.
    """
    is_valid, invalid_entries = await verify_audit_chain(
        session,
        tenant_id=DEFAULT_TENANT_ID,
        start_id=start_id,
        end_id=end_id,
    )

    # Count entries in range
    from sqlalchemy import select

    stmt = select(AuditLog.id).where(AuditLog.tenant_id == DEFAULT_TENANT_ID)
    if start_id is not None:
        stmt = stmt.where(AuditLog.id >= start_id)
    if end_id is not None:
        stmt = stmt.where(AuditLog.id <= end_id)
    result = await session.execute(stmt)
    entries_checked = len(result.all())

    message = (
        "Audit chain integrity verified"
        if is_valid
        else (f"Found {len(invalid_entries)} entries with integrity issues")
    )

    return AuditChainVerificationResponse(
        is_valid=is_valid,
        invalid_entries=invalid_entries,
        entries_checked=entries_checked,
        message=message,
    )


@router.get("/{audit_id}", response_model=AuditLogResponse)
async def get_audit_entry(
    session: DBSession,
    audit_id: int,
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
) -> AuditLogResponse:
    """Get a specific audit log entry by ID.

    Requires AUDIT_READ permission.

    Args:
        session: Database session.
        audit_id: Audit log entry ID.

    Returns:
        The requested audit log entry.

    Raises:
        HTTPException: If entry not found.
    """
    entry = await session.get(AuditLog, audit_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit log entry {audit_id} not found",
        )
    return AuditLogResponse.from_model(entry)


@router.post("/export")
async def export_audit_logs(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_EXPORT))],
    export_request: AuditExportRequest,
) -> StreamingResponse:
    """Export audit logs in the specified format.

    Supports JSON, CSV, and syslog formats for SIEM integration.

    Requires AUDIT_EXPORT permission.

    Args:
        session: Database session.
        export_request: Export parameters including format and filters.

    Returns:
        Streaming response with exported data.
    """
    import csv
    import io

    # Build filters from export request
    filters = AuditLogFilter(
        start_date=export_request.start_date,
        end_date=export_request.end_date,
    )

    # If categories specified, we'll filter manually
    entries, _total = await query_audit_logs(
        session,
        tenant_id=DEFAULT_TENANT_ID,
        filters=filters,
        offset=0,
        limit=100000,  # Large limit for export
    )

    # Filter by categories if specified
    if export_request.categories:
        entries = [e for e in entries if e.category in export_request.categories]

    if export_request.format == "json":
        # JSON export
        import json

        def generate_json():
            yield "["
            first = True
            for entry in entries:
                if not first:
                    yield ","
                first = False
                response = AuditLogResponse.from_model(entry)
                yield json.dumps(response.model_dump(), default=str)
            yield "]"

        return StreamingResponse(
            generate_json(),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=audit_logs.json"},
        )

    elif export_request.format == "csv":
        # CSV export
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        headers = [
            "id",
            "timestamp",
            "category",
            "action",
            "severity",
            "user_id",
            "username",
            "ip_address",
            "resource_type",
            "resource_id",
            "resource_name",
            "request_id",
        ]
        if export_request.include_details:
            headers.append("details")
        writer.writerow(headers)

        # Data rows
        for entry in entries:
            row = [
                entry.id,
                entry.timestamp.isoformat(),
                entry.category,
                entry.action,
                entry.severity,
                entry.user_id or "",
                entry.username or "",
                entry.ip_address or "",
                entry.resource_type or "",
                entry.resource_id or "",
                entry.resource_name or "",
                entry.request_id or "",
            ]
            if export_request.include_details:
                row.append(entry.details or "")
            writer.writerow(row)

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
        )

    elif export_request.format == "syslog":
        # Syslog format (RFC 5424)
        def generate_syslog():
            severity_map = {
                "debug": 7,
                "info": 6,
                "warning": 4,
                "error": 3,
                "critical": 2,
            }

            for entry in entries:
                pri = severity_map.get(entry.severity, 6)
                timestamp = entry.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                hostname = "atp-dashboard"
                app_name = "audit"
                proc_id = entry.request_id or "-"
                msg_id = entry.action

                structured_data = (
                    f'[meta category="{entry.category}" '
                    f'user="{entry.username or "-"}" '
                    f'resource="{entry.resource_type or "-"}"]'
                )

                message = (
                    f"{entry.action}: "
                    f"{entry.resource_type or 'unknown'} "
                    f"{entry.resource_id or ''}"
                ).strip()

                yield (
                    f"<{pri}>1 {timestamp} {hostname} {app_name} "
                    f"{proc_id} {msg_id} {structured_data} {message}\n"
                )

        return StreamingResponse(
            generate_syslog(),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=audit_logs.syslog"},
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported export format: {export_request.format}",
    )


@router.get("/retention/policy", response_model=RetentionPolicyResponse)
async def get_retention_policy(
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_READ))],
) -> RetentionPolicyResponse:
    """Get the current audit log retention policy.

    Requires AUDIT_READ permission.

    Returns:
        Current retention policy configuration.
    """
    from atp.dashboard.audit import DEFAULT_RETENTION_POLICY

    return RetentionPolicyResponse(
        default_retention_days=DEFAULT_RETENTION_POLICY.default_retention_days,
        category_retention=DEFAULT_RETENTION_POLICY.category_retention,
        min_entries_to_keep=DEFAULT_RETENTION_POLICY.min_entries_to_keep,
        max_entries=DEFAULT_RETENTION_POLICY.max_entries,
    )


@router.post("/retention/execute", response_model=RetentionExecutionResponse)
async def execute_retention(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.AUDIT_MANAGE))],
    policy_override: RetentionPolicy | None = None,
) -> RetentionExecutionResponse:
    """Execute the retention policy to clean up old audit logs.

    This operation removes audit entries older than their retention
    period. Different categories may have different retention periods.

    Requires AUDIT_MANAGE permission.

    Args:
        session: Database session.
        policy_override: Optional custom retention policy to apply.

    Returns:
        Number of entries deleted and policy applied.
    """
    from atp.dashboard.audit import DEFAULT_RETENTION_POLICY

    policy = policy_override or DEFAULT_RETENTION_POLICY

    deleted = await apply_retention_policy(
        session,
        tenant_id=DEFAULT_TENANT_ID,
        policy=policy,
    )

    await session.commit()

    return RetentionExecutionResponse(
        entries_deleted=deleted,
        policy_applied=RetentionPolicyResponse(
            default_retention_days=policy.default_retention_days,
            category_retention=policy.category_retention,
            min_entries_to_keep=policy.min_entries_to_keep,
            max_entries=policy.max_entries,
        ),
    )
