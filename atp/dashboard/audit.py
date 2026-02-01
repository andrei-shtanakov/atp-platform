"""Audit Logging module for ATP Dashboard.

This module provides comprehensive audit logging for all sensitive operations
in the ATP Dashboard, including authentication, authorization, data access,
and configuration changes.

Features:
- Structured audit log format with JSON metadata
- Tamper-evident logging with hash chains
- Retention policies for log cleanup
- Support for SIEM export
- Async-first design for high throughput

Audit Event Categories:
- AUTHENTICATION: Login, logout, password changes, token operations
- AUTHORIZATION: Permission checks, role assignments, access denials
- DATA_ACCESS: Read operations on sensitive data
- DATA_MODIFICATION: Create, update, delete operations
- CONFIGURATION: System settings changes, agent configuration
- ADMIN_ACTION: Administrative operations, user management
- SECURITY: Security events, policy violations, anomalies
"""

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Index, Integer, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from atp.dashboard.models import DEFAULT_TENANT_ID, Base, User

# Type variable for generic async context manager
T = TypeVar("T")


class AuditCategory(str, Enum):
    """Categories of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    ADMIN_ACTION = "admin_action"
    SECURITY = "security"


class AuditAction(str, Enum):
    """Specific audit actions within categories."""

    # Authentication actions
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET_REQUEST = "password_reset_request"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKE = "token_revoke"
    SSO_LOGIN = "sso_login"
    SSO_FAILURE = "sso_failure"

    # Authorization actions
    PERMISSION_CHECK = "permission_check"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"

    # Data access actions
    DATA_READ = "data_read"
    DATA_EXPORT = "data_export"
    DATA_SEARCH = "data_search"
    REPORT_GENERATED = "report_generated"

    # Data modification actions
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    BULK_UPDATE = "bulk_update"
    BULK_DELETE = "bulk_delete"

    # Configuration actions
    CONFIG_READ = "config_read"
    CONFIG_UPDATE = "config_update"
    SETTING_CHANGE = "setting_change"
    AGENT_CONFIG_CHANGE = "agent_config_change"
    SUITE_CONFIG_CHANGE = "suite_config_change"

    # Admin actions
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_ACTIVATE = "user_activate"
    USER_DEACTIVATE = "user_deactivate"
    ROLE_CREATE = "role_create"
    ROLE_UPDATE = "role_update"
    ROLE_DELETE = "role_delete"
    TENANT_CREATE = "tenant_create"
    TENANT_UPDATE = "tenant_update"
    TENANT_DELETE = "tenant_delete"
    QUOTA_UPDATE = "quota_update"
    BUDGET_CREATE = "budget_create"
    BUDGET_UPDATE = "budget_update"
    BUDGET_DELETE = "budget_delete"

    # Security actions
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    INVALID_REQUEST = "invalid_request"
    SECURITY_VIOLATION = "security_violation"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"  # Detailed debugging info
    INFO = "info"  # Normal operations
    WARNING = "warning"  # Unusual but not harmful
    ERROR = "error"  # Failed operations
    CRITICAL = "critical"  # Security incidents


class AuditLog(Base):
    """Audit log entry model.

    Stores comprehensive audit information with tamper-evident hashing.
    Each entry includes a hash of its content along with the previous
    entry's hash to form a chain.

    Attributes:
        id: Primary key.
        tenant_id: Tenant this log belongs to.
        timestamp: When the event occurred.
        category: Category of the audit event.
        action: Specific action performed.
        severity: Severity level of the event.
        user_id: ID of the user who performed the action (if applicable).
        username: Username at the time of the event (for historical accuracy).
        ip_address: Client IP address.
        user_agent: Client user agent string.
        resource_type: Type of resource affected (e.g., "agent", "suite").
        resource_id: ID of the affected resource.
        resource_name: Name/identifier of the resource.
        details: JSON-encoded event details.
        previous_hash: Hash of the previous audit entry.
        entry_hash: Hash of this entry's content.
        request_id: Correlation ID for request tracing.
        session_id: User session identifier.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default=DEFAULT_TENANT_ID, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )

    # Event classification
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False, default=AuditSeverity.INFO.value
    )

    # Actor information
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    username: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)  # IPv6
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Resource information
    resource_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resource_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Event details (JSON)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tamper-evident chain
    previous_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Correlation
    request_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    session_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )

    __table_args__ = (
        Index("idx_audit_tenant_timestamp", "tenant_id", "timestamp"),
        Index("idx_audit_category_action", "category", "action"),
        Index("idx_audit_user_timestamp", "user_id", "timestamp"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_severity", "severity", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"AuditLog(id={self.id}, category={self.category!r}, "
            f"action={self.action!r}, user={self.username!r})"
        )


def compute_entry_hash(
    timestamp: datetime,
    category: str,
    action: str,
    user_id: int | None,
    resource_type: str | None,
    resource_id: str | None,
    details: str | None,
    previous_hash: str | None,
) -> str:
    """Compute a SHA-256 hash for an audit entry.

    Creates a tamper-evident hash by including all significant fields
    and the previous entry's hash to form a chain.

    Args:
        timestamp: Event timestamp.
        category: Event category.
        action: Event action.
        user_id: User ID (if applicable).
        resource_type: Type of affected resource.
        resource_id: ID of affected resource.
        details: JSON-encoded event details.
        previous_hash: Hash of previous entry (for chain).

    Returns:
        SHA-256 hash as hexadecimal string.
    """
    content = (
        f"{timestamp.isoformat()}"
        f"{category}"
        f"{action}"
        f"{user_id or ''}"
        f"{resource_type or ''}"
        f"{resource_id or ''}"
        f"{details or ''}"
        f"{previous_hash or ''}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def get_previous_hash(session: AsyncSession, tenant_id: str) -> str | None:
    """Get the hash of the most recent audit entry for chain continuity.

    Args:
        session: Database session.
        tenant_id: Tenant ID for scoping.

    Returns:
        Hash of the most recent entry, or None if no entries exist.
    """
    stmt = (
        select(AuditLog.entry_hash)
        .where(AuditLog.tenant_id == tenant_id)
        .order_by(AuditLog.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    return row


async def audit_log(
    session: AsyncSession,
    category: AuditCategory,
    action: AuditAction,
    *,
    tenant_id: str = DEFAULT_TENANT_ID,
    severity: AuditSeverity = AuditSeverity.INFO,
    user: User | None = None,
    user_id: int | None = None,
    username: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    resource_name: str | None = None,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
) -> AuditLog:
    """Create an audit log entry.

    This is the primary function for logging audit events. It handles
    hash computation and chain maintenance automatically.

    Args:
        session: Database session.
        category: Category of the event.
        action: Specific action performed.
        tenant_id: Tenant ID for multi-tenancy.
        severity: Severity level of the event.
        user: User object (extracts id and username).
        user_id: User ID (alternative to user object).
        username: Username (alternative to user object).
        ip_address: Client IP address.
        user_agent: Client user agent string.
        resource_type: Type of affected resource.
        resource_id: ID of affected resource.
        resource_name: Name of affected resource.
        details: Additional event details as dict.
        request_id: Request correlation ID.
        session_id: User session ID.

    Returns:
        The created AuditLog entry.

    Example:
        await audit_log(
            session,
            AuditCategory.AUTHENTICATION,
            AuditAction.LOGIN_SUCCESS,
            user=current_user,
            ip_address=request.client.host,
            details={"method": "password"},
        )
    """
    timestamp = datetime.utcnow()

    # Extract user info if user object provided
    if user is not None:
        user_id = user.id
        username = user.username

    # Convert resource_id to string
    resource_id_str = str(resource_id) if resource_id is not None else None

    # Serialize details to JSON
    details_json = json.dumps(details) if details else None

    # Get previous hash for chain
    previous_hash = await get_previous_hash(session, tenant_id)

    # Compute entry hash
    entry_hash = compute_entry_hash(
        timestamp=timestamp,
        category=category.value,
        action=action.value,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id_str,
        details=details_json,
        previous_hash=previous_hash,
    )

    # Create audit log entry
    entry = AuditLog(
        tenant_id=tenant_id,
        timestamp=timestamp,
        category=category.value,
        action=action.value,
        severity=severity.value,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
        resource_type=resource_type,
        resource_id=resource_id_str,
        resource_name=resource_name,
        details=details_json,
        previous_hash=previous_hash,
        entry_hash=entry_hash,
        request_id=request_id,
        session_id=session_id,
    )

    session.add(entry)
    await session.flush()  # Ensure ID is assigned for return
    return entry


async def verify_audit_chain(
    session: AsyncSession,
    tenant_id: str = DEFAULT_TENANT_ID,
    *,
    start_id: int | None = None,
    end_id: int | None = None,
) -> tuple[bool, list[int]]:
    """Verify the integrity of the audit log chain.

    Checks that each entry's hash matches its computed value and that
    the chain of previous_hash references is intact.

    Args:
        session: Database session.
        tenant_id: Tenant ID for scoping.
        start_id: Starting entry ID (inclusive).
        end_id: Ending entry ID (inclusive).

    Returns:
        Tuple of (is_valid, list of invalid entry IDs).
    """
    stmt = select(AuditLog).where(AuditLog.tenant_id == tenant_id).order_by(AuditLog.id)

    if start_id is not None:
        stmt = stmt.where(AuditLog.id >= start_id)
    if end_id is not None:
        stmt = stmt.where(AuditLog.id <= end_id)

    result = await session.execute(stmt)
    entries = list(result.scalars().all())

    if not entries:
        return True, []

    invalid_entries: list[int] = []
    previous_hash: str | None = None

    for entry in entries:
        # Verify hash chain continuity
        if entry.previous_hash != previous_hash:
            invalid_entries.append(entry.id)

        # Verify entry hash
        computed_hash = compute_entry_hash(
            timestamp=entry.timestamp,
            category=entry.category,
            action=entry.action,
            user_id=entry.user_id,
            resource_type=entry.resource_type,
            resource_id=entry.resource_id,
            details=entry.details,
            previous_hash=entry.previous_hash,
        )

        if entry.entry_hash != computed_hash:
            if entry.id not in invalid_entries:
                invalid_entries.append(entry.id)

        previous_hash = entry.entry_hash

    return len(invalid_entries) == 0, invalid_entries


class RetentionPolicy(BaseModel):
    """Configuration for audit log retention.

    Attributes:
        default_retention_days: Default retention period in days.
        category_retention: Per-category retention overrides.
        min_entries_to_keep: Minimum entries to keep regardless of age.
        max_entries: Maximum entries before forced cleanup.
    """

    model_config = ConfigDict(frozen=True)

    default_retention_days: int = Field(
        default=90, ge=1, description="Default retention period in days"
    )
    category_retention: dict[str, int] = Field(
        default_factory=lambda: {
            AuditCategory.AUTHENTICATION.value: 365,  # 1 year for auth events
            AuditCategory.SECURITY.value: 730,  # 2 years for security events
            AuditCategory.ADMIN_ACTION.value: 365,  # 1 year for admin actions
        },
        description="Per-category retention in days",
    )
    min_entries_to_keep: int = Field(
        default=1000, ge=0, description="Minimum entries to keep regardless of age"
    )
    max_entries: int = Field(
        default=1000000, ge=1000, description="Maximum entries before forced cleanup"
    )


# Default retention policy
DEFAULT_RETENTION_POLICY = RetentionPolicy()


async def apply_retention_policy(
    session: AsyncSession,
    tenant_id: str = DEFAULT_TENANT_ID,
    policy: RetentionPolicy | None = None,
) -> int:
    """Apply retention policy to clean up old audit logs.

    Removes audit entries older than the retention period while
    respecting minimum entry counts. Different categories may have
    different retention periods.

    Args:
        session: Database session.
        tenant_id: Tenant ID for scoping.
        policy: Retention policy to apply (uses default if None).

    Returns:
        Number of entries deleted.
    """
    if policy is None:
        policy = DEFAULT_RETENTION_POLICY

    # Get total entry count
    count_stmt = select(AuditLog.id).where(AuditLog.tenant_id == tenant_id)
    count_result = await session.execute(count_stmt)
    total_count = len(count_result.all())

    # Skip if we're at or below minimum
    if total_count <= policy.min_entries_to_keep:
        return 0

    deleted_count = 0
    cutoff_date = datetime.utcnow()

    # Process each category with its specific retention
    for category in AuditCategory:
        retention_days = policy.category_retention.get(
            category.value, policy.default_retention_days
        )
        category_cutoff = cutoff_date - timedelta(days=retention_days)

        # Delete entries older than cutoff
        delete_stmt = (
            delete(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .where(AuditLog.category == category.value)
            .where(AuditLog.timestamp < category_cutoff)
        )

        result = await session.execute(delete_stmt)
        deleted_count += result.rowcount or 0  # pyrefly: ignore[missing-attribute]

    # Force cleanup if over max entries
    count_stmt = select(AuditLog.id).where(AuditLog.tenant_id == tenant_id)
    count_result = await session.execute(count_stmt)
    current_count = len(count_result.all())

    if current_count > policy.max_entries:
        # Delete oldest entries to get down to max_entries
        excess = current_count - policy.max_entries

        # Get IDs of oldest entries
        oldest_stmt = (
            select(AuditLog.id)
            .where(AuditLog.tenant_id == tenant_id)
            .order_by(AuditLog.timestamp)
            .limit(excess)
        )
        oldest_result = await session.execute(oldest_stmt)
        oldest_ids = [row[0] for row in oldest_result.all()]

        if oldest_ids:
            delete_oldest = delete(AuditLog).where(AuditLog.id.in_(oldest_ids))
            result = await session.execute(delete_oldest)
            deleted_count += result.rowcount or 0  # pyrefly: ignore[missing-attribute]

    return deleted_count


# Pydantic schemas for API responses


class AuditLogResponse(BaseModel):
    """Schema for audit log response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    tenant_id: str
    timestamp: datetime
    category: str
    action: str
    severity: str
    user_id: int | None = None
    username: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    resource_name: str | None = None
    details: dict[str, Any] | None = None
    request_id: str | None = None
    session_id: str | None = None

    @classmethod
    def from_model(cls, model: AuditLog) -> "AuditLogResponse":
        """Create response from model, parsing JSON details."""
        details = None
        if model.details:
            try:
                details = json.loads(model.details)
            except json.JSONDecodeError:
                details = {"raw": model.details}

        return cls(
            id=model.id,
            tenant_id=model.tenant_id,
            timestamp=model.timestamp,
            category=model.category,
            action=model.action,
            severity=model.severity,
            user_id=model.user_id,
            username=model.username,
            ip_address=model.ip_address,
            user_agent=model.user_agent,
            resource_type=model.resource_type,
            resource_id=model.resource_id,
            resource_name=model.resource_name,
            details=details,
            request_id=model.request_id,
            session_id=model.session_id,
        )


class AuditLogList(BaseModel):
    """Paginated list of audit logs."""

    total: int
    items: list[AuditLogResponse]
    limit: int
    offset: int


class AuditLogFilter(BaseModel):
    """Filter parameters for audit log queries."""

    category: AuditCategory | None = None
    action: AuditAction | None = None
    severity: AuditSeverity | None = None
    user_id: int | None = None
    username: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    request_id: str | None = None
    session_id: str | None = None


class AuditChainVerificationResponse(BaseModel):
    """Response for audit chain verification."""

    is_valid: bool
    invalid_entries: list[int]
    entries_checked: int
    message: str


class RetentionPolicyResponse(BaseModel):
    """Response for retention policy status."""

    default_retention_days: int
    category_retention: dict[str, int]
    min_entries_to_keep: int
    max_entries: int


class RetentionExecutionResponse(BaseModel):
    """Response for retention policy execution."""

    entries_deleted: int
    policy_applied: RetentionPolicyResponse


class AuditExportRequest(BaseModel):
    """Request for exporting audit logs."""

    format: str = Field(default="json", pattern="^(json|csv|syslog)$")
    start_date: datetime | None = None
    end_date: datetime | None = None
    categories: list[str] | None = None
    include_details: bool = True


class AuditStatistics(BaseModel):
    """Statistics about audit logs."""

    total_entries: int
    entries_by_category: dict[str, int]
    entries_by_severity: dict[str, int]
    entries_by_day: list[dict[str, Any]]
    oldest_entry: datetime | None = None
    newest_entry: datetime | None = None


# Query helper functions


async def query_audit_logs(
    session: AsyncSession,
    tenant_id: str = DEFAULT_TENANT_ID,
    filters: AuditLogFilter | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[AuditLog], int]:
    """Query audit logs with filtering and pagination.

    Args:
        session: Database session.
        tenant_id: Tenant ID for scoping.
        filters: Optional filter parameters.
        offset: Number of records to skip.
        limit: Maximum records to return.

    Returns:
        Tuple of (list of AuditLog entries, total count).
    """
    base_stmt = select(AuditLog).where(AuditLog.tenant_id == tenant_id)

    if filters:
        if filters.category:
            base_stmt = base_stmt.where(AuditLog.category == filters.category.value)
        if filters.action:
            base_stmt = base_stmt.where(AuditLog.action == filters.action.value)
        if filters.severity:
            base_stmt = base_stmt.where(AuditLog.severity == filters.severity.value)
        if filters.user_id:
            base_stmt = base_stmt.where(AuditLog.user_id == filters.user_id)
        if filters.username:
            base_stmt = base_stmt.where(
                AuditLog.username.ilike(f"%{filters.username}%")
            )
        if filters.resource_type:
            base_stmt = base_stmt.where(AuditLog.resource_type == filters.resource_type)
        if filters.resource_id:
            base_stmt = base_stmt.where(AuditLog.resource_id == filters.resource_id)
        if filters.start_date:
            base_stmt = base_stmt.where(AuditLog.timestamp >= filters.start_date)
        if filters.end_date:
            base_stmt = base_stmt.where(AuditLog.timestamp <= filters.end_date)
        if filters.request_id:
            base_stmt = base_stmt.where(AuditLog.request_id == filters.request_id)
        if filters.session_id:
            base_stmt = base_stmt.where(AuditLog.session_id == filters.session_id)

    # Get total count
    count_stmt = select(AuditLog.id).where(AuditLog.tenant_id == tenant_id)
    if filters:
        # Apply same filters for count
        if filters.category:
            count_stmt = count_stmt.where(AuditLog.category == filters.category.value)
        if filters.action:
            count_stmt = count_stmt.where(AuditLog.action == filters.action.value)
        if filters.severity:
            count_stmt = count_stmt.where(AuditLog.severity == filters.severity.value)
        if filters.user_id:
            count_stmt = count_stmt.where(AuditLog.user_id == filters.user_id)
        if filters.username:
            count_stmt = count_stmt.where(
                AuditLog.username.ilike(f"%{filters.username}%")
            )
        if filters.resource_type:
            count_stmt = count_stmt.where(
                AuditLog.resource_type == filters.resource_type
            )
        if filters.resource_id:
            count_stmt = count_stmt.where(AuditLog.resource_id == filters.resource_id)
        if filters.start_date:
            count_stmt = count_stmt.where(AuditLog.timestamp >= filters.start_date)
        if filters.end_date:
            count_stmt = count_stmt.where(AuditLog.timestamp <= filters.end_date)
        if filters.request_id:
            count_stmt = count_stmt.where(AuditLog.request_id == filters.request_id)
        if filters.session_id:
            count_stmt = count_stmt.where(AuditLog.session_id == filters.session_id)

    count_result = await session.execute(count_stmt)
    total = len(count_result.all())

    # Get paginated results
    query_stmt = (
        base_stmt.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit)
    )
    result = await session.execute(query_stmt)
    entries = list(result.scalars().all())

    return entries, total


async def get_audit_statistics(
    session: AsyncSession,
    tenant_id: str = DEFAULT_TENANT_ID,
    days: int = 30,
) -> AuditStatistics:
    """Get statistics about audit logs.

    Args:
        session: Database session.
        tenant_id: Tenant ID for scoping.
        days: Number of days to include in statistics.

    Returns:
        AuditStatistics with aggregated data.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Total entries
    total_stmt = select(AuditLog).where(AuditLog.tenant_id == tenant_id)
    total_result = await session.execute(total_stmt)
    total_entries = len(total_result.all())

    # Entries by category
    entries_by_category: dict[str, int] = {}
    for category in AuditCategory:
        cat_stmt = (
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .where(AuditLog.category == category.value)
            .where(AuditLog.timestamp >= cutoff)
        )
        cat_result = await session.execute(cat_stmt)
        entries_by_category[category.value] = len(cat_result.all())

    # Entries by severity
    entries_by_severity: dict[str, int] = {}
    for severity in AuditSeverity:
        sev_stmt = (
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .where(AuditLog.severity == severity.value)
            .where(AuditLog.timestamp >= cutoff)
        )
        sev_result = await session.execute(sev_stmt)
        entries_by_severity[severity.value] = len(sev_result.all())

    # Oldest and newest entries
    oldest_stmt = (
        select(AuditLog.timestamp)
        .where(AuditLog.tenant_id == tenant_id)
        .order_by(AuditLog.timestamp)
        .limit(1)
    )
    oldest_result = await session.execute(oldest_stmt)
    oldest_entry = oldest_result.scalar_one_or_none()

    newest_stmt = (
        select(AuditLog.timestamp)
        .where(AuditLog.tenant_id == tenant_id)
        .order_by(AuditLog.timestamp.desc())
        .limit(1)
    )
    newest_result = await session.execute(newest_stmt)
    newest_entry = newest_result.scalar_one_or_none()

    # Simple daily aggregation (last 30 days)
    entries_by_day: list[dict[str, Any]] = []
    for i in range(days):
        day_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=i)
        day_end = day_start + timedelta(days=1)

        day_stmt = (
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .where(AuditLog.timestamp >= day_start)
            .where(AuditLog.timestamp < day_end)
        )
        day_result = await session.execute(day_stmt)
        count = len(day_result.all())

        entries_by_day.append({"date": day_start.date().isoformat(), "count": count})

    entries_by_day.reverse()  # Oldest first

    return AuditStatistics(
        total_entries=total_entries,
        entries_by_category=entries_by_category,
        entries_by_severity=entries_by_severity,
        entries_by_day=entries_by_day,
        oldest_entry=oldest_entry,
        newest_entry=newest_entry,
    )


def create_audit_context(
    user: User | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create a context dictionary for audit logging.

    Helper function to create consistent audit context from request data.

    Args:
        user: Current user.
        ip_address: Client IP.
        user_agent: Client user agent.
        request_id: Request correlation ID.
        session_id: User session ID.

    Returns:
        Dictionary with context fields.
    """
    return {
        "user": user,
        "user_id": user.id if user else None,
        "username": user.username if user else None,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "request_id": request_id,
        "session_id": session_id,
    }


# Decorator for automatic audit logging


def audit_endpoint(
    category: AuditCategory,
    action: AuditAction,
    resource_type: str | None = None,
    severity: AuditSeverity = AuditSeverity.INFO,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory for automatic endpoint audit logging.

    This decorator automatically logs audit events for decorated endpoints.
    It extracts user and request information from FastAPI dependencies.

    Args:
        category: Audit event category.
        action: Audit event action.
        resource_type: Type of resource being accessed.
        severity: Severity level.

    Returns:
        Decorator function.

    Example:
        @router.post("/agents")
        @audit_endpoint(
            AuditCategory.DATA_MODIFICATION,
            AuditAction.DATA_CREATE,
            resource_type="agent"
        )
        async def create_agent(...):
            ...
    """
    import functools

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # The actual logging is done by the audit middleware
            # This decorator just marks the endpoint for audit
            return await func(*args, **kwargs)

        # Store audit metadata on the function
        wrapper._audit_category = category  # type: ignore[attr-defined]
        wrapper._audit_action = action  # type: ignore[attr-defined]
        wrapper._audit_resource_type = resource_type  # type: ignore[attr-defined]
        wrapper._audit_severity = severity  # type: ignore[attr-defined]

        return wrapper

    return decorator
