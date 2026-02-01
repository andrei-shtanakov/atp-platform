"""Audit Middleware for ATP Dashboard.

This middleware automatically logs audit events for all state-changing
HTTP operations (POST, PUT, PATCH, DELETE) and sensitive GET requests.

Features:
- Automatic detection of state-changing operations
- Extraction of user context from authenticated requests
- Request/response correlation via request IDs
- Resource identification from URL paths
- Integration with the audit logging system
"""

import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from atp.dashboard.audit import (
    AuditAction,
    AuditCategory,
    AuditSeverity,
    audit_log,
)
from atp.dashboard.database import get_database
from atp.dashboard.models import DEFAULT_TENANT_ID, User

# Resource type mapping from URL patterns
RESOURCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/api(?:/v\d+)?/agents(?:/(\d+))?"), "agent"),
    (re.compile(r"/api(?:/v\d+)?/suites(?:/(\d+))?"), "suite"),
    (re.compile(r"/api(?:/v\d+)?/tests(?:/(\d+))?"), "test"),
    (re.compile(r"/api(?:/v\d+)?/executions(?:/(\d+))?"), "execution"),
    (re.compile(r"/api(?:/v\d+)?/definitions(?:/(\d+))?"), "suite_definition"),
    (re.compile(r"/api(?:/v\d+)?/users(?:/(\d+))?"), "user"),
    (re.compile(r"/api(?:/v\d+)?/roles(?:/(\d+))?"), "role"),
    (re.compile(r"/api(?:/v\d+)?/tenants(?:/([^/]+))?"), "tenant"),
    (re.compile(r"/api(?:/v\d+)?/budgets(?:/(\d+))?"), "budget"),
    (re.compile(r"/api(?:/v\d+)?/baselines(?:/([^/]+))?"), "baseline"),
    (re.compile(r"/api(?:/v\d+)?/settings(?:/([^/]+))?"), "setting"),
    (re.compile(r"/api(?:/v\d+)?/analytics(?:/([^/]+))?"), "analytics"),
    (re.compile(r"/api(?:/v\d+)?/audit(?:/(\d+))?"), "audit"),
]

# Paths that should not be audited
EXEMPT_PATHS: set[str] = {
    "/health",
    "/healthz",
    "/ready",
    "/version",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/health",
    "/api/version",
    "/metrics",
    "/api/metrics",
}

# Auth-related paths for special handling
AUTH_PATHS: set[str] = {
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/register",
    "/api/auth/refresh",
    "/api/auth/token",
    "/api/v2/auth/login",
    "/api/v2/auth/logout",
    "/api/v2/auth/register",
    "/api/v2/auth/refresh",
    "/api/v2/auth/token",
    "/api/sso/callback",
    "/api/v2/sso/callback",
}


def extract_resource_info(path: str) -> tuple[str | None, str | None]:
    """Extract resource type and ID from URL path.

    Args:
        path: The URL path.

    Returns:
        Tuple of (resource_type, resource_id) or (None, None) if not matched.
    """
    for pattern, resource_type in RESOURCE_PATTERNS:
        match = pattern.match(path)
        if match:
            resource_id = match.group(1) if match.lastindex else None
            return resource_type, resource_id
    return None, None


def get_action_for_method(
    method: str,
    path: str,
    status_code: int,
) -> tuple[AuditCategory, AuditAction, AuditSeverity]:
    """Determine audit action based on HTTP method and path.

    Args:
        method: HTTP method (GET, POST, etc.).
        path: Request path.
        status_code: Response status code.

    Returns:
        Tuple of (category, action, severity).
    """
    # Handle auth paths specially
    if path in AUTH_PATHS:
        if "login" in path:
            if 200 <= status_code < 300:
                return (
                    AuditCategory.AUTHENTICATION,
                    AuditAction.LOGIN_SUCCESS,
                    AuditSeverity.INFO,
                )
            else:
                return (
                    AuditCategory.AUTHENTICATION,
                    AuditAction.LOGIN_FAILURE,
                    AuditSeverity.WARNING,
                )
        elif "logout" in path:
            return (
                AuditCategory.AUTHENTICATION,
                AuditAction.LOGOUT,
                AuditSeverity.INFO,
            )
        elif "register" in path:
            if 200 <= status_code < 300:
                return (
                    AuditCategory.ADMIN_ACTION,
                    AuditAction.USER_CREATE,
                    AuditSeverity.INFO,
                )
            else:
                return (
                    AuditCategory.AUTHENTICATION,
                    AuditAction.LOGIN_FAILURE,
                    AuditSeverity.WARNING,
                )
        elif "refresh" in path or "token" in path:
            return (
                AuditCategory.AUTHENTICATION,
                AuditAction.TOKEN_REFRESH,
                AuditSeverity.DEBUG,
            )
        elif "sso" in path:
            if 200 <= status_code < 300:
                return (
                    AuditCategory.AUTHENTICATION,
                    AuditAction.SSO_LOGIN,
                    AuditSeverity.INFO,
                )
            else:
                return (
                    AuditCategory.AUTHENTICATION,
                    AuditAction.SSO_FAILURE,
                    AuditSeverity.WARNING,
                )

    # Standard CRUD operations
    if method == "GET":
        # Only audit sensitive data access
        if "/export" in path or "/download" in path:
            return (
                AuditCategory.DATA_ACCESS,
                AuditAction.DATA_EXPORT,
                AuditSeverity.INFO,
            )
        elif "/search" in path:
            return (
                AuditCategory.DATA_ACCESS,
                AuditAction.DATA_SEARCH,
                AuditSeverity.DEBUG,
            )
        else:
            return (
                AuditCategory.DATA_ACCESS,
                AuditAction.DATA_READ,
                AuditSeverity.DEBUG,
            )

    elif method == "POST":
        if "/execute" in path or "/run" in path:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.DATA_CREATE,
                AuditSeverity.INFO,
            )
        elif "/bulk" in path:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.BULK_UPDATE,
                AuditSeverity.INFO,
            )
        else:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.DATA_CREATE,
                AuditSeverity.INFO,
            )

    elif method in ("PUT", "PATCH"):
        if "/config" in path or "/settings" in path:
            return (
                AuditCategory.CONFIGURATION,
                AuditAction.CONFIG_UPDATE,
                AuditSeverity.INFO,
            )
        elif "/bulk" in path:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.BULK_UPDATE,
                AuditSeverity.INFO,
            )
        else:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.DATA_UPDATE,
                AuditSeverity.INFO,
            )

    elif method == "DELETE":
        if "/bulk" in path:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.BULK_DELETE,
                AuditSeverity.WARNING,
            )
        else:
            return (
                AuditCategory.DATA_MODIFICATION,
                AuditAction.DATA_DELETE,
                AuditSeverity.WARNING,
            )

    # Default
    return (
        AuditCategory.DATA_ACCESS,
        AuditAction.DATA_READ,
        AuditSeverity.DEBUG,
    )


def should_audit_request(
    method: str,
    path: str,
    *,
    audit_get_requests: bool = False,
) -> bool:
    """Determine if a request should be audited.

    Args:
        method: HTTP method.
        path: Request path.
        audit_get_requests: Whether to audit GET requests.

    Returns:
        True if the request should be audited.
    """
    # Skip exempt paths
    if path in EXEMPT_PATHS or any(path.startswith(p) for p in EXEMPT_PATHS):
        return False

    # Skip OPTIONS requests
    if method == "OPTIONS":
        return False

    # Always audit auth paths
    if path in AUTH_PATHS:
        return True

    # Audit state-changing operations
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return True

    # Optionally audit GET requests for sensitive resources
    if audit_get_requests and method == "GET":
        resource_type, _ = extract_resource_info(path)
        # Audit access to sensitive resources
        sensitive_resources = {"user", "role", "tenant", "setting", "audit"}
        return resource_type in sensitive_resources

    return False


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic audit logging.

    This middleware intercepts requests and responses to automatically
    create audit log entries for state-changing operations.

    Attributes:
        audit_get_requests: Whether to audit GET requests (default False).
        exclude_paths: Additional paths to exclude from auditing.
    """

    def __init__(
        self,
        app: ASGIApp,
        audit_get_requests: bool = False,
        exclude_paths: set[str] | None = None,
    ) -> None:
        """Initialize the audit middleware.

        Args:
            app: The ASGI application.
            audit_get_requests: Whether to audit GET requests.
            exclude_paths: Additional paths to exclude.
        """
        super().__init__(app)
        self.audit_get_requests = audit_get_requests
        self.exclude_paths = exclude_paths or set()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and log audit events.

        Args:
            request: The incoming request.
            call_next: The next handler in the chain.

        Returns:
            The response from the handler.
        """
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        path = request.url.path
        method = request.method

        # Check if we should audit this request
        if not should_audit_request(
            method, path, audit_get_requests=self.audit_get_requests
        ):
            return await call_next(request)

        # Skip additional excluded paths
        if path in self.exclude_paths:
            return await call_next(request)

        # Record start time
        start_time = datetime.utcnow()

        # Execute the request
        response = await call_next(request)

        # Extract information for audit
        try:
            await self._create_audit_entry(request, response, request_id, start_time)
        except Exception:
            # Don't let audit failures break the request
            # In production, you might want to log this error
            pass

        return response

    async def _create_audit_entry(
        self,
        request: Request,
        response: Response,
        request_id: str,
        start_time: datetime,
    ) -> None:
        """Create an audit log entry for the request.

        Args:
            request: The request object.
            response: The response object.
            request_id: The request correlation ID.
            start_time: When the request started.
        """
        path = request.url.path
        method = request.method
        status_code = response.status_code

        # Get category, action, and severity
        category, action, severity = get_action_for_method(method, path, status_code)

        # Upgrade severity for errors
        if status_code >= 500:
            severity = AuditSeverity.ERROR
        elif status_code >= 400:
            if severity == AuditSeverity.DEBUG:
                severity = AuditSeverity.INFO

        # Extract resource info
        resource_type, resource_id = extract_resource_info(path)

        # Extract user info from request state (set by auth middleware)
        user: User | None = getattr(request.state, "user", None)
        tenant_id: str = getattr(request.state, "tenant_id", DEFAULT_TENANT_ID)
        session_id: str | None = getattr(request.state, "session_id", None)

        # Get client info
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")[:512]

        # Build details
        details: dict[str, Any] = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            "query_params": dict(request.query_params)
            if request.query_params
            else None,
        }

        # Don't log request body for security (might contain passwords)
        # But log path parameters
        if request.path_params:
            details["path_params"] = dict(request.path_params)

        # Get database session
        db = get_database()
        async with db.session() as session:
            await audit_log(
                session,
                category,
                action,
                tenant_id=tenant_id,
                severity=severity,
                user=user,
                ip_address=ip_address,
                user_agent=user_agent,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                request_id=request_id,
                session_id=session_id,
            )
            await session.commit()

    def _get_client_ip(self, request: Request) -> str | None:
        """Extract the real client IP address.

        Handles X-Forwarded-For header for proxied requests.

        Args:
            request: The request object.

        Returns:
            Client IP address or None.
        """
        # Check X-Forwarded-For header first (for proxied requests)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host

        return None


def get_audit_middleware(
    audit_get_requests: bool = False,
    exclude_paths: set[str] | None = None,
) -> type[AuditMiddleware]:
    """Factory function to create configured audit middleware.

    This is useful for adding the middleware to a FastAPI app with
    specific configuration.

    Args:
        audit_get_requests: Whether to audit GET requests.
        exclude_paths: Paths to exclude from auditing.

    Returns:
        Configured AuditMiddleware class.

    Example:
        app.add_middleware(
            get_audit_middleware(audit_get_requests=True)
        )
    """

    class ConfiguredAuditMiddleware(AuditMiddleware):
        def __init__(self, app: ASGIApp) -> None:
            super().__init__(
                app,
                audit_get_requests=audit_get_requests,
                exclude_paths=exclude_paths,
            )

    return ConfiguredAuditMiddleware
