"""Quota enforcement middleware for FastAPI.

This module provides FastAPI middleware and dependencies for enforcing
tenant quotas on API requests.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from atp.dashboard.tenancy.quotas import (
    QuotaChecker,
    QuotaExceededError,
    QuotaType,
    QuotaUsageTracker,
)

logger = logging.getLogger(__name__)


# HTTP 429 response details
QUOTA_EXCEEDED_RESPONSE = {
    "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
    "headers": {"Retry-After": "3600"},  # Default 1 hour
}


class QuotaEnforcementMiddleware(BaseHTTPMiddleware):
    """Middleware for enforcing tenant quotas on all requests.

    This middleware checks tenant quotas before processing requests
    and returns HTTP 429 if quotas are exceeded.
    """

    # Paths that should skip quota checking
    EXEMPT_PATHS = frozenset(
        {
            "/api/health",
            "/api/version",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
        }
    )

    # Methods that modify resources and should be quota-checked
    QUOTA_CHECKED_METHODS = frozenset({"POST", "PUT", "PATCH"})

    def __init__(
        self,
        app: Any,
        *,
        check_read_requests: bool = False,
        exempt_paths: set[str] | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The FastAPI/Starlette application.
            check_read_requests: Whether to check GET/HEAD requests too.
            exempt_paths: Additional paths to exempt from quota checking.
        """
        super().__init__(app)
        self._check_read_requests = check_read_requests
        self._exempt_paths = self.EXEMPT_PATHS | (exempt_paths or set())

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request, checking quotas if applicable.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler.

        Returns:
            Response from the handler or a 429 response.
        """
        # Skip quota checks for exempt paths
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        # Skip quota checks for read methods unless configured otherwise
        if not self._check_read_requests:
            if request.method not in self.QUOTA_CHECKED_METHODS:
                return await call_next(request)

        # Get tenant ID from request
        tenant_id = self._get_tenant_id(request)
        if tenant_id is None:
            # No tenant context - skip quota check
            return await call_next(request)

        # Store tenant_id in request state for later use
        request.state.tenant_id = tenant_id

        # Perform quota check
        try:
            quota_result = await self._check_quotas(request, tenant_id)
            if quota_result is not None:
                return quota_result
        except Exception as e:
            # Log error but don't block request on quota check failure
            logger.error(f"Quota check failed for tenant {tenant_id}: {e}")

        return await call_next(request)

    def _get_tenant_id(self, request: Request) -> str | None:
        """Extract tenant ID from request.

        Checks multiple sources in order:
        1. Request state (set by authentication)
        2. Header (X-Tenant-ID)
        3. Query parameter (tenant_id)

        Args:
            request: The incoming request.

        Returns:
            Tenant ID or None.
        """
        # Check request state first (set by auth middleware)
        if hasattr(request.state, "tenant_id"):
            return request.state.tenant_id

        # Check header
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id

        # Check query parameter
        tenant_id = request.query_params.get("tenant_id")
        if tenant_id:
            return tenant_id

        return None

    async def _check_quotas(
        self,
        request: Request,
        tenant_id: str,
    ) -> Response | None:
        """Check quotas for the request.

        Args:
            request: The incoming request.
            tenant_id: The tenant ID.

        Returns:
            JSONResponse with 429 if quota exceeded, None otherwise.
        """
        # Get database session from request state
        if not hasattr(request.state, "db_session"):
            return None

        session = request.state.db_session
        checker = QuotaChecker(session)

        # Determine which quotas to check based on the endpoint
        quota_type = self._get_quota_type_for_path(request.url.path, request.method)
        if quota_type is None:
            return None

        try:
            await checker.enforce_quota(tenant_id, quota_type, additional=1)
        except QuotaExceededError as e:
            return self._create_quota_exceeded_response(e)

        return None

    def _get_quota_type_for_path(
        self,
        path: str,
        method: str,
    ) -> QuotaType | None:
        """Determine which quota type to check for a given path.

        Args:
            path: The request path.
            method: The HTTP method.

        Returns:
            QuotaType to check or None.
        """
        # Map paths to quota types
        if method != "POST":
            return None

        path_mapping = {
            "/api/tests": QuotaType.TESTS_PER_DAY,
            "/api/executions": QuotaType.TESTS_PER_DAY,
            "/api/agents": QuotaType.AGENTS,
            "/api/users": QuotaType.USERS,
            "/api/suites": QuotaType.SUITES,
        }

        for prefix, quota_type in path_mapping.items():
            if path.startswith(prefix):
                return quota_type

        return None

    def _create_quota_exceeded_response(
        self,
        error: QuotaExceededError,
    ) -> JSONResponse:
        """Create a 429 response for quota exceeded.

        Args:
            error: The quota exceeded error.

        Returns:
            JSONResponse with 429 status.
        """
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": error.message,
                "quota_type": error.quota_type.value,
                "current_value": error.current_value,
                "limit_value": error.limit_value,
            },
            headers={"Retry-After": "3600"},
        )


def get_quota_checker(request: Request) -> QuotaChecker | None:
    """Get a QuotaChecker instance from request state.

    Args:
        request: The incoming request.

    Returns:
        QuotaChecker or None if no session available.
    """
    if not hasattr(request.state, "db_session"):
        return None
    return QuotaChecker(request.state.db_session)


def get_quota_tracker(request: Request) -> QuotaUsageTracker | None:
    """Get a QuotaUsageTracker instance from request state.

    Args:
        request: The incoming request.

    Returns:
        QuotaUsageTracker or None if no session available.
    """
    if not hasattr(request.state, "db_session"):
        return None
    return QuotaUsageTracker(request.state.db_session)


async def get_quota_checker_dep(request: Request) -> QuotaChecker:
    """Dependency for getting a QuotaChecker.

    Args:
        request: The incoming request.

    Returns:
        QuotaChecker instance.

    Raises:
        HTTPException: If database session not available.
    """
    checker = get_quota_checker(request)
    if checker is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database session not available",
        )
    return checker


async def get_quota_tracker_dep(request: Request) -> QuotaUsageTracker:
    """Dependency for getting a QuotaUsageTracker.

    Args:
        request: The incoming request.

    Returns:
        QuotaUsageTracker instance.

    Raises:
        HTTPException: If database session not available.
    """
    tracker = get_quota_tracker(request)
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database session not available",
        )
    return tracker


# Type aliases for cleaner route signatures
QuotaCheckerDep = Annotated[QuotaChecker, Depends(get_quota_checker_dep)]
QuotaTrackerDep = Annotated[QuotaUsageTracker, Depends(get_quota_tracker_dep)]


def require_quota(quota_type: QuotaType, additional: int | float = 1):
    """Dependency that enforces a specific quota.

    Use this decorator on routes that consume quota resources.

    Args:
        quota_type: The type of quota to check.
        additional: Amount of quota that will be consumed.

    Returns:
        Dependency function.

    Example:
        @app.post(
            "/tests",
            dependencies=[Depends(require_quota(QuotaType.TESTS_PER_DAY))]
        )
        async def create_test(...):
            ...
    """

    async def _check_quota(
        request: Request,
        checker: QuotaCheckerDep,
    ) -> None:
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id is None:
            # No tenant context - skip check
            return

        try:
            await checker.enforce_quota(tenant_id, quota_type, additional)
        except QuotaExceededError as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=e.message,
                headers={"Retry-After": "3600"},
            ) from e

    return _check_quota


def create_quota_exceeded_response(
    error: QuotaExceededError,
    retry_after: int = 3600,
) -> JSONResponse:
    """Create a standardized quota exceeded response.

    Args:
        error: The quota exceeded error.
        retry_after: Seconds to suggest retrying after.

    Returns:
        JSONResponse with 429 status.
    """
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": error.message,
            "error": "quota_exceeded",
            "quota_type": error.quota_type.value,
            "current_value": error.current_value,
            "limit_value": error.limit_value,
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )
