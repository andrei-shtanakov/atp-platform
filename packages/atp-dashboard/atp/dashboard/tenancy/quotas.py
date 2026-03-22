"""Tenant quota enforcement for multi-tenancy support.

This module provides quota checking and enforcement for tenant resources.
It includes quota types, usage tracking, and middleware for HTTP request handling.

Quota Types:
- tests/day: Maximum tests that can be run per day
- parallel_runs: Maximum concurrent test runs
- storage: Maximum storage in GB
- agents: Maximum number of agents
- budget: Monthly LLM budget in USD
- users: Maximum number of users
- suites: Maximum number of test suites
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.tenancy.models import Tenant, TenantQuotas

logger = logging.getLogger(__name__)

# Cache for storage usage: {tenant_id: (timestamp, value_gb)}
_storage_cache: dict[str, tuple[float, float]] = {}
STORAGE_CACHE_TTL_SECONDS = 300  # 5 minutes


class QuotaType(str, Enum):
    """Types of quotas that can be enforced."""

    TESTS_PER_DAY = "tests_per_day"
    PARALLEL_RUNS = "parallel_runs"
    STORAGE_GB = "storage_gb"
    AGENTS = "agents"
    LLM_BUDGET_MONTHLY = "llm_budget_monthly"
    USERS = "users"
    SUITES = "suites"


class QuotaViolation(BaseModel):
    """Details of a quota violation."""

    model_config = ConfigDict(extra="forbid")

    quota_type: QuotaType
    current_value: float | int
    limit_value: float | int
    message: str


class QuotaCheckResult(BaseModel):
    """Result of a quota check operation."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    passed: bool
    violations: list[QuotaViolation] = Field(default_factory=list)
    warnings: list[QuotaViolation] = Field(
        default_factory=list,
        description="Quotas approaching limit (>80%)",
    )


class QuotaUsage(BaseModel):
    """Current usage for all quota types."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    tests_today: int = 0
    parallel_runs: int = 0
    storage_gb: float = 0.0
    agents: int = 0
    llm_cost_this_month: float = 0.0
    users: int = 0
    suites: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class QuotaExceededError(Exception):
    """Raised when a quota limit has been exceeded."""

    def __init__(
        self,
        quota_type: QuotaType,
        current_value: float | int,
        limit_value: float | int,
        message: str | None = None,
    ) -> None:
        self.quota_type = quota_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.message = message or (
            f"Quota exceeded for {quota_type.value}: {current_value} / {limit_value}"
        )
        super().__init__(self.message)


class QuotaChecker:
    """Checks tenant quotas against current usage.

    This class is responsible for validating that a tenant's resource usage
    is within their allocated quotas.
    """

    WARNING_THRESHOLD = 0.8  # Warn at 80% usage

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the quota checker.

        Args:
            session: Database session for queries.
        """
        self._session = session

    async def get_tenant_quotas(self, tenant_id: str) -> TenantQuotas | None:
        """Get quotas for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            TenantQuotas or None if tenant not found.
        """
        tenant = await self._session.get(Tenant, tenant_id)
        if tenant is None:
            return None
        return tenant.quotas

    async def get_usage(self, tenant_id: str) -> QuotaUsage:
        """Get current resource usage for a tenant.

        This method queries various tables to determine current usage.

        Args:
            tenant_id: The tenant ID.

        Returns:
            QuotaUsage with current values.
        """
        usage = QuotaUsage(tenant_id=tenant_id)

        # Get tests run today
        usage.tests_today = await self._get_tests_today(tenant_id)

        # Get current parallel runs
        usage.parallel_runs = await self._get_parallel_runs(tenant_id)

        # Get storage usage
        usage.storage_gb = await self._get_storage_usage(tenant_id)

        # Get agent count
        usage.agents = await self._get_agent_count(tenant_id)

        # Get LLM cost this month
        usage.llm_cost_this_month = await self._get_llm_cost_this_month(tenant_id)

        # Get user count
        usage.users = await self._get_user_count(tenant_id)

        # Get suite count
        usage.suites = await self._get_suite_count(tenant_id)

        return usage

    async def _get_tests_today(self, tenant_id: str) -> int:
        """Get the number of tests run today for a tenant."""
        try:
            # Import here to avoid circular imports
            from atp.dashboard.models import SuiteExecution

            today_start = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            query = select(func.count(SuiteExecution.id)).where(
                SuiteExecution.tenant_id == tenant_id,
                SuiteExecution.started_at >= today_start,
            )
            result = await self._session.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.debug(f"Error getting tests today for {tenant_id}: {e}")
            return 0

    async def _get_parallel_runs(self, tenant_id: str) -> int:
        """Get the number of currently running tests for a tenant."""
        try:
            from atp.dashboard.models import SuiteExecution

            query = select(func.count(SuiteExecution.id)).where(
                SuiteExecution.tenant_id == tenant_id,
                SuiteExecution.status == "running",
            )
            result = await self._session.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.debug(f"Error getting parallel runs for {tenant_id}: {e}")
            return 0

    async def _get_storage_usage(self, tenant_id: str) -> float:
        """Get storage usage in GB for a tenant.

        Calculates storage from three sources:
        1. Artifact sizes stored in the database
        2. Run result data (response JSON, events) in the database
        3. Trace files on disk (~/.atp/traces/)

        Results are cached for 5 minutes to avoid expensive
        recalculation on every quota check.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Storage usage in GB.
        """
        now = time.monotonic()
        cached = _storage_cache.get(tenant_id)
        if cached is not None:
            cached_time, cached_value = cached
            if now - cached_time < STORAGE_CACHE_TTL_SECONDS:
                return cached_value

        total_bytes = 0

        # 1. Artifact storage from DB
        total_bytes += await self._get_artifact_storage_bytes(tenant_id)

        # 2. Run result data storage from DB
        total_bytes += await self._get_run_result_storage_bytes(tenant_id)

        # 3. Trace file storage on disk
        total_bytes += self._get_trace_storage_bytes(tenant_id)

        total_gb = total_bytes / (1024**3)
        _storage_cache[tenant_id] = (now, total_gb)
        return total_gb

    async def _get_artifact_storage_bytes(self, tenant_id: str) -> int:
        """Get total artifact size in bytes for a tenant."""
        try:
            from atp.dashboard.models import (
                Artifact,
                RunResult,
                SuiteExecution,
                TestExecution,
            )

            query = (
                select(func.coalesce(func.sum(Artifact.size_bytes), 0))
                .join(
                    RunResult,
                    Artifact.run_result_id == RunResult.id,
                )
                .join(
                    TestExecution,
                    RunResult.test_execution_id == TestExecution.id,
                )
                .join(
                    SuiteExecution,
                    TestExecution.suite_execution_id == SuiteExecution.id,
                )
                .where(
                    SuiteExecution.tenant_id == tenant_id,
                )
            )
            result = await self._session.execute(query)
            return int(result.scalar() or 0)
        except Exception as e:
            logger.debug(
                "Error getting artifact storage for %s: %s",
                tenant_id,
                e,
            )
            return 0

    async def _get_run_result_storage_bytes(self, tenant_id: str) -> int:
        """Estimate run result JSON data size for a tenant.

        Uses the DB length of response_json and events_json
        columns to approximate storage.
        """
        try:
            from atp.dashboard.models import (
                RunResult,
                SuiteExecution,
                TestExecution,
            )

            query = (
                select(
                    func.coalesce(
                        func.sum(
                            func.coalesce(
                                func.length(RunResult.response_json),
                                0,
                            )
                            + func.coalesce(
                                func.length(RunResult.events_json),
                                0,
                            )
                        ),
                        0,
                    )
                )
                .join(
                    TestExecution,
                    RunResult.test_execution_id == TestExecution.id,
                )
                .join(
                    SuiteExecution,
                    TestExecution.suite_execution_id == SuiteExecution.id,
                )
                .where(
                    SuiteExecution.tenant_id == tenant_id,
                )
            )
            result = await self._session.execute(query)
            return int(result.scalar() or 0)
        except Exception as e:
            logger.debug(
                "Error getting run result storage for %s: %s",
                tenant_id,
                e,
            )
            return 0

    @staticmethod
    def _get_trace_storage_bytes(tenant_id: str) -> int:
        """Get trace file storage in bytes for a tenant.

        Scans the trace directory for files belonging to the
        given tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Total size in bytes of trace files for the tenant.
        """
        try:
            from atp.tracing.storage import DEFAULT_TRACES_DIR

            traces_dir = DEFAULT_TRACES_DIR / tenant_id
            if not traces_dir.is_dir():
                return 0
            return sum(f.stat().st_size for f in traces_dir.iterdir() if f.is_file())
        except Exception as e:
            logger.debug(
                "Error getting trace storage for %s: %s",
                tenant_id,
                e,
            )
            return 0

    async def _get_agent_count(self, tenant_id: str) -> int:
        """Get the number of agents for a tenant."""
        try:
            from atp.dashboard.models import Agent

            query = select(func.count(Agent.id)).where(Agent.tenant_id == tenant_id)
            result = await self._session.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.debug(f"Error getting agent count for {tenant_id}: {e}")
            return 0

    async def _get_llm_cost_this_month(self, tenant_id: str) -> float:
        """Get LLM cost this month for a tenant.

        Calculates total cost by summing cost_usd from RunResult
        for all test executions this month.
        """
        try:
            from atp.dashboard.models import (
                RunResult,
                SuiteExecution,
                TestExecution,
            )

            month_start = datetime.now().replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            # Join through TestExecution and SuiteExecution to filter by tenant
            query = (
                select(func.coalesce(func.sum(RunResult.cost_usd), 0.0))
                .join(TestExecution, RunResult.test_execution_id == TestExecution.id)
                .join(
                    SuiteExecution,
                    TestExecution.suite_execution_id == SuiteExecution.id,
                )
                .where(
                    SuiteExecution.tenant_id == tenant_id,
                    SuiteExecution.started_at >= month_start,
                )
            )
            result = await self._session.execute(query)
            return float(result.scalar() or 0.0)
        except Exception as e:
            logger.debug(f"Error getting LLM cost for {tenant_id}: {e}")
            return 0.0

    async def _get_user_count(self, tenant_id: str) -> int:
        """Get the number of users for a tenant."""
        try:
            from atp.dashboard.models import User

            query = select(func.count(User.id)).where(User.tenant_id == tenant_id)
            result = await self._session.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.debug(f"Error getting user count for {tenant_id}: {e}")
            return 0

    async def _get_suite_count(self, tenant_id: str) -> int:
        """Get the number of test suites for a tenant."""
        try:
            from atp.dashboard.models import SuiteDefinition

            query = select(func.count(SuiteDefinition.id)).where(
                SuiteDefinition.tenant_id == tenant_id
            )
            result = await self._session.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.debug(f"Error getting suite count for {tenant_id}: {e}")
            return 0

    async def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        additional: float | int = 0,
    ) -> QuotaCheckResult:
        """Check a specific quota for a tenant.

        Args:
            tenant_id: The tenant ID.
            quota_type: The type of quota to check.
            additional: Additional usage to add (for pre-check).

        Returns:
            QuotaCheckResult with pass/fail and details.
        """
        quotas = await self.get_tenant_quotas(tenant_id)
        if quotas is None:
            # Tenant not found - fail closed
            return QuotaCheckResult(
                tenant_id=tenant_id,
                passed=False,
                violations=[
                    QuotaViolation(
                        quota_type=quota_type,
                        current_value=0,
                        limit_value=0,
                        message=f"Tenant '{tenant_id}' not found",
                    )
                ],
            )

        usage = await self.get_usage(tenant_id)
        current, limit = self._get_quota_values(quota_type, usage, quotas)
        effective_current = current + additional

        violations: list[QuotaViolation] = []
        warnings: list[QuotaViolation] = []

        if effective_current > limit:
            msg = self._get_violation_message(quota_type, effective_current, limit)
            violations.append(
                QuotaViolation(
                    quota_type=quota_type,
                    current_value=effective_current,
                    limit_value=limit,
                    message=msg,
                )
            )
        elif limit > 0 and (effective_current / limit) >= self.WARNING_THRESHOLD:
            pct = (effective_current / limit) * 100
            msg = f"{quota_type.value} at {pct:.0f}% of limit"
            warnings.append(
                QuotaViolation(
                    quota_type=quota_type,
                    current_value=effective_current,
                    limit_value=limit,
                    message=msg,
                )
            )

        return QuotaCheckResult(
            tenant_id=tenant_id,
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    async def check_all_quotas(self, tenant_id: str) -> QuotaCheckResult:
        """Check all quotas for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            QuotaCheckResult with all violations and warnings.
        """
        quotas = await self.get_tenant_quotas(tenant_id)
        if quotas is None:
            return QuotaCheckResult(
                tenant_id=tenant_id,
                passed=False,
                violations=[
                    QuotaViolation(
                        quota_type=QuotaType.TESTS_PER_DAY,
                        current_value=0,
                        limit_value=0,
                        message=f"Tenant '{tenant_id}' not found",
                    )
                ],
            )

        usage = await self.get_usage(tenant_id)
        violations: list[QuotaViolation] = []
        warnings: list[QuotaViolation] = []

        for quota_type in QuotaType:
            current, limit = self._get_quota_values(quota_type, usage, quotas)

            if current > limit:
                msg = self._get_violation_message(quota_type, current, limit)
                violations.append(
                    QuotaViolation(
                        quota_type=quota_type,
                        current_value=current,
                        limit_value=limit,
                        message=msg,
                    )
                )
            elif limit > 0 and (current / limit) >= self.WARNING_THRESHOLD:
                pct = (current / limit) * 100
                msg = f"{quota_type.value} at {pct:.0f}% of limit"
                warnings.append(
                    QuotaViolation(
                        quota_type=quota_type,
                        current_value=current,
                        limit_value=limit,
                        message=msg,
                    )
                )

        return QuotaCheckResult(
            tenant_id=tenant_id,
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def _get_quota_values(
        self,
        quota_type: QuotaType,
        usage: QuotaUsage,
        quotas: TenantQuotas,
    ) -> tuple[float | int, float | int]:
        """Get current and limit values for a quota type."""
        mapping: dict[QuotaType, tuple[Any, Any]] = {
            QuotaType.TESTS_PER_DAY: (usage.tests_today, quotas.max_tests_per_day),
            QuotaType.PARALLEL_RUNS: (usage.parallel_runs, quotas.max_parallel_runs),
            QuotaType.STORAGE_GB: (usage.storage_gb, quotas.max_storage_gb),
            QuotaType.AGENTS: (usage.agents, quotas.max_agents),
            QuotaType.LLM_BUDGET_MONTHLY: (
                usage.llm_cost_this_month,
                quotas.llm_budget_monthly,
            ),
            QuotaType.USERS: (usage.users, quotas.max_users),
            QuotaType.SUITES: (usage.suites, quotas.max_suites),
        }
        return mapping.get(quota_type, (0, 0))

    def _get_violation_message(
        self,
        quota_type: QuotaType,
        current: float | int,
        limit: float | int,
    ) -> str:
        """Get a human-readable violation message."""
        messages = {
            QuotaType.TESTS_PER_DAY: (
                f"Daily test limit exceeded: {current}/{limit} tests today"
            ),
            QuotaType.PARALLEL_RUNS: (
                f"Parallel run limit exceeded: {current}/{limit} concurrent runs"
            ),
            QuotaType.STORAGE_GB: (
                f"Storage limit exceeded: {current:.2f}/{limit:.2f} GB"
            ),
            QuotaType.AGENTS: f"Agent limit exceeded: {current}/{limit} agents",
            QuotaType.LLM_BUDGET_MONTHLY: (
                f"Monthly LLM budget exceeded: ${current:.2f}/${limit:.2f}"
            ),
            QuotaType.USERS: f"User limit exceeded: {current}/{limit} users",
            QuotaType.SUITES: f"Suite limit exceeded: {current}/{limit} suites",
        }
        return messages.get(
            quota_type,
            f"{quota_type.value} exceeded: {current}/{limit}",
        )

    async def enforce_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        additional: float | int = 0,
    ) -> None:
        """Enforce a quota, raising an exception if exceeded.

        Args:
            tenant_id: The tenant ID.
            quota_type: The type of quota to enforce.
            additional: Additional usage to add (for pre-check).

        Raises:
            QuotaExceededError: If the quota is exceeded.
        """
        result = await self.check_quota(tenant_id, quota_type, additional)
        if not result.passed:
            violation = result.violations[0]
            raise QuotaExceededError(
                quota_type=violation.quota_type,
                current_value=violation.current_value,
                limit_value=violation.limit_value,
                message=violation.message,
            )


class QuotaUsageTracker:
    """Tracks quota usage in real-time.

    This class provides methods for incrementing and decrementing
    usage counters, particularly useful for parallel run tracking.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the usage tracker.

        Args:
            session: Database session.
        """
        self._session = session
        self._checker = QuotaChecker(session)

    async def record_test_start(
        self,
        tenant_id: str,
        *,
        enforce: bool = True,
    ) -> QuotaCheckResult:
        """Record the start of a test run.

        Args:
            tenant_id: The tenant ID.
            enforce: Whether to enforce quotas (raises exception if exceeded).

        Returns:
            QuotaCheckResult for the operation.

        Raises:
            QuotaExceededError: If enforce=True and quota exceeded.
        """
        # Check both daily limit and parallel runs
        daily_result = await self._checker.check_quota(
            tenant_id,
            QuotaType.TESTS_PER_DAY,
            additional=1,
        )
        parallel_result = await self._checker.check_quota(
            tenant_id,
            QuotaType.PARALLEL_RUNS,
            additional=1,
        )

        # Combine results
        violations = daily_result.violations + parallel_result.violations
        warnings = daily_result.warnings + parallel_result.warnings

        result = QuotaCheckResult(
            tenant_id=tenant_id,
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

        if enforce and not result.passed:
            violation = result.violations[0]
            raise QuotaExceededError(
                quota_type=violation.quota_type,
                current_value=violation.current_value,
                limit_value=violation.limit_value,
                message=violation.message,
            )

        return result

    async def record_test_end(self, tenant_id: str) -> None:
        """Record the end of a test run.

        This is primarily for updating parallel run counts which are
        derived from current "running" status tests.

        Args:
            tenant_id: The tenant ID.
        """
        # Parallel runs are calculated from running status, not tracked separately
        logger.debug(f"Test ended for tenant {tenant_id}")

    async def record_llm_cost(
        self,
        tenant_id: str,
        cost: float,
        *,
        enforce: bool = True,
    ) -> QuotaCheckResult:
        """Record an LLM cost.

        Args:
            tenant_id: The tenant ID.
            cost: The cost in USD.
            enforce: Whether to enforce quotas.

        Returns:
            QuotaCheckResult for the operation.

        Raises:
            QuotaExceededError: If enforce=True and quota exceeded.
        """
        result = await self._checker.check_quota(
            tenant_id,
            QuotaType.LLM_BUDGET_MONTHLY,
            additional=cost,
        )

        if enforce and not result.passed:
            violation = result.violations[0]
            raise QuotaExceededError(
                quota_type=violation.quota_type,
                current_value=violation.current_value,
                limit_value=violation.limit_value,
                message=violation.message,
            )

        return result

    async def check_can_add_agent(
        self,
        tenant_id: str,
        *,
        enforce: bool = True,
    ) -> QuotaCheckResult:
        """Check if a new agent can be added.

        Args:
            tenant_id: The tenant ID.
            enforce: Whether to enforce quotas.

        Returns:
            QuotaCheckResult for the operation.

        Raises:
            QuotaExceededError: If enforce=True and quota exceeded.
        """
        result = await self._checker.check_quota(
            tenant_id,
            QuotaType.AGENTS,
            additional=1,
        )

        if enforce and not result.passed:
            violation = result.violations[0]
            raise QuotaExceededError(
                quota_type=violation.quota_type,
                current_value=violation.current_value,
                limit_value=violation.limit_value,
                message=violation.message,
            )

        return result

    async def check_can_add_user(
        self,
        tenant_id: str,
        *,
        enforce: bool = True,
    ) -> QuotaCheckResult:
        """Check if a new user can be added.

        Args:
            tenant_id: The tenant ID.
            enforce: Whether to enforce quotas.

        Returns:
            QuotaCheckResult for the operation.

        Raises:
            QuotaExceededError: If enforce=True and quota exceeded.
        """
        result = await self._checker.check_quota(
            tenant_id,
            QuotaType.USERS,
            additional=1,
        )

        if enforce and not result.passed:
            violation = result.violations[0]
            raise QuotaExceededError(
                quota_type=violation.quota_type,
                current_value=violation.current_value,
                limit_value=violation.limit_value,
                message=violation.message,
            )

        return result

    async def check_can_add_suite(
        self,
        tenant_id: str,
        *,
        enforce: bool = True,
    ) -> QuotaCheckResult:
        """Check if a new test suite can be added.

        Args:
            tenant_id: The tenant ID.
            enforce: Whether to enforce quotas.

        Returns:
            QuotaCheckResult for the operation.

        Raises:
            QuotaExceededError: If enforce=True and quota exceeded.
        """
        result = await self._checker.check_quota(
            tenant_id,
            QuotaType.SUITES,
            additional=1,
        )

        if enforce and not result.passed:
            violation = result.violations[0]
            raise QuotaExceededError(
                quota_type=violation.quota_type,
                current_value=violation.current_value,
                limit_value=violation.limit_value,
                message=violation.message,
            )

        return result

    async def get_quota_status(self, tenant_id: str) -> dict[str, Any]:
        """Get a summary of quota status for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Dictionary with quota status information.
        """
        quotas = await self._checker.get_tenant_quotas(tenant_id)
        if quotas is None:
            return {"error": f"Tenant '{tenant_id}' not found"}

        usage = await self._checker.get_usage(tenant_id)
        result = await self._checker.check_all_quotas(tenant_id)

        return {
            "tenant_id": tenant_id,
            "usage": usage.model_dump(),
            "quotas": quotas.model_dump(),
            "check_result": result.model_dump(),
            "any_exceeded": not result.passed,
            "any_warning": len(result.warnings) > 0,
        }
