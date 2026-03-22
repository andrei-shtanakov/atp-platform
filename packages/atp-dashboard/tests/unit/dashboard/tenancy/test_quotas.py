"""Tests for tenant quota enforcement."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tenancy.models import Tenant, TenantQuotas
from atp.dashboard.tenancy.quotas import (
    QuotaChecker,
    QuotaCheckResult,
    QuotaExceededError,
    QuotaType,
    QuotaUsage,
    QuotaUsageTracker,
    QuotaViolation,
)


class TestQuotaType:
    """Tests for QuotaType enum."""

    def test_all_quota_types_defined(self) -> None:
        """Test that all expected quota types are defined."""
        assert QuotaType.TESTS_PER_DAY.value == "tests_per_day"
        assert QuotaType.PARALLEL_RUNS.value == "parallel_runs"
        assert QuotaType.STORAGE_GB.value == "storage_gb"
        assert QuotaType.AGENTS.value == "agents"
        assert QuotaType.LLM_BUDGET_MONTHLY.value == "llm_budget_monthly"
        assert QuotaType.USERS.value == "users"
        assert QuotaType.SUITES.value == "suites"

    def test_quota_type_count(self) -> None:
        """Test that we have the expected number of quota types."""
        assert len(QuotaType) == 7


class TestQuotaViolation:
    """Tests for QuotaViolation model."""

    def test_create_violation(self) -> None:
        """Test creating a quota violation."""
        violation = QuotaViolation(
            quota_type=QuotaType.TESTS_PER_DAY,
            current_value=150,
            limit_value=100,
            message="Daily test limit exceeded",
        )
        assert violation.quota_type == QuotaType.TESTS_PER_DAY
        assert violation.current_value == 150
        assert violation.limit_value == 100
        assert "exceeded" in violation.message.lower()

    def test_violation_serialization(self) -> None:
        """Test that violations can be serialized."""
        violation = QuotaViolation(
            quota_type=QuotaType.AGENTS,
            current_value=15,
            limit_value=10,
            message="Agent limit exceeded",
        )
        data = violation.model_dump()
        assert data["quota_type"] == "agents"
        assert data["current_value"] == 15
        assert data["limit_value"] == 10


class TestQuotaCheckResult:
    """Tests for QuotaCheckResult model."""

    def test_passed_result(self) -> None:
        """Test creating a passed check result."""
        result = QuotaCheckResult(
            tenant_id="test-tenant",
            passed=True,
            violations=[],
            warnings=[],
        )
        assert result.passed is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0

    def test_failed_result_with_violations(self) -> None:
        """Test creating a failed check result."""
        violation = QuotaViolation(
            quota_type=QuotaType.TESTS_PER_DAY,
            current_value=150,
            limit_value=100,
            message="Limit exceeded",
        )
        result = QuotaCheckResult(
            tenant_id="test-tenant",
            passed=False,
            violations=[violation],
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].quota_type == QuotaType.TESTS_PER_DAY

    def test_result_with_warnings(self) -> None:
        """Test check result with warnings."""
        warning = QuotaViolation(
            quota_type=QuotaType.STORAGE_GB,
            current_value=8.5,
            limit_value=10.0,
            message="Storage at 85% of limit",
        )
        result = QuotaCheckResult(
            tenant_id="test-tenant",
            passed=True,
            violations=[],
            warnings=[warning],
        )
        assert result.passed is True
        assert len(result.warnings) == 1


class TestQuotaUsage:
    """Tests for QuotaUsage model."""

    def test_default_values(self) -> None:
        """Test default usage values."""
        usage = QuotaUsage(tenant_id="test")
        assert usage.tests_today == 0
        assert usage.parallel_runs == 0
        assert usage.storage_gb == 0.0
        assert usage.agents == 0
        assert usage.llm_cost_this_month == 0.0
        assert usage.users == 0
        assert usage.suites == 0

    def test_custom_values(self) -> None:
        """Test custom usage values."""
        usage = QuotaUsage(
            tenant_id="test",
            tests_today=50,
            parallel_runs=3,
            storage_gb=5.5,
            agents=8,
            llm_cost_this_month=75.50,
            users=15,
            suites=25,
        )
        assert usage.tests_today == 50
        assert usage.parallel_runs == 3
        assert usage.storage_gb == 5.5
        assert usage.llm_cost_this_month == 75.50

    def test_timestamp_default(self) -> None:
        """Test that timestamp is set by default."""
        usage = QuotaUsage(tenant_id="test")
        assert usage.timestamp is not None
        assert isinstance(usage.timestamp, datetime)


class TestQuotaExceededError:
    """Tests for QuotaExceededError exception."""

    def test_create_error(self) -> None:
        """Test creating a quota exceeded error."""
        error = QuotaExceededError(
            quota_type=QuotaType.TESTS_PER_DAY,
            current_value=150,
            limit_value=100,
        )
        assert error.quota_type == QuotaType.TESTS_PER_DAY
        assert error.current_value == 150
        assert error.limit_value == 100
        assert "tests_per_day" in str(error)
        assert "150" in str(error)
        assert "100" in str(error)

    def test_error_with_custom_message(self) -> None:
        """Test creating error with custom message."""
        error = QuotaExceededError(
            quota_type=QuotaType.AGENTS,
            current_value=15,
            limit_value=10,
            message="Custom message",
        )
        assert error.message == "Custom message"
        assert str(error) == "Custom message"


class TestQuotaChecker:
    """Tests for QuotaChecker class."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create a mock async session."""
        return AsyncMock()

    @pytest.fixture
    def checker(self, mock_session: AsyncMock) -> QuotaChecker:
        """Create a QuotaChecker with mock session."""
        return QuotaChecker(mock_session)

    @pytest.fixture
    def sample_tenant(self) -> Tenant:
        """Create a sample tenant."""
        return Tenant(
            id="test-tenant",
            name="Test Tenant",
            plan="pro",
            schema_name="tenant_test",
            quotas_json=TenantQuotas(
                max_tests_per_day=100,
                max_parallel_runs=5,
                max_storage_gb=10.0,
                max_agents=10,
                llm_budget_monthly=100.0,
                max_users=10,
                max_suites=50,
            ).model_dump(),
            settings_json={},
        )

    @pytest.mark.anyio
    async def test_get_tenant_quotas_found(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test getting quotas for existing tenant."""
        mock_session.get.return_value = sample_tenant
        quotas = await checker.get_tenant_quotas("test-tenant")
        assert quotas is not None
        assert quotas.max_tests_per_day == 100

    @pytest.mark.anyio
    async def test_get_tenant_quotas_not_found(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test getting quotas for non-existent tenant."""
        mock_session.get.return_value = None
        quotas = await checker.get_tenant_quotas("nonexistent")
        assert quotas is None

    @pytest.mark.anyio
    async def test_get_usage(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test getting usage statistics."""
        # Mock all the query results
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result

        usage = await checker.get_usage("test-tenant")
        assert usage.tenant_id == "test-tenant"
        assert usage.tests_today >= 0

    @pytest.mark.anyio
    async def test_check_quota_passed(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test quota check that passes."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50  # Under limit
        mock_session.execute.return_value = mock_result

        result = await checker.check_quota(
            "test-tenant",
            QuotaType.TESTS_PER_DAY,
        )
        assert result.passed is True
        assert len(result.violations) == 0

    @pytest.mark.anyio
    async def test_check_quota_exceeded(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test quota check that fails."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 150  # Over limit of 100
        mock_session.execute.return_value = mock_result

        result = await checker.check_quota(
            "test-tenant",
            QuotaType.TESTS_PER_DAY,
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].quota_type == QuotaType.TESTS_PER_DAY

    @pytest.mark.anyio
    async def test_check_quota_with_additional(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test quota check with additional usage."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 99  # Just under limit
        mock_session.execute.return_value = mock_result

        # Without additional, should pass
        result = await checker.check_quota(
            "test-tenant",
            QuotaType.TESTS_PER_DAY,
            additional=0,
        )
        assert result.passed is True

        # With additional=2, should fail (99+2=101 > 100)
        result = await checker.check_quota(
            "test-tenant",
            QuotaType.TESTS_PER_DAY,
            additional=2,
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_check_quota_warning_threshold(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test quota check warning at 80% threshold."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 85  # 85% of 100
        mock_session.execute.return_value = mock_result

        result = await checker.check_quota(
            "test-tenant",
            QuotaType.TESTS_PER_DAY,
        )
        assert result.passed is True  # Not exceeded
        assert len(result.warnings) == 1  # But warning issued

    @pytest.mark.anyio
    async def test_check_quota_tenant_not_found(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test quota check for non-existent tenant."""
        mock_session.get.return_value = None

        result = await checker.check_quota(
            "nonexistent",
            QuotaType.TESTS_PER_DAY,
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert "not found" in result.violations[0].message.lower()

    @pytest.mark.anyio
    async def test_check_all_quotas(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test checking all quotas at once."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0  # All under limit
        mock_session.execute.return_value = mock_result

        result = await checker.check_all_quotas("test-tenant")
        assert result.passed is True
        assert len(result.violations) == 0

    @pytest.mark.anyio
    async def test_enforce_quota_passes(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test enforce_quota when quota is not exceeded."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50
        mock_session.execute.return_value = mock_result

        # Should not raise
        await checker.enforce_quota("test-tenant", QuotaType.TESTS_PER_DAY)

    @pytest.mark.anyio
    async def test_enforce_quota_raises(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test enforce_quota raises when quota exceeded."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 150
        mock_session.execute.return_value = mock_result

        with pytest.raises(QuotaExceededError) as exc_info:
            await checker.enforce_quota("test-tenant", QuotaType.TESTS_PER_DAY)

        assert exc_info.value.quota_type == QuotaType.TESTS_PER_DAY

    def test_get_violation_message_tests(self, checker: QuotaChecker) -> None:
        """Test violation message for tests quota."""
        msg = checker._get_violation_message(QuotaType.TESTS_PER_DAY, 150, 100)
        assert "150" in msg
        assert "100" in msg
        assert "test" in msg.lower()

    def test_get_violation_message_storage(self, checker: QuotaChecker) -> None:
        """Test violation message for storage quota."""
        msg = checker._get_violation_message(QuotaType.STORAGE_GB, 15.5, 10.0)
        assert "15.5" in msg
        assert "10.0" in msg
        assert "storage" in msg.lower()

    def test_get_violation_message_budget(self, checker: QuotaChecker) -> None:
        """Test violation message for budget quota."""
        msg = checker._get_violation_message(QuotaType.LLM_BUDGET_MONTHLY, 150.0, 100.0)
        assert "$150" in msg
        assert "$100" in msg
        assert "budget" in msg.lower()


class TestQuotaUsageTracker:
    """Tests for QuotaUsageTracker class."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create a mock async session."""
        return AsyncMock()

    @pytest.fixture
    def tracker(self, mock_session: AsyncMock) -> QuotaUsageTracker:
        """Create a QuotaUsageTracker with mock session."""
        return QuotaUsageTracker(mock_session)

    @pytest.fixture
    def sample_tenant(self) -> Tenant:
        """Create a sample tenant with high quotas."""
        return Tenant(
            id="test-tenant",
            name="Test Tenant",
            plan="pro",
            schema_name="tenant_test",
            quotas_json=TenantQuotas(
                max_tests_per_day=100,
                max_parallel_runs=100,  # High limit for testing
                max_agents=100,
                max_users=100,
                max_suites=100,
                llm_budget_monthly=1000.0,
            ).model_dump(),
            settings_json={},
        )

    @pytest.mark.anyio
    async def test_record_test_start_allowed(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test recording test start when under quota."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 10  # Under all limits
        mock_session.execute.return_value = mock_result

        result = await tracker.record_test_start("test-tenant")
        assert result.passed is True

    @pytest.mark.anyio
    async def test_record_test_start_denied(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test recording test start when over quota."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100  # At limit
        mock_session.execute.return_value = mock_result

        with pytest.raises(QuotaExceededError):
            await tracker.record_test_start("test-tenant", enforce=True)

    @pytest.mark.anyio
    async def test_record_test_start_no_enforce(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test recording test start without enforcement."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result

        # Should not raise when enforce=False
        result = await tracker.record_test_start("test-tenant", enforce=False)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_record_llm_cost(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test recording LLM cost."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50.0
        mock_session.execute.return_value = mock_result

        result = await tracker.record_llm_cost("test-tenant", 10.0)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_check_can_add_agent(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test checking if agent can be added."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        result = await tracker.check_can_add_agent("test-tenant")
        assert result.passed is True

    @pytest.mark.anyio
    async def test_check_can_add_user(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test checking if user can be added."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        result = await tracker.check_can_add_user("test-tenant")
        assert result.passed is True

    @pytest.mark.anyio
    async def test_check_can_add_suite(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test checking if suite can be added."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 25
        mock_session.execute.return_value = mock_result

        result = await tracker.check_can_add_suite("test-tenant")
        assert result.passed is True

    @pytest.mark.anyio
    async def test_get_quota_status(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test getting quota status summary."""
        mock_session.get.return_value = sample_tenant
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50
        mock_session.execute.return_value = mock_result

        status = await tracker.get_quota_status("test-tenant")
        assert status["tenant_id"] == "test-tenant"
        assert "usage" in status
        assert "quotas" in status
        assert "check_result" in status

    @pytest.mark.anyio
    async def test_get_quota_status_tenant_not_found(
        self,
        tracker: QuotaUsageTracker,
        mock_session: AsyncMock,
    ) -> None:
        """Test quota status for non-existent tenant."""
        mock_session.get.return_value = None

        status = await tracker.get_quota_status("nonexistent")
        assert "error" in status
