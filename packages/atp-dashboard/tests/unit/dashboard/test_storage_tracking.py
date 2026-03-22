"""Tests for real storage tracking in tenant quota enforcement."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.dashboard.tenancy.models import Tenant, TenantQuotas
from atp.dashboard.tenancy.quotas import (
    STORAGE_CACHE_TTL_SECONDS,
    QuotaChecker,
    _storage_cache,
)


@pytest.fixture(autouse=True)
def _clear_storage_cache() -> None:
    """Clear the storage cache before each test."""
    _storage_cache.clear()


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    return AsyncMock()


@pytest.fixture
def checker(mock_session: AsyncMock) -> QuotaChecker:
    """Create a QuotaChecker with mock session."""
    return QuotaChecker(mock_session)


@pytest.fixture
def sample_tenant() -> Tenant:
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


class TestGetStorageUsage:
    """Tests for _get_storage_usage method."""

    @pytest.mark.anyio
    async def test_returns_sum_of_all_sources(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test that storage usage sums artifacts, run data, and traces."""
        # Mock artifact query returns 1 GB in bytes
        artifact_bytes = 1 * 1024**3
        # Mock run result query returns 0.5 GB in bytes
        run_result_bytes = 512 * 1024**2
        # Total expected: 1.5 GB (traces will be 0 — no dir)

        mock_results = [
            MagicMock(scalar=MagicMock(return_value=artifact_bytes)),
            MagicMock(scalar=MagicMock(return_value=run_result_bytes)),
        ]
        mock_session.execute.side_effect = mock_results

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            result = await checker._get_storage_usage("test-tenant")

        expected_gb = (artifact_bytes + run_result_bytes) / (1024**3)
        assert abs(result - expected_gb) < 0.001

    @pytest.mark.anyio
    async def test_returns_zero_when_no_data(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test that storage returns 0 when there is no data."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            result = await checker._get_storage_usage("test-tenant")

        assert result == 0.0

    @pytest.mark.anyio
    async def test_caches_result(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test that results are cached and reused."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1024
        mock_session.execute.return_value = mock_result

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            result1 = await checker._get_storage_usage("test-tenant")
            result2 = await checker._get_storage_usage("test-tenant")

        assert result1 == result2
        # DB should only be called twice (artifact + run result)
        # for the first call; second call uses cache
        assert mock_session.execute.call_count == 2

    @pytest.mark.anyio
    async def test_cache_expires(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test that cache expires after TTL."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1024
        mock_session.execute.return_value = mock_result

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            await checker._get_storage_usage("test-tenant")

            # Simulate cache expiry by manipulating the cache
            tenant_cache = _storage_cache["test-tenant"]
            expired_time = tenant_cache[0] - STORAGE_CACHE_TTL_SECONDS - 1
            _storage_cache["test-tenant"] = (
                expired_time,
                tenant_cache[1],
            )

            await checker._get_storage_usage("test-tenant")

        # DB should be called 4 times total (2 per uncached call)
        assert mock_session.execute.call_count == 4

    @pytest.mark.anyio
    async def test_different_tenants_cached_separately(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test that each tenant has its own cache entry."""
        call_count = 0
        values = [500, 1000]

        def make_mock_result() -> MagicMock:
            nonlocal call_count
            idx = min(call_count // 2, 1)
            mock = MagicMock()
            mock.scalar.return_value = values[idx]
            call_count += 1
            return mock

        mock_session.execute.side_effect = lambda _: make_mock_result()

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            await checker._get_storage_usage("tenant-a")
            await checker._get_storage_usage("tenant-b")

        assert "tenant-a" in _storage_cache
        assert "tenant-b" in _storage_cache

    @pytest.mark.anyio
    async def test_handles_db_errors_gracefully(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test that DB errors return 0 for each component."""
        mock_session.execute.side_effect = Exception("DB error")

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            result = await checker._get_storage_usage("test-tenant")

        assert result == 0.0


class TestGetArtifactStorageBytes:
    """Tests for _get_artifact_storage_bytes method."""

    @pytest.mark.anyio
    async def test_returns_artifact_sum(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test returning sum of artifact sizes."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5000000
        mock_session.execute.return_value = mock_result

        result = await checker._get_artifact_storage_bytes("test-tenant")
        assert result == 5000000

    @pytest.mark.anyio
    async def test_returns_zero_on_no_artifacts(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test returning 0 when no artifacts exist."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        result = await checker._get_artifact_storage_bytes("test-tenant")
        assert result == 0

    @pytest.mark.anyio
    async def test_returns_zero_on_error(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test returning 0 on database error."""
        mock_session.execute.side_effect = Exception("DB error")

        result = await checker._get_artifact_storage_bytes("test-tenant")
        assert result == 0


class TestGetRunResultStorageBytes:
    """Tests for _get_run_result_storage_bytes method."""

    @pytest.mark.anyio
    async def test_returns_json_length_sum(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test returning sum of JSON field lengths."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 2000000
        mock_session.execute.return_value = mock_result

        result = await checker._get_run_result_storage_bytes("test-tenant")
        assert result == 2000000

    @pytest.mark.anyio
    async def test_returns_zero_on_no_results(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test returning 0 when no run results exist."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        result = await checker._get_run_result_storage_bytes("test-tenant")
        assert result == 0

    @pytest.mark.anyio
    async def test_returns_zero_on_error(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
    ) -> None:
        """Test returning 0 on database error."""
        mock_session.execute.side_effect = Exception("DB error")

        result = await checker._get_run_result_storage_bytes("test-tenant")
        assert result == 0


class TestGetTraceStorageBytes:
    """Tests for _get_trace_storage_bytes static method."""

    def test_returns_zero_when_dir_not_exists(self, tmp_path: Path) -> None:
        """Test returning 0 when tenant trace dir doesn't exist."""
        with patch(
            "atp.tracing.storage.DEFAULT_TRACES_DIR",
            tmp_path / "traces",
        ):
            result = QuotaChecker._get_trace_storage_bytes("nonexistent-tenant")
        assert result == 0

    def test_returns_sum_of_file_sizes(self, tmp_path: Path) -> None:
        """Test returning sum of file sizes in tenant dir."""
        tenant_dir = tmp_path / "traces" / "test-tenant"
        tenant_dir.mkdir(parents=True)

        # Create test files with known sizes
        (tenant_dir / "trace1.json").write_text("a" * 1000)
        (tenant_dir / "trace2.json").write_text("b" * 2000)

        with patch(
            "atp.tracing.storage.DEFAULT_TRACES_DIR",
            tmp_path / "traces",
        ):
            result = QuotaChecker._get_trace_storage_bytes("test-tenant")

        assert result == 3000

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        """Test that subdirectories are not counted."""
        tenant_dir = tmp_path / "traces" / "test-tenant"
        tenant_dir.mkdir(parents=True)

        (tenant_dir / "trace1.json").write_text("a" * 500)
        sub = tenant_dir / "subdir"
        sub.mkdir()
        (sub / "nested.json").write_text("c" * 9999)

        with patch(
            "atp.tracing.storage.DEFAULT_TRACES_DIR",
            tmp_path / "traces",
        ):
            result = QuotaChecker._get_trace_storage_bytes("test-tenant")

        assert result == 500

    def test_returns_zero_on_error(self) -> None:
        """Test returning 0 when an exception occurs."""
        with patch(
            "atp.tracing.storage.DEFAULT_TRACES_DIR",
            side_effect=Exception("fail"),
        ):
            # If the import itself fails, it will be caught
            result = QuotaChecker._get_trace_storage_bytes("test-tenant")
        assert result == 0


class TestStorageInQuotaIntegration:
    """Integration tests for storage tracking in quota checks."""

    @pytest.mark.anyio
    async def test_storage_quota_exceeded(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test that storage quota violation is detected."""
        mock_session.get.return_value = sample_tenant

        # Return large sizes for artifact and run result queries
        # Other queries return 0
        ten_gb_bytes = 11 * 1024**3  # 11 GB > 10 GB limit

        call_count = 0

        def mock_execute(query: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            m = MagicMock()
            # The artifact query is the 3rd call
            # (after tests_today, parallel_runs)
            if call_count == 3:
                m.scalar.return_value = ten_gb_bytes
            else:
                m.scalar.return_value = 0
            return m

        mock_session.execute.side_effect = mock_execute

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            result = await checker.check_all_quotas("test-tenant")

        # Should have a storage violation
        storage_violations = [
            v for v in result.violations if v.quota_type.value == "storage_gb"
        ]
        assert len(storage_violations) == 1

    @pytest.mark.anyio
    async def test_storage_quota_passes(
        self,
        checker: QuotaChecker,
        mock_session: AsyncMock,
        sample_tenant: Tenant,
    ) -> None:
        """Test that storage quota passes when under limit."""
        mock_session.get.return_value = sample_tenant

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result

        with patch.object(
            QuotaChecker,
            "_get_trace_storage_bytes",
            return_value=0,
        ):
            result = await checker.check_all_quotas("test-tenant")

        storage_violations = [
            v for v in result.violations if v.quota_type.value == "storage_gb"
        ]
        assert len(storage_violations) == 0
