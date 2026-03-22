"""Unit tests for audit logging functionality."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.audit import (
    AuditAction,
    AuditCategory,
    AuditLog,
    AuditLogFilter,
    AuditLogResponse,
    AuditSeverity,
    RetentionPolicy,
    audit_log,
    compute_entry_hash,
    create_audit_context,
)
from atp.dashboard.models import DEFAULT_TENANT_ID, User


class TestComputeEntryHash:
    """Tests for compute_entry_hash function."""

    def test_hash_is_deterministic(self) -> None:
        """Hash should be the same for the same inputs."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        hash1 = compute_entry_hash(
            timestamp=timestamp,
            category="authentication",
            action="login_success",
            user_id=1,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash=None,
        )
        hash2 = compute_entry_hash(
            timestamp=timestamp,
            category="authentication",
            action="login_success",
            user_id=1,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash=None,
        )
        assert hash1 == hash2

    def test_hash_changes_with_different_inputs(self) -> None:
        """Hash should differ for different inputs."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        hash1 = compute_entry_hash(
            timestamp=timestamp,
            category="authentication",
            action="login_success",
            user_id=1,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash=None,
        )
        hash2 = compute_entry_hash(
            timestamp=timestamp,
            category="authentication",
            action="login_failure",  # Different action
            user_id=1,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash=None,
        )
        assert hash1 != hash2

    def test_hash_includes_previous_hash(self) -> None:
        """Hash should change with different previous hash."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        hash1 = compute_entry_hash(
            timestamp=timestamp,
            category="authentication",
            action="login_success",
            user_id=1,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash=None,
        )
        hash2 = compute_entry_hash(
            timestamp=timestamp,
            category="authentication",
            action="login_success",
            user_id=1,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash="abc123",
        )
        assert hash1 != hash2

    def test_hash_is_64_chars(self) -> None:
        """SHA-256 hash should be 64 hex characters."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        hash_val = compute_entry_hash(
            timestamp=timestamp,
            category="test",
            action="test",
            user_id=None,
            resource_type=None,
            resource_id=None,
            details=None,
            previous_hash=None,
        )
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestAuditLog:
    """Tests for audit_log function."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        # Mock execute to return empty result for previous hash query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)
        return session

    @pytest.fixture
    def test_user(self) -> User:
        """Create a test user."""
        user = User(
            id=1,
            tenant_id=DEFAULT_TENANT_ID,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )
        return user

    @pytest.mark.anyio
    async def test_creates_audit_entry(self, mock_session: AsyncMock) -> None:
        """Should create an audit log entry."""
        entry = await audit_log(
            mock_session,
            AuditCategory.AUTHENTICATION,
            AuditAction.LOGIN_SUCCESS,
        )

        assert entry is not None
        assert entry.category == AuditCategory.AUTHENTICATION.value
        assert entry.action == AuditAction.LOGIN_SUCCESS.value
        assert entry.severity == AuditSeverity.INFO.value
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_extracts_user_info(
        self, mock_session: AsyncMock, test_user: User
    ) -> None:
        """Should extract user info from user object."""
        entry = await audit_log(
            mock_session,
            AuditCategory.AUTHENTICATION,
            AuditAction.LOGIN_SUCCESS,
            user=test_user,
        )

        assert entry.user_id == test_user.id
        assert entry.username == test_user.username

    @pytest.mark.anyio
    async def test_uses_explicit_user_info(self, mock_session: AsyncMock) -> None:
        """Should use explicit user_id and username."""
        entry = await audit_log(
            mock_session,
            AuditCategory.DATA_ACCESS,
            AuditAction.DATA_READ,
            user_id=42,
            username="explicit_user",
        )

        assert entry.user_id == 42
        assert entry.username == "explicit_user"

    @pytest.mark.anyio
    async def test_serializes_details(self, mock_session: AsyncMock) -> None:
        """Should serialize details dict to JSON."""
        details = {"key": "value", "number": 123}
        entry = await audit_log(
            mock_session,
            AuditCategory.DATA_MODIFICATION,
            AuditAction.DATA_CREATE,
            details=details,
        )

        assert entry.details is not None
        parsed = json.loads(entry.details)
        assert parsed == details

    @pytest.mark.anyio
    async def test_converts_resource_id_to_string(
        self, mock_session: AsyncMock
    ) -> None:
        """Should convert numeric resource_id to string."""
        entry = await audit_log(
            mock_session,
            AuditCategory.DATA_MODIFICATION,
            AuditAction.DATA_UPDATE,
            resource_type="agent",
            resource_id=123,
        )

        assert entry.resource_id == "123"

    @pytest.mark.anyio
    async def test_computes_entry_hash(self, mock_session: AsyncMock) -> None:
        """Should compute hash for the entry."""
        entry = await audit_log(
            mock_session,
            AuditCategory.AUTHENTICATION,
            AuditAction.LOGIN_SUCCESS,
        )

        assert entry.entry_hash is not None
        assert len(entry.entry_hash) == 64

    @pytest.mark.anyio
    async def test_includes_previous_hash(self, mock_session: AsyncMock) -> None:
        """Should include previous hash in chain."""
        # Mock to return a previous hash
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "previous_hash_value"
        mock_session.execute = AsyncMock(return_value=mock_result)

        entry = await audit_log(
            mock_session,
            AuditCategory.AUTHENTICATION,
            AuditAction.LOGIN_SUCCESS,
        )

        assert entry.previous_hash == "previous_hash_value"


class TestRetentionPolicy:
    """Tests for RetentionPolicy model."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        policy = RetentionPolicy()

        assert policy.default_retention_days == 90
        assert policy.min_entries_to_keep == 1000
        assert policy.max_entries == 1000000
        assert AuditCategory.AUTHENTICATION.value in policy.category_retention
        assert AuditCategory.SECURITY.value in policy.category_retention

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        policy = RetentionPolicy(
            default_retention_days=30,
            min_entries_to_keep=500,
            max_entries=50000,
        )

        assert policy.default_retention_days == 30
        assert policy.min_entries_to_keep == 500
        assert policy.max_entries == 50000

    def test_is_frozen(self) -> None:
        """Should be immutable."""
        policy = RetentionPolicy()
        with pytest.raises(Exception):  # ValidationError or AttributeError
            policy.default_retention_days = 30  # type: ignore[misc]


class TestAuditLogResponse:
    """Tests for AuditLogResponse schema."""

    def test_from_model_parses_details(self) -> None:
        """Should parse JSON details from model."""
        model = AuditLog(
            id=1,
            tenant_id=DEFAULT_TENANT_ID,
            timestamp=datetime.utcnow(),
            category="authentication",
            action="login_success",
            severity="info",
            entry_hash="abc123",
            details='{"key": "value"}',
        )

        response = AuditLogResponse.from_model(model)

        assert response.details == {"key": "value"}

    def test_from_model_handles_invalid_json(self) -> None:
        """Should handle invalid JSON in details."""
        model = AuditLog(
            id=1,
            tenant_id=DEFAULT_TENANT_ID,
            timestamp=datetime.utcnow(),
            category="authentication",
            action="login_success",
            severity="info",
            entry_hash="abc123",
            details="not valid json",
        )

        response = AuditLogResponse.from_model(model)

        assert response.details == {"raw": "not valid json"}

    def test_from_model_handles_null_details(self) -> None:
        """Should handle null details."""
        model = AuditLog(
            id=1,
            tenant_id=DEFAULT_TENANT_ID,
            timestamp=datetime.utcnow(),
            category="authentication",
            action="login_success",
            severity="info",
            entry_hash="abc123",
            details=None,
        )

        response = AuditLogResponse.from_model(model)

        assert response.details is None


class TestAuditLogFilter:
    """Tests for AuditLogFilter model."""

    def test_empty_filter(self) -> None:
        """Should allow empty filter."""
        filter_obj = AuditLogFilter()

        assert filter_obj.category is None
        assert filter_obj.action is None
        assert filter_obj.severity is None
        assert filter_obj.user_id is None

    def test_with_category(self) -> None:
        """Should accept category filter."""
        filter_obj = AuditLogFilter(category=AuditCategory.AUTHENTICATION)

        assert filter_obj.category == AuditCategory.AUTHENTICATION

    def test_with_date_range(self) -> None:
        """Should accept date range filter."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        filter_obj = AuditLogFilter(start_date=start, end_date=end)

        assert filter_obj.start_date == start
        assert filter_obj.end_date == end


class TestCreateAuditContext:
    """Tests for create_audit_context helper."""

    def test_with_user(self) -> None:
        """Should extract user info."""
        user = User(
            id=1,
            tenant_id=DEFAULT_TENANT_ID,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )

        context = create_audit_context(
            user=user,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_id="req-123",
        )

        assert context["user"] == user
        assert context["user_id"] == 1
        assert context["username"] == "testuser"
        assert context["ip_address"] == "192.168.1.1"
        assert context["user_agent"] == "Mozilla/5.0"
        assert context["request_id"] == "req-123"

    def test_without_user(self) -> None:
        """Should handle missing user."""
        context = create_audit_context(
            ip_address="192.168.1.1",
        )

        assert context["user"] is None
        assert context["user_id"] is None
        assert context["username"] is None
        assert context["ip_address"] == "192.168.1.1"


class TestAuditEnums:
    """Tests for audit enums."""

    def test_audit_category_values(self) -> None:
        """Should have expected categories."""
        categories = [c.value for c in AuditCategory]

        assert "authentication" in categories
        assert "authorization" in categories
        assert "data_access" in categories
        assert "data_modification" in categories
        assert "configuration" in categories
        assert "admin_action" in categories
        assert "security" in categories

    def test_audit_severity_values(self) -> None:
        """Should have expected severities."""
        severities = [s.value for s in AuditSeverity]

        assert "debug" in severities
        assert "info" in severities
        assert "warning" in severities
        assert "error" in severities
        assert "critical" in severities

    def test_audit_action_auth_values(self) -> None:
        """Should have authentication actions."""
        actions = [a.value for a in AuditAction]

        assert "login_success" in actions
        assert "login_failure" in actions
        assert "logout" in actions
        assert "password_change" in actions

    def test_audit_action_data_values(self) -> None:
        """Should have data modification actions."""
        actions = [a.value for a in AuditAction]

        assert "data_create" in actions
        assert "data_update" in actions
        assert "data_delete" in actions
        assert "data_read" in actions
        assert "data_export" in actions
