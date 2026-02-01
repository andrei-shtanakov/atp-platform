"""Integration tests for audit routes."""

import json
import uuid
from datetime import datetime, timedelta

import pytest

from atp.dashboard.audit import (
    AuditAction,
    AuditCategory,
    AuditLog,
    AuditSeverity,
    audit_log,
)
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import DEFAULT_TENANT_ID, User


@pytest.fixture
async def test_db():
    """Create isolated test database for each test."""
    # Create unique in-memory database
    db = Database("sqlite+aiosqlite:///:memory:")
    set_database(db)
    await db.create_tables()
    yield db


@pytest.fixture
async def db_session(test_db):
    """Get database session."""
    async with test_db.session() as session:
        yield session


@pytest.fixture
async def admin_user(test_db) -> User:
    """Create admin user for testing with unique credentials."""
    async with test_db.session() as session:
        unique_id = str(uuid.uuid4())[:8]
        user = User(
            tenant_id=DEFAULT_TENANT_ID,
            username=f"admin_{unique_id}",
            email=f"admin_{unique_id}@example.com",
            hashed_password="hashed",
            is_admin=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest.fixture
async def sample_audit_logs(test_db, admin_user) -> list[AuditLog]:
    """Create sample audit logs for testing."""
    async with test_db.session() as session:
        logs = []

        # Create logs with different categories
        categories = [
            (AuditCategory.AUTHENTICATION, AuditAction.LOGIN_SUCCESS),
            (AuditCategory.AUTHENTICATION, AuditAction.LOGOUT),
            (AuditCategory.DATA_MODIFICATION, AuditAction.DATA_CREATE),
            (AuditCategory.DATA_ACCESS, AuditAction.DATA_READ),
            (AuditCategory.CONFIGURATION, AuditAction.CONFIG_UPDATE),
        ]

        for i, (category, action) in enumerate(categories):
            entry = await audit_log(
                session,
                category,
                action,
                user=admin_user,
                ip_address=f"192.168.1.{i}",
                resource_type="agent",
                resource_id=str(i),
                resource_name=f"test-agent-{i}",
                details={"test_index": i},
            )
            logs.append(entry)

        await session.commit()
        return logs


class TestAuditLogModel:
    """Tests for AuditLog model operations."""

    @pytest.mark.anyio
    async def test_create_audit_log(self, test_db) -> None:
        """Should create audit log entry."""
        async with test_db.session() as session:
            entry = await audit_log(
                session,
                AuditCategory.AUTHENTICATION,
                AuditAction.LOGIN_SUCCESS,
                ip_address="127.0.0.1",
                user_agent="Test Client",
            )
            await session.commit()

            assert entry.id is not None
            assert entry.category == AuditCategory.AUTHENTICATION.value
            assert entry.action == AuditAction.LOGIN_SUCCESS.value
            assert entry.entry_hash is not None

    @pytest.mark.anyio
    async def test_audit_log_chain(self, test_db) -> None:
        """Should maintain hash chain across entries."""
        async with test_db.session() as session:
            entry1 = await audit_log(
                session,
                AuditCategory.AUTHENTICATION,
                AuditAction.LOGIN_SUCCESS,
            )
            await session.commit()

            entry2 = await audit_log(
                session,
                AuditCategory.AUTHENTICATION,
                AuditAction.LOGOUT,
            )
            await session.commit()

            # Second entry should reference first entry's hash
            assert entry2.previous_hash == entry1.entry_hash

    @pytest.mark.anyio
    async def test_audit_log_with_details(self, test_db) -> None:
        """Should store JSON details."""
        async with test_db.session() as session:
            details = {
                "method": "password",
                "mfa_used": True,
                "ip_country": "US",
            }

            entry = await audit_log(
                session,
                AuditCategory.AUTHENTICATION,
                AuditAction.LOGIN_SUCCESS,
                details=details,
            )
            await session.commit()

            assert entry.details is not None
            parsed = json.loads(entry.details)
            assert parsed == details

    @pytest.mark.anyio
    async def test_audit_log_severity_levels(self, test_db) -> None:
        """Should handle different severity levels."""
        async with test_db.session() as session:
            for severity in AuditSeverity:
                entry = await audit_log(
                    session,
                    AuditCategory.SECURITY,
                    AuditAction.SUSPICIOUS_ACTIVITY,
                    severity=severity,
                )
                await session.commit()

                assert entry.severity == severity.value


class TestAuditLogQueries:
    """Tests for audit log query functions."""

    @pytest.mark.anyio
    async def test_query_by_category(self, test_db, sample_audit_logs) -> None:
        """Should filter by category."""
        from atp.dashboard.audit import AuditLogFilter, query_audit_logs

        async with test_db.session() as session:
            filters = AuditLogFilter(category=AuditCategory.AUTHENTICATION)
            entries, total = await query_audit_logs(
                session,
                filters=filters,
            )

            assert total >= 2  # We created 2 auth logs
            assert all(
                e.category == AuditCategory.AUTHENTICATION.value for e in entries
            )

    @pytest.mark.anyio
    async def test_query_by_date_range(self, test_db, sample_audit_logs) -> None:
        """Should filter by date range."""
        from atp.dashboard.audit import AuditLogFilter, query_audit_logs

        async with test_db.session() as session:
            # All logs should be within the last hour
            filters = AuditLogFilter(
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow() + timedelta(hours=1),
            )
            entries, total = await query_audit_logs(
                session,
                filters=filters,
            )

            assert total == len(sample_audit_logs)

    @pytest.mark.anyio
    async def test_query_pagination(self, test_db, sample_audit_logs) -> None:
        """Should handle pagination."""
        from atp.dashboard.audit import query_audit_logs

        async with test_db.session() as session:
            entries, total = await query_audit_logs(
                session,
                offset=0,
                limit=2,
            )

            assert len(entries) == 2
            assert total >= len(sample_audit_logs)


class TestAuditChainVerification:
    """Tests for audit chain verification."""

    @pytest.mark.anyio
    async def test_verify_valid_chain(self, test_db) -> None:
        """Should verify a valid chain."""
        from atp.dashboard.audit import verify_audit_chain

        async with test_db.session() as session:
            # Create a chain of entries
            for _ in range(5):
                await audit_log(
                    session,
                    AuditCategory.AUTHENTICATION,
                    AuditAction.LOGIN_SUCCESS,
                )
                await session.commit()

            is_valid, invalid_entries = await verify_audit_chain(session)

            assert is_valid
            assert len(invalid_entries) == 0


class TestRetentionPolicy:
    """Tests for retention policy functionality."""

    @pytest.mark.anyio
    async def test_retention_skips_recent_entries(self, test_db) -> None:
        """Should not delete recent entries."""
        from atp.dashboard.audit import RetentionPolicy, apply_retention_policy

        async with test_db.session() as session:
            # Create some entries
            for _ in range(5):
                await audit_log(
                    session,
                    AuditCategory.DATA_ACCESS,
                    AuditAction.DATA_READ,
                )
            await session.commit()

            # Apply retention with short period but high min entries
            policy = RetentionPolicy(
                default_retention_days=1,
                min_entries_to_keep=1000,  # High minimum
            )

            deleted = await apply_retention_policy(
                session,
                policy=policy,
            )
            await session.commit()

            # Should not delete since we're below min_entries
            assert deleted == 0

    @pytest.mark.anyio
    async def test_retention_enforces_max_entries(self, test_db) -> None:
        """Should enforce max entries limit."""
        from atp.dashboard.audit import RetentionPolicy, apply_retention_policy

        async with test_db.session() as session:
            # Create many entries (more than 1000 to test max_entries)
            for _ in range(1050):
                await audit_log(
                    session,
                    AuditCategory.DATA_ACCESS,
                    AuditAction.DATA_READ,
                )
            await session.commit()

            # Apply retention with max entries at 1000 (minimum allowed)
            policy = RetentionPolicy(
                default_retention_days=365,  # Long retention
                min_entries_to_keep=0,
                max_entries=1000,  # Minimum allowed value
            )

            deleted = await apply_retention_policy(
                session,
                policy=policy,
            )
            await session.commit()

            # Should delete to get down to max_entries
            assert deleted == 50


class TestAuditStatistics:
    """Tests for audit statistics."""

    @pytest.mark.anyio
    async def test_get_statistics(self, test_db, sample_audit_logs) -> None:
        """Should return audit statistics."""
        from atp.dashboard.audit import get_audit_statistics

        async with test_db.session() as session:
            stats = await get_audit_statistics(session, days=30)

            assert stats.total_entries >= len(sample_audit_logs)
            assert AuditCategory.AUTHENTICATION.value in stats.entries_by_category
            assert AuditSeverity.INFO.value in stats.entries_by_severity
