"""Tests for ATP Dashboard v2 dependencies module."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.dependencies import (
    PaginationParams,
    get_dashboard_config,
    get_db,
    get_db_session,
    require_feature,
)


class TestGetDbSession:
    """Tests for get_db_session dependency."""

    @pytest.mark.anyio
    async def test_yields_session(self, tmp_path: Path) -> None:
        """Test that get_db_session yields a session."""
        from atp.dashboard.database import Database, set_database

        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        db = Database(url=url)
        await db.create_tables()
        set_database(db)

        try:
            session = None
            async for s in get_db_session():
                session = s
                break
            assert session is not None
        finally:
            await db.close()

    @pytest.mark.anyio
    async def test_commits_on_success(self, tmp_path: Path) -> None:
        """Test that session commits on successful completion."""
        from atp.dashboard.database import Database, set_database

        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        db = Database(url=url)
        await db.create_tables()
        set_database(db)

        try:
            async for session in get_db_session():
                # Session should be usable
                assert session is not None
        finally:
            await db.close()

    @pytest.mark.anyio
    async def test_rollback_on_exception(self, tmp_path: Path) -> None:
        """Test that session rolls back on exception."""
        from atp.dashboard.database import Database, set_database

        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        db = Database(url=url)
        await db.create_tables()
        set_database(db)

        try:
            with pytest.raises(ValueError):
                async for session in get_db_session():
                    raise ValueError("Test error")
        finally:
            await db.close()


class TestGetDashboardConfig:
    """Tests for get_dashboard_config dependency."""

    def test_returns_dashboard_config(self) -> None:
        """Test that get_dashboard_config returns DashboardConfig."""
        config = get_dashboard_config()
        assert isinstance(config, DashboardConfig)

    def test_returns_cached_config(self) -> None:
        """Test that get_dashboard_config returns same cached config."""
        config1 = get_dashboard_config()
        config2 = get_dashboard_config()
        assert config1 is config2


class TestGetDb:
    """Tests for get_db dependency."""

    def test_returns_database_instance(self) -> None:
        """Test that get_db returns Database instance."""
        from atp.dashboard.database import Database

        db = get_db()
        assert isinstance(db, Database)


class TestPaginationParams:
    """Tests for PaginationParams class."""

    def test_default_values(self) -> None:
        """Test default pagination values."""
        params = PaginationParams()
        assert params.offset == 0
        assert params.limit == 50

    def test_custom_values(self) -> None:
        """Test custom pagination values."""
        params = PaginationParams(offset=10, limit=25)
        assert params.offset == 10
        assert params.limit == 25

    def test_negative_offset_clamped(self) -> None:
        """Test that negative offset is clamped to 0."""
        params = PaginationParams(offset=-5)
        assert params.offset == 0

    def test_limit_minimum_clamped(self) -> None:
        """Test that limit is clamped to minimum of 1."""
        params = PaginationParams(limit=0)
        assert params.limit == 1
        params = PaginationParams(limit=-5)
        assert params.limit == 1

    def test_limit_maximum_clamped(self) -> None:
        """Test that limit is clamped to maximum of 100."""
        params = PaginationParams(limit=200)
        assert params.limit == 100


class TestRequireFeature:
    """Tests for require_feature dependency."""

    def test_returns_callable(self) -> None:
        """Test that require_feature returns a callable."""
        check_feature = require_feature("test_feature")
        assert callable(check_feature)

    def test_raises_404_when_feature_disabled(self) -> None:
        """Test that require_feature raises 404 when feature is disabled."""
        check_feature = require_feature("disabled_feature")
        config = DashboardConfig(debug=False)

        with pytest.raises(HTTPException) as exc_info:
            check_feature(config)
        assert exc_info.value.status_code == 404
        assert "disabled_feature" in str(exc_info.value.detail)

    def test_passes_when_debug_enabled(self) -> None:
        """Test that require_feature passes when debug is enabled."""
        check_feature = require_feature("any_feature")
        config = DashboardConfig(debug=True)

        # Should not raise
        check_feature(config)
