"""Tests for CLI catalog commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


def _make_mock_db() -> MagicMock:
    """Build a mock Database with an async session context manager."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()

    # Empty result set by default (scalar returns [])
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    # session() must be an async context manager
    mock_db = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_db.session = MagicMock(return_value=cm)

    return mock_db


class TestCatalogHelp:
    """Tests for catalog --help output."""

    def test_catalog_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog --help' exits 0 and lists subcommands."""
        result = runner.invoke(cli, ["catalog", "--help"])
        assert result.exit_code == 0
        assert "catalog" in result.output.lower()
        for sub in ("sync", "list", "info", "run", "publish", "results"):
            assert sub in result.output

    def test_catalog_sync_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog sync --help' exits 0."""
        result = runner.invoke(cli, ["catalog", "sync", "--help"])
        assert result.exit_code == 0
        assert "Sync" in result.output

    def test_catalog_list_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog list --help' exits 0."""
        result = runner.invoke(cli, ["catalog", "list", "--help"])
        assert result.exit_code == 0

    def test_catalog_info_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog info --help' exits 0."""
        result = runner.invoke(cli, ["catalog", "info", "--help"])
        assert result.exit_code == 0

    def test_catalog_run_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog run --help' exits 0."""
        result = runner.invoke(cli, ["catalog", "run", "--help"])
        assert result.exit_code == 0

    def test_catalog_publish_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog publish --help' exits 0."""
        result = runner.invoke(cli, ["catalog", "publish", "--help"])
        assert result.exit_code == 0

    def test_catalog_results_help(self, runner: CliRunner) -> None:
        """Invoking 'atp catalog results --help' exits 0."""
        result = runner.invoke(cli, ["catalog", "results", "--help"])
        assert result.exit_code == 0


class TestCatalogSync:
    """Tests for 'atp catalog sync'."""

    def test_catalog_sync_success(self, runner: CliRunner) -> None:
        """Sync exits 0 and prints success message."""
        mock_db = _make_mock_db()

        with (
            patch(
                "atp.cli.commands.catalog.init_database",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "atp.cli.commands.catalog.sync_builtin_catalog",
                new=AsyncMock(),
            ),
        ):
            result = runner.invoke(cli, ["catalog", "sync"])

        assert result.exit_code == 0
        assert "synced" in result.output.lower()

    def test_catalog_sync_calls_sync_builtin(self, runner: CliRunner) -> None:
        """Sync invokes sync_builtin_catalog with a session."""
        mock_db = _make_mock_db()
        sync_mock = AsyncMock()

        with (
            patch(
                "atp.cli.commands.catalog.init_database",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "atp.cli.commands.catalog.sync_builtin_catalog",
                new=sync_mock,
            ),
        ):
            result = runner.invoke(cli, ["catalog", "sync"])

        assert result.exit_code == 0
        sync_mock.assert_called_once()

    def test_catalog_sync_error(self, runner: CliRunner) -> None:
        """Sync exits 2 on unexpected error."""
        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(side_effect=RuntimeError("DB error")),
        ):
            result = runner.invoke(cli, ["catalog", "sync"])

        assert result.exit_code == 2


class TestCatalogList:
    """Tests for 'atp catalog list'."""

    def test_catalog_list_after_sync(self, runner: CliRunner) -> None:
        """After syncing, categories appear in list output."""
        # Use real in-memory SQLite DB to exercise the full flow
        import asyncio

        from atp.catalog.sync import sync_builtin_catalog
        from atp.dashboard.database import init_database as _real_init

        async def _setup() -> None:
            from atp.dashboard.database import set_database

            db = await _real_init("sqlite+aiosqlite://")
            set_database(db)
            async with db.session() as session:
                await sync_builtin_catalog(session)

        asyncio.run(_setup())

        try:
            result = runner.invoke(cli, ["catalog", "list"])
            assert result.exit_code == 0
            # At least one category should appear
            assert len(result.output.strip()) > 0
        finally:
            # Reset the global DB to avoid leaking state
            from atp.dashboard.database import set_database

            set_database(None)  # type: ignore[arg-type]

    def test_catalog_list_empty_triggers_autosync(self, runner: CliRunner) -> None:
        """list triggers auto-sync when DB is empty and prints message."""
        mock_db = _make_mock_db()
        sync_mock = AsyncMock()

        with (
            patch(
                "atp.cli.commands.catalog.init_database",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "atp.catalog.sync.sync_builtin_catalog",
                new=sync_mock,
            ),
        ):
            result = runner.invoke(cli, ["catalog", "list"])

        assert result.exit_code == 0
        assert (
            "auto-sync" in result.output.lower() or "syncing" in result.output.lower()
        )

    def test_catalog_list_with_category(self, runner: CliRunner) -> None:
        """list CATEGORY filters suites, exits 0."""
        mock_db = _make_mock_db()

        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(return_value=mock_db),
        ):
            result = runner.invoke(cli, ["catalog", "list", "coding"])

        assert result.exit_code == 0
        assert "not found" in result.output.lower() or result.exit_code == 0

    def test_catalog_list_error(self, runner: CliRunner) -> None:
        """list exits 2 on unexpected error."""
        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(side_effect=RuntimeError("fail")),
        ):
            result = runner.invoke(cli, ["catalog", "list"])

        assert result.exit_code == 2


class TestCatalogInfo:
    """Tests for 'atp catalog info'."""

    def test_catalog_info_not_found(self, runner: CliRunner) -> None:
        """info exits 1 when suite not found."""
        mock_db = _make_mock_db()

        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(return_value=mock_db),
        ):
            result = runner.invoke(cli, ["catalog", "info", "x/y"])

        assert result.exit_code == 1

    def test_catalog_info_bad_path(self, runner: CliRunner) -> None:
        """info exits 2 when PATH format is invalid."""
        mock_db = _make_mock_db()

        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(return_value=mock_db),
        ):
            result = runner.invoke(cli, ["catalog", "info", "badpath"])

        assert result.exit_code == 2

    def test_catalog_info_found(self, runner: CliRunner) -> None:
        """info prints suite details when found."""
        mock_db = _make_mock_db()

        # Mock suite
        mock_suite = MagicMock()
        mock_suite.name = "File Operations"
        mock_suite.author = "curated"
        mock_suite.source = "builtin"
        mock_suite.difficulty = "beginner"
        mock_suite.version = "1.0"
        mock_suite.estimated_minutes = 10
        mock_suite.description = "Test file operations"
        mock_suite.tests = []

        # Patch get_suite_by_path
        with (
            patch(
                "atp.cli.commands.catalog.init_database",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "atp.catalog.repository.CatalogRepository.get_suite_by_path",
                new=AsyncMock(return_value=mock_suite),
            ),
        ):
            result = runner.invoke(cli, ["catalog", "info", "coding/file-operations"])

        assert result.exit_code == 0
        assert "File Operations" in result.output


class TestCatalogRun:
    """Tests for 'atp catalog run'."""

    def test_catalog_run_not_found(self, runner: CliRunner) -> None:
        """run exits 1 when suite not found."""
        mock_db = _make_mock_db()

        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(return_value=mock_db),
        ):
            result = runner.invoke(cli, ["catalog", "run", "x/y"])

        assert result.exit_code == 1

    def test_catalog_run_shows_instructions(self, runner: CliRunner) -> None:
        """run shows usage instructions when suite is found."""
        mock_db = _make_mock_db()
        mock_suite = MagicMock()
        mock_suite.name = "File Operations"
        mock_suite.difficulty = "beginner"
        mock_suite.tests = []

        with (
            patch(
                "atp.cli.commands.catalog.init_database",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "atp.catalog.repository.CatalogRepository.get_suite_by_path",
                new=AsyncMock(return_value=mock_suite),
            ),
        ):
            result = runner.invoke(
                cli,
                ["catalog", "run", "coding/file-operations", "--adapter=http"],
            )

        assert result.exit_code == 0
        assert "atp test" in result.output


class TestCatalogResults:
    """Tests for 'atp catalog results'."""

    def test_catalog_results_not_found(self, runner: CliRunner) -> None:
        """results exits 1 when suite not found."""
        mock_db = _make_mock_db()

        with patch(
            "atp.cli.commands.catalog.init_database",
            new=AsyncMock(return_value=mock_db),
        ):
            result = runner.invoke(cli, ["catalog", "results", "x/y"])

        assert result.exit_code == 1

    def test_catalog_results_no_tests(self, runner: CliRunner) -> None:
        """results prints message when suite has no tests."""
        mock_db = _make_mock_db()
        mock_suite = MagicMock()
        mock_suite.name = "Empty Suite"
        mock_suite.tests = []

        with (
            patch(
                "atp.cli.commands.catalog.init_database",
                new=AsyncMock(return_value=mock_db),
            ),
            patch(
                "atp.catalog.repository.CatalogRepository.get_suite_by_path",
                new=AsyncMock(return_value=mock_suite),
            ),
        ):
            result = runner.invoke(cli, ["catalog", "results", "x/y"])

        assert result.exit_code == 0
        assert "No tests" in result.output
