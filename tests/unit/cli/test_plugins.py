"""Tests for CLI plugins commands."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from atp.cli.commands.plugins import (
    EXIT_ERROR,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    GROUP_DISPLAY_MAP,
    GROUP_TYPE_MAP,
    _create_plugins_table,
    _find_plugin_by_name,
    _get_all_plugins,
    _get_groups_from_type,
    _get_plugin_type_display,
)
from atp.cli.main import cli
from atp.plugins import ADAPTER_GROUP, ALL_GROUPS, EVALUATOR_GROUP, REPORTER_GROUP


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_plugin_info() -> MagicMock:
    """Create a mock PluginInfo."""
    info = MagicMock()
    info.name = "http"
    info.group = ADAPTER_GROUP
    info.module = "atp.adapters.http"
    info.attr = "HTTPAdapter"
    info.version = "1.0.0"
    info.author = "ATP Team"
    info.description = "HTTP adapter for remote agents"
    info.package = "atp-platform"
    info.full_path = "atp.adapters.http:HTTPAdapter"
    info.has_config_schema = False
    info.config_metadata = None
    return info


@pytest.fixture
def mock_lazy_plugin(mock_plugin_info: MagicMock) -> MagicMock:
    """Create a mock LazyPlugin."""
    lazy = MagicMock()
    lazy.info = mock_plugin_info
    lazy.is_loaded = False
    lazy.has_error = False
    lazy.load.return_value = MagicMock()
    return lazy


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_plugin_type_display(self) -> None:
        """Test getting display name for plugin groups."""
        assert _get_plugin_type_display(ADAPTER_GROUP) == "adapter"
        assert _get_plugin_type_display(EVALUATOR_GROUP) == "evaluator"
        assert _get_plugin_type_display(REPORTER_GROUP) == "reporter"
        assert _get_plugin_type_display("unknown.group") == "unknown.group"

    def test_get_groups_from_type_all(self) -> None:
        """Test getting all groups when type is None."""
        groups = _get_groups_from_type(None)
        assert groups == ALL_GROUPS
        assert ADAPTER_GROUP in groups
        assert EVALUATOR_GROUP in groups
        assert REPORTER_GROUP in groups

    def test_get_groups_from_type_adapter(self) -> None:
        """Test getting adapter group only."""
        groups = _get_groups_from_type("adapter")
        assert groups == [ADAPTER_GROUP]

    def test_get_groups_from_type_evaluator(self) -> None:
        """Test getting evaluator group only."""
        groups = _get_groups_from_type("evaluator")
        assert groups == [EVALUATOR_GROUP]

    def test_get_groups_from_type_reporter(self) -> None:
        """Test getting reporter group only."""
        groups = _get_groups_from_type("reporter")
        assert groups == [REPORTER_GROUP]

    def test_get_groups_from_type_unknown(self) -> None:
        """Test getting empty list for unknown type."""
        groups = _get_groups_from_type("unknown")
        assert groups == []

    def test_create_plugins_table(self) -> None:
        """Test creating a Rich table for plugins."""
        table = _create_plugins_table("Test Plugins")
        assert table.title == "Test Plugins"
        # Check columns exist
        assert len(table.columns) == 5

    def test_group_type_map(self) -> None:
        """Test GROUP_TYPE_MAP contains expected mappings."""
        assert GROUP_TYPE_MAP["adapter"] == ADAPTER_GROUP
        assert GROUP_TYPE_MAP["evaluator"] == EVALUATOR_GROUP
        assert GROUP_TYPE_MAP["reporter"] == REPORTER_GROUP

    def test_group_display_map(self) -> None:
        """Test GROUP_DISPLAY_MAP contains expected mappings."""
        assert GROUP_DISPLAY_MAP[ADAPTER_GROUP] == "adapter"
        assert GROUP_DISPLAY_MAP[EVALUATOR_GROUP] == "evaluator"
        assert GROUP_DISPLAY_MAP[REPORTER_GROUP] == "reporter"


class TestGetAllPlugins:
    """Tests for _get_all_plugins function."""

    def test_get_all_plugins(self, mock_lazy_plugin: MagicMock) -> None:
        """Test getting all plugins from groups."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {"http": mock_lazy_plugin}
            mock_manager.return_value = manager

            plugins = _get_all_plugins([ADAPTER_GROUP])

            assert len(plugins) == 1
            assert plugins[0][0] == ADAPTER_GROUP
            assert plugins[0][1] == mock_lazy_plugin

    def test_get_all_plugins_multiple_groups(self, mock_lazy_plugin: MagicMock) -> None:
        """Test getting plugins from multiple groups."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {"http": mock_lazy_plugin}
            mock_manager.return_value = manager

            plugins = _get_all_plugins([ADAPTER_GROUP, EVALUATOR_GROUP])

            assert len(plugins) == 2
            manager.discover_plugins.assert_any_call(ADAPTER_GROUP)
            manager.discover_plugins.assert_any_call(EVALUATOR_GROUP)

    def test_get_all_plugins_empty(self) -> None:
        """Test getting plugins when none exist."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {}
            mock_manager.return_value = manager

            plugins = _get_all_plugins([ADAPTER_GROUP])

            assert len(plugins) == 0


class TestFindPluginByName:
    """Tests for _find_plugin_by_name function."""

    def test_find_plugin_by_name_found(self, mock_lazy_plugin: MagicMock) -> None:
        """Test finding a plugin by name."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            mock_manager.return_value = manager

            result = _find_plugin_by_name("http", "adapter")

            assert result is not None
            assert result[0] == ADAPTER_GROUP
            assert result[1] == mock_lazy_plugin

    def test_find_plugin_by_name_not_found(self) -> None:
        """Test finding a plugin that doesn't exist."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = None
            mock_manager.return_value = manager

            result = _find_plugin_by_name("nonexistent", "adapter")

            assert result is None

    def test_find_plugin_by_name_all_groups(self, mock_lazy_plugin: MagicMock) -> None:
        """Test finding a plugin across all groups."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.side_effect = [None, mock_lazy_plugin, None]
            mock_manager.return_value = manager

            result = _find_plugin_by_name("evaluator_plugin", None)

            # Should check all groups until found
            assert result is not None


class TestPluginsListCommand:
    """Tests for plugins list command."""

    def test_plugins_list_help(self, runner: CliRunner) -> None:
        """Test plugins list help."""
        result = runner.invoke(cli, ["plugins", "list", "--help"])
        assert result.exit_code == 0
        assert "List all discovered plugins" in result.output
        assert "--type" in result.output

    def test_plugins_list_all(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test listing all plugins."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {"http": mock_lazy_plugin}
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "list"])

            assert result.exit_code == EXIT_SUCCESS
            assert "http" in result.output

    def test_plugins_list_with_type_filter(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test listing plugins with type filter."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {"http": mock_lazy_plugin}
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "list", "--type=adapter"])

            assert result.exit_code == EXIT_SUCCESS
            # Should only query adapter group
            manager.discover_plugins.assert_called_once_with(ADAPTER_GROUP)

    def test_plugins_list_empty(self, runner: CliRunner) -> None:
        """Test listing plugins when none exist."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {}
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "list"])

            assert result.exit_code == EXIT_SUCCESS
            assert "No" in result.output and "plugins found" in result.output

    def test_plugins_list_verbose(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test listing plugins with verbose flag."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.discover_plugins.return_value = {"http": mock_lazy_plugin}
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "list", "--verbose"])

            assert result.exit_code == EXIT_SUCCESS
            assert "Total:" in result.output


class TestPluginsInfoCommand:
    """Tests for plugins info command."""

    def test_plugins_info_help(self, runner: CliRunner) -> None:
        """Test plugins info help."""
        result = runner.invoke(cli, ["plugins", "info", "--help"])
        assert result.exit_code == 0
        assert "Show detailed information about a plugin" in result.output

    def test_plugins_info_found(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test showing info for an existing plugin."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            manager.get_validation_errors.return_value = ([], None)
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "info", "http"])

            assert result.exit_code == EXIT_SUCCESS
            assert "http" in result.output
            assert "Plugin:" in result.output

    def test_plugins_info_not_found(self, runner: CliRunner) -> None:
        """Test showing info for a non-existent plugin."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = None
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "info", "nonexistent"])

            assert result.exit_code == EXIT_FAILURE
            assert "not found" in result.output

    def test_plugins_info_with_type(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test showing info with type filter."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            manager.get_validation_errors.return_value = ([], None)
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "info", "http", "--type=adapter"])

            assert result.exit_code == EXIT_SUCCESS
            # Should only check adapter group
            manager.get_plugin.assert_called_once_with(ADAPTER_GROUP, "http")

    def test_plugins_info_load_error(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test showing info when plugin fails to load."""
        mock_lazy_plugin.load.side_effect = ImportError("Module not found")

        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "info", "http"])

            assert result.exit_code == EXIT_SUCCESS
            assert "Failed to load plugin" in result.output


class TestPluginsEnableCommand:
    """Tests for plugins enable command."""

    def test_plugins_enable_help(self, runner: CliRunner) -> None:
        """Test plugins enable help."""
        result = runner.invoke(cli, ["plugins", "enable", "--help"])
        assert result.exit_code == 0
        assert "Enable a plugin" in result.output

    def test_plugins_enable_found(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test enabling an existing plugin."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "enable", "http"])

            assert result.exit_code == EXIT_SUCCESS
            assert "enabled" in result.output

    def test_plugins_enable_not_found(self, runner: CliRunner) -> None:
        """Test enabling a non-existent plugin."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = None
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "enable", "nonexistent"])

            assert result.exit_code == EXIT_FAILURE
            assert "not found" in result.output

    def test_plugins_enable_load_error(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test enabling a plugin that fails to load."""
        mock_lazy_plugin.load.side_effect = ImportError("Module not found")

        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "enable", "http"])

            assert result.exit_code == EXIT_FAILURE
            assert "Cannot enable plugin" in result.output


class TestPluginsDisableCommand:
    """Tests for plugins disable command."""

    def test_plugins_disable_help(self, runner: CliRunner) -> None:
        """Test plugins disable help."""
        result = runner.invoke(cli, ["plugins", "disable", "--help"])
        assert result.exit_code == 0
        assert "Disable a plugin" in result.output

    def test_plugins_disable_found(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test disabling an existing plugin."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "disable", "http"])

            assert result.exit_code == EXIT_SUCCESS
            assert "disable" in result.output.lower()

    def test_plugins_disable_not_found(self, runner: CliRunner) -> None:
        """Test disabling a non-existent plugin."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = None
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "disable", "nonexistent"])

            assert result.exit_code == EXIT_FAILURE
            assert "not found" in result.output

    def test_plugins_disable_shows_config_hint(
        self, runner: CliRunner, mock_lazy_plugin: MagicMock
    ) -> None:
        """Test disabling shows configuration hint."""
        with patch("atp.cli.commands.plugins.get_plugin_manager") as mock_manager:
            manager = MagicMock()
            manager.get_plugin.return_value = mock_lazy_plugin
            mock_manager.return_value = manager

            result = runner.invoke(cli, ["plugins", "disable", "http"])

            assert result.exit_code == EXIT_SUCCESS
            assert "atp.config.yaml" in result.output
            assert "plugins:" in result.output


class TestPluginsCommandGroup:
    """Tests for plugins command group."""

    def test_plugins_help(self, runner: CliRunner) -> None:
        """Test plugins group help."""
        result = runner.invoke(cli, ["plugins", "--help"])
        assert result.exit_code == 0
        assert "Manage ATP plugins" in result.output
        assert "list" in result.output
        assert "info" in result.output
        assert "enable" in result.output
        assert "disable" in result.output

    def test_plugins_no_subcommand(self, runner: CliRunner) -> None:
        """Test plugins without subcommand shows usage."""
        result = runner.invoke(cli, ["plugins"])
        # Without a subcommand, Click shows usage and exits with code 2
        # (missing required subcommand)
        assert result.exit_code == EXIT_ERROR
        assert "Usage:" in result.output


class TestExitCodes:
    """Tests for exit codes."""

    def test_exit_codes_defined(self) -> None:
        """Test that exit codes are defined correctly."""
        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_ERROR == 2
