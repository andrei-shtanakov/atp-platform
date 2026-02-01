"""Unit tests for plugin discovery system."""

from collections.abc import Generator
from importlib.metadata import EntryPoint
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from atp.plugins.discovery import (
    ADAPTER_GROUP,
    ALL_GROUPS,
    EVALUATOR_GROUP,
    REPORTER_GROUP,
    LazyPlugin,
    PluginInfo,
    PluginManager,
    get_plugin_manager,
)
from atp.plugins.interfaces import (
    PluginValidationError,
    PluginVersionError,
)
from atp.protocol import ATPRequest, ATPResponse


class TestPluginInfo:
    """Tests for PluginInfo model."""

    def test_plugin_info_creation(self) -> None:
        """Test creating plugin info with required fields."""
        info = PluginInfo(
            name="test_plugin",
            group="atp.adapters",
            module="test_module",
            attr="TestClass",
        )
        assert info.name == "test_plugin"
        assert info.group == "atp.adapters"
        assert info.module == "test_module"
        assert info.attr == "TestClass"
        assert info.version is None
        assert info.author is None
        assert info.description is None

    def test_plugin_info_with_all_fields(self) -> None:
        """Test creating plugin info with all fields."""
        info = PluginInfo(
            name="test_plugin",
            group="atp.evaluators",
            module="my_package.module",
            attr="MyEvaluator",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            package="my-package",
        )
        assert info.version == "1.0.0"
        assert info.author == "Test Author"
        assert info.description == "A test plugin"
        assert info.package == "my-package"

    def test_full_path_property(self) -> None:
        """Test full_path property returns module:attr format."""
        info = PluginInfo(
            name="test",
            group="atp.adapters",
            module="foo.bar",
            attr="Baz",
        )
        assert info.full_path == "foo.bar:Baz"


class TestLazyPlugin:
    """Tests for LazyPlugin class."""

    def test_lazy_plugin_not_loaded_initially(self) -> None:
        """Test that plugin is not loaded on creation."""
        ep = MagicMock(spec=EntryPoint)
        ep.name = "test"
        ep.value = "test.module:TestClass"

        info = PluginInfo(
            name="test",
            group="atp.adapters",
            module="test.module",
            attr="TestClass",
        )

        plugin = LazyPlugin(ep, info)
        assert not plugin.is_loaded
        assert not plugin.has_error
        assert plugin.load_error is None
        assert plugin.info == info

    def test_lazy_plugin_loads_on_demand(self) -> None:
        """Test that plugin is loaded when accessed."""
        mock_class = MagicMock()

        ep = MagicMock(spec=EntryPoint)
        ep.name = "test"
        ep.value = "test.module:TestClass"
        ep.load.return_value = mock_class

        info = PluginInfo(
            name="test",
            group="atp.adapters",
            module="test.module",
            attr="TestClass",
        )

        plugin = LazyPlugin(ep, info)
        result = plugin.load()

        assert result is mock_class
        assert plugin.is_loaded
        ep.load.assert_called_once()

    def test_lazy_plugin_caches_loaded_class(self) -> None:
        """Test that loaded class is cached."""
        mock_class = MagicMock()

        ep = MagicMock(spec=EntryPoint)
        ep.name = "test"
        ep.value = "test.module:TestClass"
        ep.load.return_value = mock_class

        info = PluginInfo(
            name="test",
            group="atp.adapters",
            module="test.module",
            attr="TestClass",
        )

        plugin = LazyPlugin(ep, info)
        plugin.load()
        plugin.load()
        plugin.load()

        ep.load.assert_called_once()

    def test_lazy_plugin_handles_load_error(self) -> None:
        """Test that load errors are captured and re-raised."""
        ep = MagicMock(spec=EntryPoint)
        ep.name = "test"
        ep.value = "test.module:TestClass"
        ep.load.side_effect = ImportError("Module not found")

        info = PluginInfo(
            name="test",
            group="atp.adapters",
            module="test.module",
            attr="TestClass",
        )

        plugin = LazyPlugin(ep, info)

        with pytest.raises(ImportError, match="Module not found"):
            plugin.load()

        assert plugin.has_error
        assert isinstance(plugin.load_error, ImportError)

    def test_lazy_plugin_reraises_cached_error(self) -> None:
        """Test that cached errors are re-raised on subsequent calls."""
        ep = MagicMock(spec=EntryPoint)
        ep.name = "test"
        ep.value = "test.module:TestClass"
        ep.load.side_effect = ImportError("Module not found")

        info = PluginInfo(
            name="test",
            group="atp.adapters",
            module="test.module",
            attr="TestClass",
        )

        plugin = LazyPlugin(ep, info)

        with pytest.raises(ImportError):
            plugin.load()

        with pytest.raises(ImportError):
            plugin.load()

        # Should only attempt to load once
        ep.load.assert_called_once()


class TestPluginManager:
    """Tests for PluginManager class."""

    def test_discover_plugins_empty_group(self) -> None:
        """Test discovering plugins from an empty group."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            plugins = manager.discover_plugins("atp.test")

        assert plugins == {}
        assert manager.is_discovered("atp.test")

    def test_discover_plugins_caches_results(self) -> None:
        """Test that discovery results are cached."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "test"
        mock_ep.value = "test.module:TestClass"
        mock_ep.dist = None

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ) as mock_entry_points:
            manager.discover_plugins("atp.test")
            manager.discover_plugins("atp.test")

        mock_entry_points.assert_called_once_with(group="atp.test")

    def test_discover_plugins_extracts_metadata(self) -> None:
        """Test that plugin metadata is extracted from entry points."""
        manager = PluginManager()

        mock_dist = MagicMock()
        mock_dist.name = "my-package"
        mock_dist.version = "1.2.3"
        mock_dist.metadata = {
            "Author": "Test Author",
            "Summary": "Test description",
        }

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "my_adapter"
        mock_ep.value = "my_package.adapters:MyAdapter"
        mock_ep.dist = mock_dist

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            plugins = manager.discover_plugins("atp.adapters")

        assert "my_adapter" in plugins
        info = plugins["my_adapter"].info
        assert info.name == "my_adapter"
        assert info.module == "my_package.adapters"
        assert info.attr == "MyAdapter"
        assert info.version == "1.2.3"
        assert info.author == "Test Author"
        assert info.description == "Test description"
        assert info.package == "my-package"

    def test_get_plugin_returns_lazy_plugin(self) -> None:
        """Test getting a specific plugin."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "test"
        mock_ep.value = "test.module:TestClass"
        mock_ep.dist = None

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            plugin = manager.get_plugin("atp.test", "test")

        assert plugin is not None
        assert plugin.info.name == "test"

    def test_get_plugin_returns_none_for_missing(self) -> None:
        """Test getting a non-existent plugin returns None."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            plugin = manager.get_plugin("atp.test", "missing")

        assert plugin is None

    def test_load_plugin_returns_class(self) -> None:
        """Test loading a plugin returns the class."""
        manager = PluginManager()
        mock_class = MagicMock()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "test"
        mock_ep.value = "test.module:TestClass"
        mock_ep.dist = None
        mock_ep.load.return_value = mock_class

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            result = manager.load_plugin("atp.test", "test")

        assert result is mock_class

    def test_load_plugin_returns_none_for_missing(self) -> None:
        """Test loading non-existent plugin returns None."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            result = manager.load_plugin("atp.test", "missing")

        assert result is None

    def test_load_plugin_returns_none_on_error(self) -> None:
        """Test loading plugin that fails returns None."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "test"
        mock_ep.value = "test.module:TestClass"
        mock_ep.dist = None
        mock_ep.load.side_effect = ImportError("Not found")

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            result = manager.load_plugin("atp.test", "test")

        assert result is None

    def test_list_plugins_returns_names(self) -> None:
        """Test listing plugin names."""
        manager = PluginManager()

        eps = []
        for name in ["plugin_a", "plugin_b", "plugin_c"]:
            ep = MagicMock(spec=EntryPoint)
            ep.name = name
            ep.value = f"test.{name}:Class"
            ep.dist = None
            eps.append(ep)

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=eps,
        ):
            names = manager.list_plugins("atp.test")

        assert set(names) == {"plugin_a", "plugin_b", "plugin_c"}

    def test_list_plugin_info_returns_metadata(self) -> None:
        """Test listing plugin metadata."""
        manager = PluginManager()

        eps = []
        for name in ["plugin_a", "plugin_b"]:
            ep = MagicMock(spec=EntryPoint)
            ep.name = name
            ep.value = f"test.{name}:Class"
            ep.dist = None
            eps.append(ep)

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=eps,
        ):
            infos = manager.list_plugin_info("atp.test")

        assert len(infos) == 2
        assert all(isinstance(i, PluginInfo) for i in infos)

    def test_discover_all_discovers_all_groups(self) -> None:
        """Test discovering all plugin groups."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ) as mock_entry_points:
            manager.discover_all()

        assert mock_entry_points.call_count == len(ALL_GROUPS)
        for group in ALL_GROUPS:
            assert manager.is_discovered(group)

    def test_clear_cache_clears_specific_group(self) -> None:
        """Test clearing cache for specific group."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            manager.discover_plugins("atp.adapters")
            manager.discover_plugins("atp.evaluators")

        assert manager.is_discovered("atp.adapters")
        assert manager.is_discovered("atp.evaluators")

        manager.clear_cache("atp.adapters")

        assert not manager.is_discovered("atp.adapters")
        assert manager.is_discovered("atp.evaluators")

    def test_clear_cache_clears_all(self) -> None:
        """Test clearing entire cache."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            manager.discover_all()

        manager.clear_cache()

        for group in ALL_GROUPS:
            assert not manager.is_discovered(group)

    def test_get_all_adapters(self) -> None:
        """Test convenience method for getting adapters."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "http"
        mock_ep.value = "atp.adapters.http:HTTPAdapter"
        mock_ep.dist = None

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            adapters = manager.get_all_adapters()

        assert "http" in adapters
        assert manager.is_discovered(ADAPTER_GROUP)

    def test_get_all_evaluators(self) -> None:
        """Test convenience method for getting evaluators."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "artifact"
        mock_ep.value = "atp.evaluators.artifact:ArtifactEvaluator"
        mock_ep.dist = None

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            evaluators = manager.get_all_evaluators()

        assert "artifact" in evaluators
        assert manager.is_discovered(EVALUATOR_GROUP)

    def test_get_all_reporters(self) -> None:
        """Test convenience method for getting reporters."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "console"
        mock_ep.value = "atp.reporters.console:ConsoleReporter"
        mock_ep.dist = None

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            reporters = manager.get_all_reporters()

        assert "console" in reporters
        assert manager.is_discovered(REPORTER_GROUP)


class TestGetPluginManager:
    """Tests for global plugin manager."""

    def test_get_plugin_manager_returns_singleton(self) -> None:
        """Test that get_plugin_manager returns same instance."""
        # Reset global state
        import atp.plugins.discovery as discovery_module

        discovery_module._plugin_manager = None

        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is manager2

    def test_get_plugin_manager_returns_plugin_manager(self) -> None:
        """Test that get_plugin_manager returns PluginManager instance."""
        import atp.plugins.discovery as discovery_module

        discovery_module._plugin_manager = None

        manager = get_plugin_manager()

        assert isinstance(manager, PluginManager)


class TestEntryPointGroups:
    """Tests for entry point group constants."""

    def test_adapter_group_constant(self) -> None:
        """Test adapter group constant."""
        assert ADAPTER_GROUP == "atp.adapters"

    def test_evaluator_group_constant(self) -> None:
        """Test evaluator group constant."""
        assert EVALUATOR_GROUP == "atp.evaluators"

    def test_reporter_group_constant(self) -> None:
        """Test reporter group constant."""
        assert REPORTER_GROUP == "atp.reporters"

    def test_all_groups_contains_all(self) -> None:
        """Test ALL_GROUPS contains all groups."""
        assert ADAPTER_GROUP in ALL_GROUPS
        assert EVALUATOR_GROUP in ALL_GROUPS
        assert REPORTER_GROUP in ALL_GROUPS
        assert len(ALL_GROUPS) == 3


class TestPluginManagerValidation:
    """Tests for PluginManager validation methods."""

    def test_validate_plugin_valid_adapter(self) -> None:
        """Test _validate_plugin with valid adapter."""
        manager = PluginManager()

        class ValidAdapter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def adapter_type(self) -> str:
                return "valid"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        result = manager._validate_plugin(ValidAdapter, ADAPTER_GROUP, "test")
        assert result is True

    def test_validate_plugin_invalid_adapter_raises(self) -> None:
        """Test _validate_plugin raises for invalid adapter."""
        manager = PluginManager()

        class InvalidAdapter:
            pass

        with pytest.raises(PluginValidationError) as exc_info:
            manager._validate_plugin(InvalidAdapter, ADAPTER_GROUP, "test")

        assert exc_info.value.plugin_name == "test"
        assert exc_info.value.plugin_type == ADAPTER_GROUP
        assert len(exc_info.value.errors) > 0

    def test_validate_plugin_version_incompatible_raises(self) -> None:
        """Test _validate_plugin raises for version incompatibility."""
        manager = PluginManager()

        class IncompatibleAdapter:
            atp_version: ClassVar[str] = "99.0.0"

            @property
            def adapter_type(self) -> str:
                return "test"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        with pytest.raises(PluginVersionError) as exc_info:
            manager._validate_plugin(IncompatibleAdapter, ADAPTER_GROUP, "test")

        assert exc_info.value.plugin_name == "test"
        assert exc_info.value.required_version == "99.0.0"

    def test_load_and_validate_plugin_valid(self) -> None:
        """Test load_and_validate_plugin with valid plugin."""
        manager = PluginManager()

        class ValidAdapter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def adapter_type(self) -> str:
                return "valid"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "valid"
        mock_ep.value = "test.module:ValidAdapter"
        mock_ep.dist = None
        mock_ep.load.return_value = ValidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            result = manager.load_and_validate_plugin(ADAPTER_GROUP, "valid")

        assert result is ValidAdapter

    def test_load_and_validate_plugin_invalid_raises(self) -> None:
        """Test load_and_validate_plugin raises for invalid plugin."""
        manager = PluginManager()

        class InvalidAdapter:
            pass

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "invalid"
        mock_ep.value = "test.module:InvalidAdapter"
        mock_ep.dist = None
        mock_ep.load.return_value = InvalidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            with pytest.raises(PluginValidationError):
                manager.load_and_validate_plugin(ADAPTER_GROUP, "invalid")

    def test_load_and_validate_plugin_skips_validation_when_disabled(self) -> None:
        """Test load_and_validate_plugin skips validation when disabled."""
        manager = PluginManager()

        class InvalidAdapter:
            pass

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "invalid"
        mock_ep.value = "test.module:InvalidAdapter"
        mock_ep.dist = None
        mock_ep.load.return_value = InvalidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            # Should not raise because validation is disabled
            result = manager.load_and_validate_plugin(
                ADAPTER_GROUP, "invalid", validate=False
            )

        assert result is InvalidAdapter

    def test_load_and_validate_plugin_not_found(self) -> None:
        """Test load_and_validate_plugin returns None for missing plugin."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            result = manager.load_and_validate_plugin(ADAPTER_GROUP, "missing")

        assert result is None

    def test_discover_and_validate_plugins_filters_invalid(self) -> None:
        """Test discover_and_validate_plugins filters out invalid plugins."""
        manager = PluginManager()

        class ValidAdapter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def adapter_type(self) -> str:
                return "valid"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        class InvalidAdapter:
            pass

        valid_ep = MagicMock(spec=EntryPoint)
        valid_ep.name = "valid"
        valid_ep.value = "test.module:ValidAdapter"
        valid_ep.dist = None
        valid_ep.load.return_value = ValidAdapter

        invalid_ep = MagicMock(spec=EntryPoint)
        invalid_ep.name = "invalid"
        invalid_ep.value = "test.module:InvalidAdapter"
        invalid_ep.dist = None
        invalid_ep.load.return_value = InvalidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[valid_ep, invalid_ep],
        ):
            result = manager.discover_and_validate_plugins(ADAPTER_GROUP, validate=True)

        assert "valid" in result
        assert "invalid" not in result

    def test_discover_and_validate_plugins_includes_all_when_no_validation(
        self,
    ) -> None:
        """Test discover_and_validate_plugins includes all when validation disabled."""
        manager = PluginManager()

        class InvalidAdapter:
            pass

        invalid_ep = MagicMock(spec=EntryPoint)
        invalid_ep.name = "invalid"
        invalid_ep.value = "test.module:InvalidAdapter"
        invalid_ep.dist = None
        invalid_ep.load.return_value = InvalidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[invalid_ep],
        ):
            result = manager.discover_and_validate_plugins(
                ADAPTER_GROUP, validate=False
            )

        assert "invalid" in result

    def test_get_validation_errors_returns_errors(self) -> None:
        """Test get_validation_errors returns errors for invalid plugin."""
        manager = PluginManager()

        class InvalidAdapter:
            pass

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "invalid"
        mock_ep.value = "test.module:InvalidAdapter"
        mock_ep.dist = None
        mock_ep.load.return_value = InvalidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            result = manager.get_validation_errors(ADAPTER_GROUP, "invalid")

        assert result is not None
        errors, version_error = result
        assert len(errors) > 0

    def test_get_validation_errors_returns_none_for_missing(self) -> None:
        """Test get_validation_errors returns None for missing plugin."""
        manager = PluginManager()

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[],
        ):
            result = manager.get_validation_errors(ADAPTER_GROUP, "missing")

        assert result is None

    def test_get_validation_errors_returns_none_for_load_failure(self) -> None:
        """Test get_validation_errors returns None if plugin fails to load."""
        manager = PluginManager()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "broken"
        mock_ep.value = "test.module:BrokenAdapter"
        mock_ep.dist = None
        mock_ep.load.side_effect = ImportError("Module not found")

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            result = manager.get_validation_errors(ADAPTER_GROUP, "broken")

        assert result is None

    def test_get_validation_errors_valid_plugin(self) -> None:
        """Test get_validation_errors returns empty for valid plugin."""
        manager = PluginManager()

        class ValidAdapter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def adapter_type(self) -> str:
                return "valid"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        mock_ep = MagicMock(spec=EntryPoint)
        mock_ep.name = "valid"
        mock_ep.value = "test.module:ValidAdapter"
        mock_ep.dist = None
        mock_ep.load.return_value = ValidAdapter

        with patch(
            "atp.plugins.discovery.entry_points",
            return_value=[mock_ep],
        ):
            result = manager.get_validation_errors(ADAPTER_GROUP, "valid")

        assert result is not None
        errors, version_error = result
        assert errors == []
        assert version_error is None
