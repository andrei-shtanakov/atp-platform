"""Plugin discovery and management via Python entry points."""

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from pydantic import BaseModel, Field

import atp
from atp.plugins.config import ConfigMetadata, PluginConfig
from atp.plugins.interfaces import (
    PluginValidationError,
    PluginVersionError,
    check_version_compatibility,
    get_plugin_config_schema,
    validate_plugin,
)

logger = logging.getLogger(__name__)


# Entry point group names
ADAPTER_GROUP = "atp.adapters"
EVALUATOR_GROUP = "atp.evaluators"
REPORTER_GROUP = "atp.reporters"

ALL_GROUPS = [ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP]


class PluginInfo(BaseModel):
    """Metadata for a discovered plugin."""

    name: str = Field(..., description="Plugin name (entry point name)")
    group: str = Field(..., description="Entry point group (e.g., atp.adapters)")
    module: str = Field(..., description="Module path where plugin is defined")
    attr: str = Field(..., description="Attribute name in module")
    version: str | None = Field(None, description="Plugin package version")
    author: str | None = Field(None, description="Plugin author")
    description: str | None = Field(None, description="Plugin description")
    package: str | None = Field(None, description="Package name providing this plugin")
    config_metadata: ConfigMetadata | None = Field(
        None, description="Configuration schema metadata if defined"
    )

    @property
    def full_path(self) -> str:
        """Return full import path (module:attr)."""
        return f"{self.module}:{self.attr}"

    @property
    def has_config_schema(self) -> bool:
        """Check if plugin has a configuration schema defined."""
        return self.config_metadata is not None


class LazyPlugin:
    """
    Lazy-loading wrapper for plugins.

    The actual plugin class is not loaded until accessed.
    """

    def __init__(self, entry_point: EntryPoint, info: PluginInfo) -> None:
        """Initialize lazy plugin wrapper.

        Args:
            entry_point: The entry point to load from.
            info: Plugin metadata.
        """
        self._entry_point = entry_point
        self._info = info
        self._loaded: type[Any] | None = None
        self._load_error: Exception | None = None

    @property
    def info(self) -> PluginInfo:
        """Return plugin metadata."""
        return self._info

    @property
    def is_loaded(self) -> bool:
        """Check if plugin has been loaded."""
        return self._loaded is not None

    @property
    def has_error(self) -> bool:
        """Check if plugin failed to load."""
        return self._load_error is not None

    @property
    def load_error(self) -> Exception | None:
        """Return the load error if any."""
        return self._load_error

    def load(self) -> type[Any]:
        """Load and return the plugin class.

        Returns:
            The loaded plugin class.

        Raises:
            Exception: If plugin fails to load.
        """
        if self._loaded is not None:
            return self._loaded

        if self._load_error is not None:
            raise self._load_error

        try:
            self._loaded = self._entry_point.load()
            return self._loaded
        except Exception as e:
            self._load_error = e
            logger.warning(
                f"Failed to load plugin '{self._info.name}' from "
                f"'{self._info.full_path}': {e}"
            )
            raise


class PluginManager:
    """
    Manager for discovering and loading plugins via entry points.

    Provides lazy loading and caching of plugins.
    """

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        # Cache of discovered plugins: {group: {name: LazyPlugin}}
        self._plugins: dict[str, dict[str, LazyPlugin]] = {}
        # Track which groups have been discovered
        self._discovered_groups: set[str] = set()

    def discover_plugins(self, group: str) -> dict[str, LazyPlugin]:
        """
        Discover plugins from entry points for a given group.

        Uses cached results if already discovered.

        Args:
            group: Entry point group name (e.g., 'atp.adapters').

        Returns:
            Dictionary mapping plugin names to LazyPlugin instances.
        """
        if group in self._discovered_groups:
            return self._plugins.get(group, {})

        plugins: dict[str, LazyPlugin] = {}
        eps = entry_points(group=group)

        for ep in eps:
            try:
                info = self._extract_plugin_info(ep, group)
                plugins[ep.name] = LazyPlugin(ep, info)
                logger.debug(f"Discovered plugin '{ep.name}' in group '{group}'")
            except Exception as e:
                logger.warning(
                    f"Failed to process entry point '{ep.name}' in group '{group}': {e}"
                )

        self._plugins[group] = plugins
        self._discovered_groups.add(group)
        return plugins

    def _extract_plugin_info(self, ep: EntryPoint, group: str) -> PluginInfo:
        """Extract plugin metadata from entry point.

        Args:
            ep: Entry point to extract info from.
            group: The entry point group.

        Returns:
            PluginInfo with plugin metadata.
        """
        # Parse module and attr from entry point value
        module = ep.value.rsplit(":", 1)[0] if ":" in ep.value else ep.value
        attr = ep.value.rsplit(":", 1)[1] if ":" in ep.value else ""

        # Get package metadata if available
        version = None
        author = None
        description = None
        package = None

        # Try to get distribution info
        try:
            if hasattr(ep, "dist") and ep.dist is not None:
                dist = ep.dist
                package = dist.name
                version = dist.version
                # Try to get author from metadata
                metadata = dist.metadata
                author = metadata.get("Author") or metadata.get("Author-email")
                description = metadata.get("Summary")
        except Exception:
            # Distribution info not available
            pass

        return PluginInfo(
            name=ep.name,
            group=group,
            module=module,
            attr=attr,
            version=version,
            author=author,
            description=description,
            package=package,
        )

    def get_plugin(self, group: str, name: str) -> LazyPlugin | None:
        """Get a specific plugin by group and name.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            LazyPlugin if found, None otherwise.
        """
        self.discover_plugins(group)
        return self._plugins.get(group, {}).get(name)

    def load_plugin(self, group: str, name: str) -> type[Any] | None:
        """Load and return a plugin class.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            Plugin class if found and loaded successfully, None otherwise.
        """
        plugin = self.get_plugin(group, name)
        if plugin is None:
            return None

        try:
            return plugin.load()
        except Exception:
            return None

    def list_plugins(self, group: str) -> list[str]:
        """List all discovered plugin names for a group.

        Args:
            group: Entry point group.

        Returns:
            List of plugin names.
        """
        self.discover_plugins(group)
        return list(self._plugins.get(group, {}).keys())

    def list_plugin_info(self, group: str) -> list[PluginInfo]:
        """List plugin metadata for all plugins in a group.

        Args:
            group: Entry point group.

        Returns:
            List of PluginInfo for all discovered plugins.
        """
        self.discover_plugins(group)
        return [p.info for p in self._plugins.get(group, {}).values()]

    def discover_all(self) -> dict[str, dict[str, LazyPlugin]]:
        """Discover plugins from all known groups.

        Returns:
            Dictionary mapping group names to plugin dictionaries.
        """
        for group in ALL_GROUPS:
            self.discover_plugins(group)
        return self._plugins

    def clear_cache(self, group: str | None = None) -> None:
        """Clear the plugin cache.

        Args:
            group: Optional specific group to clear. If None, clears all.
        """
        if group is not None:
            self._plugins.pop(group, None)
            self._discovered_groups.discard(group)
        else:
            self._plugins.clear()
            self._discovered_groups.clear()

    def is_discovered(self, group: str) -> bool:
        """Check if a group has been discovered.

        Args:
            group: Entry point group.

        Returns:
            True if group has been discovered, False otherwise.
        """
        return group in self._discovered_groups

    def get_all_adapters(self) -> dict[str, LazyPlugin]:
        """Get all discovered adapters.

        Returns:
            Dictionary mapping adapter names to LazyPlugin instances.
        """
        return self.discover_plugins(ADAPTER_GROUP)

    def get_all_evaluators(self) -> dict[str, LazyPlugin]:
        """Get all discovered evaluators.

        Returns:
            Dictionary mapping evaluator names to LazyPlugin instances.
        """
        return self.discover_plugins(EVALUATOR_GROUP)

    def get_all_reporters(self) -> dict[str, LazyPlugin]:
        """Get all discovered reporters.

        Returns:
            Dictionary mapping reporter names to LazyPlugin instances.
        """
        return self.discover_plugins(REPORTER_GROUP)

    def _validate_plugin(
        self,
        plugin_class: type[Any],
        group: str,
        plugin_name: str | None = None,
    ) -> bool:
        """Validate that a plugin class implements the required interface.

        This method checks that the plugin class has all required attributes
        and methods for the given plugin group, and that the plugin's version
        requirements are met.

        Args:
            plugin_class: The plugin class to validate.
            group: Entry point group name (e.g., 'atp.adapters').
            plugin_name: Optional name for error messages.

        Returns:
            True if the plugin is valid.

        Raises:
            PluginValidationError: If the plugin fails interface validation.
            PluginVersionError: If the plugin has incompatible version requirements.
        """
        name = plugin_name or getattr(plugin_class, "__name__", str(plugin_class))

        # Validate interface compliance
        errors = validate_plugin(plugin_class, group, name)
        if errors:
            raise PluginValidationError(name, group, errors)

        # Check version compatibility
        current_version = atp.__version__
        version_error = check_version_compatibility(plugin_class, current_version, name)
        if version_error:
            required_version = getattr(plugin_class, "atp_version", "0.1.0")
            raise PluginVersionError(name, required_version, current_version)

        return True

    def load_and_validate_plugin(
        self,
        group: str,
        name: str,
        validate: bool = True,
        config: dict[str, Any] | None = None,
    ) -> type[Any] | None:
        """Load and optionally validate a plugin class.

        This method loads the plugin and validates it against the required
        interface for the plugin group.

        Args:
            group: Entry point group.
            name: Plugin name.
            validate: Whether to validate the plugin (default: True).
            config: Optional configuration data to validate against
                the plugin's config schema.

        Returns:
            Plugin class if found, loaded, and validated successfully.
            None if plugin not found.

        Raises:
            PluginValidationError: If validation is enabled and plugin fails.
            PluginVersionError: If plugin has incompatible version requirements.
            PluginConfigError: If config is provided and fails validation.
            Exception: If plugin fails to load.
        """
        plugin = self.get_plugin(group, name)
        if plugin is None:
            return None

        plugin_class = plugin.load()

        if validate:
            self._validate_plugin(plugin_class, group, name)

        # Validate configuration if provided
        if config is not None:
            self._validate_plugin_config(plugin_class, config, name)

        return plugin_class

    def _validate_plugin_config(
        self,
        plugin_class: type[Any],
        config_data: dict[str, Any],
        plugin_name: str | None = None,
    ) -> PluginConfig | None:
        """Validate configuration data against a plugin's config schema.

        Args:
            plugin_class: The plugin class to validate config for.
            config_data: Configuration data dictionary to validate.
            plugin_name: Optional plugin name for error messages.

        Returns:
            Validated PluginConfig instance if schema exists, None otherwise.

        Raises:
            PluginConfigError: If validation fails.
        """
        from atp.plugins.config import validate_plugin_config

        config_schema = get_plugin_config_schema(plugin_class)
        if config_schema is None:
            return None

        name = plugin_name or getattr(plugin_class, "__name__", str(plugin_class))
        return validate_plugin_config(config_schema, config_data, name)

    def get_plugin_config_schema(
        self,
        group: str,
        name: str,
    ) -> type[PluginConfig] | None:
        """Get the configuration schema for a plugin.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            The PluginConfig subclass if defined, None otherwise.
        """
        plugin = self.get_plugin(group, name)
        if plugin is None:
            return None

        try:
            plugin_class = plugin.load()
            return get_plugin_config_schema(plugin_class)
        except Exception:
            return None

    def get_plugin_config_metadata(
        self,
        group: str,
        name: str,
    ) -> ConfigMetadata | None:
        """Get configuration metadata for a plugin.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            ConfigMetadata if plugin has a config schema, None otherwise.
        """
        config_schema = self.get_plugin_config_schema(group, name)
        if config_schema is None:
            return None

        return ConfigMetadata.from_config_class(config_schema)

    def update_plugin_info_with_config(
        self,
        lazy_plugin: LazyPlugin,
    ) -> None:
        """Update plugin info with configuration metadata after loading.

        Args:
            lazy_plugin: The LazyPlugin to update.
        """
        if not lazy_plugin.is_loaded:
            return

        plugin_class = lazy_plugin._loaded
        if plugin_class is None:
            return

        config_schema = get_plugin_config_schema(plugin_class)
        if config_schema is not None:
            lazy_plugin._info.config_metadata = ConfigMetadata.from_config_class(
                config_schema
            )

    def discover_and_validate_plugins(
        self,
        group: str,
        validate: bool = True,
    ) -> dict[str, LazyPlugin]:
        """Discover plugins and optionally validate them when loaded.

        This method discovers plugins from entry points and can optionally
        validate each plugin as it is loaded.

        Args:
            group: Entry point group name (e.g., 'atp.adapters').
            validate: Whether to validate plugins when loaded.

        Returns:
            Dictionary mapping plugin names to LazyPlugin instances.
            Invalid plugins are excluded from the result.
        """
        all_plugins = self.discover_plugins(group)

        if not validate:
            return all_plugins

        valid_plugins: dict[str, LazyPlugin] = {}
        for name, lazy_plugin in all_plugins.items():
            try:
                plugin_class = lazy_plugin.load()
                self._validate_plugin(plugin_class, group, name)
                valid_plugins[name] = lazy_plugin
            except PluginValidationError as e:
                logger.warning(f"Plugin '{name}' failed validation: {e.errors}")
            except PluginVersionError as e:
                logger.warning(
                    f"Plugin '{name}' version incompatible: "
                    f"requires {e.required_version}, current is {e.current_version}"
                )
            except Exception as e:
                logger.warning(f"Failed to load plugin '{name}': {e}")

        return valid_plugins

    def get_validation_errors(
        self,
        group: str,
        name: str,
    ) -> tuple[list[str], str | None] | None:
        """Get validation errors for a specific plugin without raising.

        This is useful for diagnostics and error reporting.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            Tuple of (interface_errors, version_error) if plugin exists.
            None if plugin not found or couldn't be loaded.
        """
        plugin = self.get_plugin(group, name)
        if plugin is None:
            return None

        try:
            plugin_class = plugin.load()
        except Exception:
            return None

        errors = validate_plugin(plugin_class, group, name)
        version_error = check_version_compatibility(plugin_class, atp.__version__, name)

        return errors, version_error


# Global plugin manager instance
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance.

    Returns:
        Global PluginManager instance.
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
