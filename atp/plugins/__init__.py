"""ATP Plugin system for discovering and loading plugins via entry points."""

from atp.plugins.config import (
    ConfigMetadata,
    PluginConfig,
    PluginConfigError,
    validate_plugin_config,
)
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
    MIN_ATP_VERSION,
    AdapterPlugin,
    EvaluatorPlugin,
    PluginProtocol,
    PluginValidationError,
    PluginVersionError,
    ReporterPlugin,
    check_version_compatibility,
    get_plugin_config_metadata,
    get_plugin_config_schema,
    get_plugin_protocol_for_group,
    get_required_attributes,
    get_required_methods,
    validate_plugin,
    validate_plugin_full,
)

__all__ = [
    # Config
    "ConfigMetadata",
    "PluginConfig",
    "PluginConfigError",
    "validate_plugin_config",
    # Discovery
    "ADAPTER_GROUP",
    "ALL_GROUPS",
    "EVALUATOR_GROUP",
    "REPORTER_GROUP",
    "LazyPlugin",
    "PluginInfo",
    "PluginManager",
    "get_plugin_manager",
    # Interfaces
    "MIN_ATP_VERSION",
    "AdapterPlugin",
    "EvaluatorPlugin",
    "PluginProtocol",
    "PluginValidationError",
    "PluginVersionError",
    "ReporterPlugin",
    "check_version_compatibility",
    "get_plugin_config_metadata",
    "get_plugin_config_schema",
    "get_plugin_protocol_for_group",
    "get_required_attributes",
    "get_required_methods",
    "validate_plugin",
    "validate_plugin_full",
]
