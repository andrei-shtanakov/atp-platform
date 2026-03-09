"""Plugin dependency resolution.

Resolves inter-plugin dependencies ensuring plugins are loaded
in correct order and all requirements are satisfied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginDependency:
    """A declared dependency on another plugin."""

    name: str
    group: str
    min_version: str | None = None


@dataclass
class DependencyNode:
    """Node in the dependency graph."""

    name: str
    group: str
    dependencies: list[PluginDependency] = field(default_factory=list)
    resolved: bool = False


class DependencyCycleError(Exception):
    """Raised when a circular dependency is detected."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        path = " -> ".join(cycle)
        super().__init__(f"Circular plugin dependency: {path}")


class MissingDependencyError(Exception):
    """Raised when a required dependency is not available."""

    def __init__(self, plugin: str, missing: str, group: str) -> None:
        self.plugin = plugin
        self.missing = missing
        self.group = group
        super().__init__(
            f"Plugin '{plugin}' requires '{missing}' "
            f"(group: {group}) which is not available"
        )


def extract_dependencies(plugin_class: type[Any]) -> list[PluginDependency]:
    """Extract declared dependencies from a plugin class.

    Plugins declare dependencies via a class attribute::

        class MyPlugin:
            plugin_dependencies = [
                {"name": "base_adapter", "group": "atp.adapters"},
                {"name": "llm_judge", "group": "atp.evaluators",
                 "min_version": "1.0.0"},
            ]

    Args:
        plugin_class: Plugin class to inspect.

    Returns:
        List of PluginDependency objects.
    """
    raw_deps = getattr(plugin_class, "plugin_dependencies", None)
    if not raw_deps:
        return []

    deps: list[PluginDependency] = []
    for dep in raw_deps:
        if isinstance(dep, dict):
            deps.append(
                PluginDependency(
                    name=dep.get("name", ""),
                    group=dep.get("group", ""),
                    min_version=dep.get("min_version"),
                )
            )
        elif isinstance(dep, PluginDependency):
            deps.append(dep)
    return deps


def resolve_load_order(
    plugins: dict[str, type[Any]],
    group: str,
) -> list[str]:
    """Resolve plugin load order based on dependencies.

    Uses topological sort to determine the correct order.

    Args:
        plugins: Mapping of plugin name to plugin class.
        group: Plugin group (e.g. "atp.evaluators").

    Returns:
        List of plugin names in dependency-resolved order.

    Raises:
        DependencyCycleError: If circular dependencies exist.
        MissingDependencyError: If a required dependency is missing.
    """
    # Build dependency graph
    nodes: dict[str, DependencyNode] = {}
    for name, cls in plugins.items():
        deps = extract_dependencies(cls)
        nodes[name] = DependencyNode(name=name, group=group, dependencies=deps)

    # Topological sort (Kahn's algorithm variant with cycle detection)
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, path: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle_start = path.index(name)
            raise DependencyCycleError(path[cycle_start:] + [name])

        visiting.add(name)
        path.append(name)

        node = nodes.get(name)
        if node:
            for dep in node.dependencies:
                # Only resolve deps within the same group
                if dep.group == group:
                    if dep.name not in nodes:
                        raise MissingDependencyError(name, dep.name, dep.group)
                    visit(dep.name, path.copy())

        visiting.discard(name)
        visited.add(name)
        order.append(name)

    for name in nodes:
        visit(name, [])

    return order


def check_dependencies_satisfied(
    plugin_name: str,
    plugin_class: type[Any],
    available: dict[str, set[str]],
) -> list[str]:
    """Check if all dependencies of a plugin are satisfied.

    Args:
        plugin_name: Name of the plugin being checked.
        plugin_class: Plugin class to inspect.
        available: Mapping of group -> set of available plugin names.

    Returns:
        List of error messages (empty if all deps satisfied).
    """
    deps = extract_dependencies(plugin_class)
    errors: list[str] = []

    for dep in deps:
        group_plugins = available.get(dep.group, set())
        if dep.name not in group_plugins:
            errors.append(f"Missing dependency '{dep.name}' (group: {dep.group})")

    return errors
