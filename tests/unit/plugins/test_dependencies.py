"""Tests for plugin dependency resolution."""

import pytest

from atp.plugins.dependencies import (
    DependencyCycleError,
    MissingDependencyError,
    PluginDependency,
    check_dependencies_satisfied,
    extract_dependencies,
    resolve_load_order,
)


class _PluginNoDeps:
    pass


class _PluginWithDeps:
    plugin_dependencies = [
        {"name": "base", "group": "atp.evaluators"},
    ]


class _PluginWithVersionDep:
    plugin_dependencies = [
        {"name": "llm_judge", "group": "atp.evaluators", "min_version": "1.0.0"},
    ]


class TestExtractDependencies:
    def test_no_deps(self) -> None:
        assert extract_dependencies(_PluginNoDeps) == []

    def test_dict_deps(self) -> None:
        deps = extract_dependencies(_PluginWithDeps)
        assert len(deps) == 1
        assert deps[0].name == "base"
        assert deps[0].group == "atp.evaluators"

    def test_version_dep(self) -> None:
        deps = extract_dependencies(_PluginWithVersionDep)
        assert deps[0].min_version == "1.0.0"

    def test_dataclass_deps(self) -> None:
        class _Plugin:
            plugin_dependencies = [
                PluginDependency(name="x", group="atp.adapters"),
            ]

        deps = extract_dependencies(_Plugin)
        assert len(deps) == 1
        assert deps[0].name == "x"


class TestResolveLoadOrder:
    def test_no_dependencies(self) -> None:
        plugins = {"a": _PluginNoDeps, "b": _PluginNoDeps}
        order = resolve_load_order(plugins, "atp.evaluators")
        assert set(order) == {"a", "b"}

    def test_dependency_order(self) -> None:
        class _Dependent:
            plugin_dependencies = [
                {"name": "base", "group": "atp.evaluators"},
            ]

        plugins = {"base": _PluginNoDeps, "dependent": _Dependent}
        order = resolve_load_order(plugins, "atp.evaluators")
        assert order.index("base") < order.index("dependent")

    def test_cycle_detection(self) -> None:
        class _A:
            plugin_dependencies = [
                {"name": "b", "group": "g"},
            ]

        class _B:
            plugin_dependencies = [
                {"name": "a", "group": "g"},
            ]

        with pytest.raises(DependencyCycleError, match="Circular"):
            resolve_load_order({"a": _A, "b": _B}, "g")

    def test_missing_dependency(self) -> None:
        class _NeedsMissing:
            plugin_dependencies = [
                {"name": "nonexistent", "group": "g"},
            ]

        with pytest.raises(MissingDependencyError, match="nonexistent"):
            resolve_load_order({"a": _NeedsMissing}, "g")

    def test_cross_group_deps_ignored(self) -> None:
        class _CrossGroup:
            plugin_dependencies = [
                {"name": "external", "group": "other.group"},
            ]

        # Should not raise — cross-group deps are not resolved here
        order = resolve_load_order({"a": _CrossGroup}, "atp.evaluators")
        assert order == ["a"]


class TestCheckDependenciesSatisfied:
    def test_all_satisfied(self) -> None:
        available = {"atp.evaluators": {"base", "llm_judge"}}
        errors = check_dependencies_satisfied("my_plugin", _PluginWithDeps, available)
        assert errors == []

    def test_missing(self) -> None:
        available = {"atp.evaluators": set()}
        errors = check_dependencies_satisfied("my_plugin", _PluginWithDeps, available)
        assert len(errors) == 1
        assert "base" in errors[0]

    def test_no_deps(self) -> None:
        errors = check_dependencies_satisfied("my_plugin", _PluginNoDeps, {})
        assert errors == []
