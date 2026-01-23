"""Tests for the startup module."""

import pytest

from atp.performance.startup import (
    DeferredRegistry,
    ImportAnalyzer,
    ImportTiming,
    LazyClass,
    LazyModule,
    get_lazy_adapter_registry,
    measure_startup_time,
)


class TestImportTiming:
    """Tests for ImportTiming."""

    def test_import_time_ms(self) -> None:
        """Test conversion to milliseconds."""
        timing = ImportTiming(
            module_name="test.module",
            import_time_seconds=0.05,
        )
        assert timing.import_time_ms == 50.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        timing = ImportTiming(
            module_name="test.module",
            import_time_seconds=0.1,
            parent_module="parent.module",
            dependencies=["dep1", "dep2"],
        )
        d = timing.to_dict()

        assert d["module_name"] == "test.module"
        assert d["import_time_ms"] == 100.0
        assert d["parent_module"] == "parent.module"
        assert d["dependencies"] == ["dep1", "dep2"]


class TestLazyModule:
    """Tests for LazyModule."""

    def test_is_loaded_initially_false(self) -> None:
        """Test that module is not loaded initially."""
        lazy = LazyModule("json")
        assert lazy.is_loaded is False

    def test_loads_on_attribute_access(self) -> None:
        """Test that module loads on first attribute access."""
        lazy = LazyModule("json")

        # Access an attribute
        _ = lazy.dumps

        assert lazy.is_loaded is True
        assert lazy.import_time is not None

    def test_module_functionality(self) -> None:
        """Test that lazy module works correctly."""
        lazy = LazyModule("json")

        # Use the module
        result = lazy.dumps({"key": "value"})

        assert result == '{"key": "value"}'

    def test_private_attribute_raises(self) -> None:
        """Test that private attribute access raises AttributeError."""
        lazy = LazyModule("json")

        with pytest.raises(AttributeError):
            _ = lazy._private


class TestLazyClass:
    """Tests for LazyClass."""

    def test_is_created_initially_false(self) -> None:
        """Test that instance is not created initially."""
        lazy = LazyClass(lambda: "instance")
        assert lazy.is_created is False

    def test_creates_on_get(self) -> None:
        """Test that instance is created on get."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        lazy = LazyClass(factory)

        # First get
        instance1 = lazy.get()
        assert instance1["count"] == 1
        assert lazy.is_created is True

        # Second get - should return same instance
        instance2 = lazy.get()
        assert instance2 is instance1
        assert call_count == 1  # Factory only called once

    def test_creation_time_tracked(self) -> None:
        """Test that creation time is tracked."""
        lazy = LazyClass(lambda: "value")

        assert lazy.creation_time is None

        lazy.get()

        assert lazy.creation_time is not None
        assert lazy.creation_time > 0


class TestDeferredRegistry:
    """Tests for DeferredRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and getting items."""
        registry: DeferredRegistry[str] = DeferredRegistry("TestRegistry")

        registry.register("item1", lambda: "value1")
        registry.register("item2", lambda: "value2")

        assert registry.get("item1") == "value1"
        assert registry.get("item2") == "value2"

    def test_deferred_creation(self) -> None:
        """Test that creation is deferred."""
        created = []

        def factory1():
            created.append("item1")
            return "value1"

        def factory2():
            created.append("item2")
            return "value2"

        registry: DeferredRegistry[str] = DeferredRegistry("TestRegistry")
        registry.register("item1", factory1)
        registry.register("item2", factory2)

        # Nothing created yet
        assert len(created) == 0

        # Get item1
        registry.get("item1")
        assert created == ["item1"]

        # Get item1 again - should not recreate
        registry.get("item1")
        assert created == ["item1"]

        # Get item2
        registry.get("item2")
        assert created == ["item1", "item2"]

    def test_unknown_key_raises(self) -> None:
        """Test that unknown key raises KeyError."""
        registry: DeferredRegistry[str] = DeferredRegistry("TestRegistry")

        with pytest.raises(KeyError, match="unknown"):
            registry.get("unknown")

    def test_is_created(self) -> None:
        """Test is_created method."""
        registry: DeferredRegistry[str] = DeferredRegistry("TestRegistry")
        registry.register("item", lambda: "value")

        assert registry.is_created("item") is False

        registry.get("item")

        assert registry.is_created("item") is True

    def test_list_keys(self) -> None:
        """Test listing registered keys."""
        registry: DeferredRegistry[str] = DeferredRegistry("TestRegistry")
        registry.register("a", lambda: "1")
        registry.register("b", lambda: "2")
        registry.register("c", lambda: "3")

        keys = registry.list_keys()
        assert sorted(keys) == ["a", "b", "c"]

    def test_get_creation_times(self) -> None:
        """Test getting creation times."""
        registry: DeferredRegistry[str] = DeferredRegistry("TestRegistry")
        registry.register("item1", lambda: "value1")
        registry.register("item2", lambda: "value2")

        registry.get("item1")

        times = registry.get_creation_times()
        assert "item1" in times
        assert "item2" not in times


class TestImportAnalyzer:
    """Tests for ImportAnalyzer."""

    def test_start_and_stop(self) -> None:
        """Test starting and stopping analysis."""
        analyzer = ImportAnalyzer()
        analyzer.start()

        # Import something
        import collections.abc  # noqa: F401

        timings = analyzer.stop()

        # Should have some timings (may or may not include collections.abc
        # depending on whether it was already imported)
        assert isinstance(timings, list)

    def test_format_report(self) -> None:
        """Test report formatting."""
        analyzer = ImportAnalyzer()
        analyzer._timings = [
            ImportTiming(module_name="test.module", import_time_seconds=0.05),
        ]

        report = analyzer.format_report()

        assert "Import Time Analysis" in report
        assert "test.module" in report


class TestMeasureStartupTime:
    """Tests for measure_startup_time."""

    def test_measure_startup_time(self) -> None:
        """Test measuring startup time."""
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return call_count

        result, time_taken = measure_startup_time(operation)

        assert result == 1
        assert time_taken > 0

    def test_measure_with_warmup(self) -> None:
        """Test measuring with warmup iterations."""
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return call_count

        result, _ = measure_startup_time(operation, warmup=3)

        assert result == 4  # 3 warmup + 1 measured
        assert call_count == 4

    def test_measure_with_iterations(self) -> None:
        """Test measuring with multiple iterations."""
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return call_count

        result, _ = measure_startup_time(operation, iterations=5)

        # Result should be from last iteration
        assert result == 5
        assert call_count == 5


class TestGetLazyAdapterRegistry:
    """Tests for get_lazy_adapter_registry."""

    def test_creates_registry(self) -> None:
        """Test that registry is created."""
        registry = get_lazy_adapter_registry()
        assert registry is not None

    def test_has_all_adapters(self) -> None:
        """Test that all adapter types are registered."""
        registry = get_lazy_adapter_registry()

        keys = registry.list_keys()
        assert "http" in keys
        assert "container" in keys
        assert "cli" in keys
        assert "langgraph" in keys
        assert "crewai" in keys
        assert "autogen" in keys

    def test_deferred_loading(self) -> None:
        """Test that adapters are loaded on demand."""
        registry = get_lazy_adapter_registry()

        # Nothing should be created yet
        for key in registry.list_keys():
            assert registry.is_created(key) is False

        # Get one adapter
        http_classes = registry.get("http")

        # Should be a tuple of (AdapterClass, ConfigClass)
        assert isinstance(http_classes, tuple)
        assert len(http_classes) == 2
        assert registry.is_created("http") is True

        # Others should still not be created
        assert registry.is_created("container") is False
