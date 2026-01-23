"""Startup time optimization utilities for ATP.

Provides utilities for:
- Lazy loading of modules and adapters
- Import timing analysis
- Deferred initialization patterns
"""

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImportTiming:
    """Timing information for a module import."""

    module_name: str
    import_time_seconds: float
    parent_module: str | None = None
    dependencies: list[str] = field(default_factory=list)

    @property
    def import_time_ms(self) -> float:
        """Import time in milliseconds."""
        return self.import_time_seconds * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module_name": self.module_name,
            "import_time_seconds": self.import_time_seconds,
            "import_time_ms": self.import_time_ms,
            "parent_module": self.parent_module,
            "dependencies": self.dependencies,
        }


class LazyModule:
    """
    Lazy-loading module wrapper.

    Defers module import until first attribute access.
    """

    def __init__(self, module_name: str) -> None:
        """
        Initialize lazy module.

        Args:
            module_name: Full module name to import lazily.
        """
        self._module_name = module_name
        self._module: Any = None
        self._import_time: float | None = None

    def _load(self) -> Any:
        """Load the module if not already loaded."""
        if self._module is None:
            start = time.perf_counter()
            self._module = importlib.import_module(self._module_name)
            self._import_time = time.perf_counter() - start
            logger.debug(
                "Lazy loaded %s in %.3fms",
                self._module_name,
                self._import_time * 1000,
            )
        return self._module

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the loaded module."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._load(), name)

    @property
    def import_time(self) -> float | None:
        """Get import time (None if not yet loaded)."""
        return self._import_time

    @property
    def is_loaded(self) -> bool:
        """Check if module is loaded."""
        return self._module is not None


class LazyClass[T]:
    """
    Lazy-loading class wrapper.

    Defers class instantiation until first use.
    """

    def __init__(
        self,
        factory: Callable[[], T],
        name: str | None = None,
    ) -> None:
        """
        Initialize lazy class.

        Args:
            factory: Callable that creates the instance.
            name: Optional name for logging.
        """
        self._factory = factory
        self._name = name or factory.__name__
        self._instance: T | None = None
        self._creation_time: float | None = None

    def get(self) -> T:
        """Get or create the instance."""
        if self._instance is None:
            start = time.perf_counter()
            self._instance = self._factory()
            self._creation_time = time.perf_counter() - start
            logger.debug(
                "Lazy created %s in %.3fms",
                self._name,
                self._creation_time * 1000,
            )
        return self._instance

    @property
    def creation_time(self) -> float | None:
        """Get creation time (None if not yet created)."""
        return self._creation_time

    @property
    def is_created(self) -> bool:
        """Check if instance is created."""
        return self._instance is not None


class DeferredRegistry[T]:
    """
    Registry that defers item creation until needed.

    Items are registered with factories and only instantiated on first access.
    """

    def __init__(self, name: str = "DeferredRegistry") -> None:
        """
        Initialize deferred registry.

        Args:
            name: Registry name for logging.
        """
        self._name = name
        self._factories: dict[str, Callable[[], T]] = {}
        self._instances: dict[str, T] = {}
        self._creation_times: dict[str, float] = {}

    def register(self, key: str, factory: Callable[[], T]) -> None:
        """
        Register an item factory.

        Args:
            key: Unique key for the item.
            factory: Callable that creates the item.
        """
        self._factories[key] = factory

    def get(self, key: str) -> T:
        """
        Get or create an item by key.

        Args:
            key: Item key.

        Returns:
            The item instance.

        Raises:
            KeyError: If key is not registered.
        """
        if key in self._instances:
            return self._instances[key]

        if key not in self._factories:
            raise KeyError(f"Unknown key in {self._name}: {key}")

        start = time.perf_counter()
        instance = self._factories[key]()
        creation_time = time.perf_counter() - start

        self._instances[key] = instance
        self._creation_times[key] = creation_time

        logger.debug(
            "Deferred creation of %s[%s] in %.3fms",
            self._name,
            key,
            creation_time * 1000,
        )

        return instance

    def is_created(self, key: str) -> bool:
        """Check if an item has been created."""
        return key in self._instances

    def list_keys(self) -> list[str]:
        """List all registered keys."""
        return list(self._factories.keys())

    def get_creation_times(self) -> dict[str, float]:
        """Get creation times for all created items."""
        return dict(self._creation_times)


class ImportAnalyzer:
    """
    Analyzes module import times.

    Useful for identifying slow imports that affect startup time.
    """

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._timings: list[ImportTiming] = []
        self._original_import: Any = None

    def start(self) -> None:
        """Start analyzing imports."""
        import builtins

        self._original_import = builtins.__import__
        self._timings.clear()

        def timed_import(
            name: str,
            globals: dict | None = None,
            locals: dict | None = None,
            fromlist: tuple = (),
            level: int = 0,
        ):
            start = time.perf_counter()
            result = self._original_import(name, globals, locals, fromlist, level)
            elapsed = time.perf_counter() - start

            # Only track if import took meaningful time
            if elapsed > 0.001:  # > 1ms
                parent = globals.get("__name__") if globals else None
                self._timings.append(
                    ImportTiming(
                        module_name=name,
                        import_time_seconds=elapsed,
                        parent_module=parent,
                    )
                )

            return result

        builtins.__import__ = timed_import

    def stop(self) -> list[ImportTiming]:
        """
        Stop analyzing and return results.

        Returns:
            List of import timings, sorted by time descending.
        """
        import builtins

        if self._original_import:
            builtins.__import__ = self._original_import
            self._original_import = None

        return sorted(
            self._timings,
            key=lambda t: t.import_time_seconds,
            reverse=True,
        )

    def get_slow_imports(self, threshold_ms: float = 10.0) -> list[ImportTiming]:
        """
        Get imports slower than threshold.

        Args:
            threshold_ms: Threshold in milliseconds.

        Returns:
            List of slow imports.
        """
        threshold_s = threshold_ms / 1000
        return [t for t in self._timings if t.import_time_seconds > threshold_s]

    def format_report(self, top_n: int = 20) -> str:
        """
        Format a report of import timings.

        Args:
            top_n: Number of slowest imports to include.

        Returns:
            Formatted report string.
        """
        lines = [
            "Import Time Analysis",
            "=" * 50,
            "",
            f"{'Module':<40} {'Time':>10}",
            "-" * 50,
        ]

        sorted_timings = sorted(
            self._timings,
            key=lambda t: t.import_time_seconds,
            reverse=True,
        )

        for timing in sorted_timings[:top_n]:
            lines.append(f"{timing.module_name:<40} {timing.import_time_ms:>8.2f}ms")

        total = sum(t.import_time_seconds for t in self._timings)
        lines.append("-" * 50)
        lines.append(f"{'Total tracked imports':<40} {total * 1000:>8.2f}ms")
        lines.append(f"Number of imports tracked: {len(self._timings)}")

        return "\n".join(lines)


def measure_startup_time[T](
    func: Callable[[], T],
    warmup: int = 0,
    iterations: int = 1,
) -> tuple[T, float]:
    """
    Measure the startup/initialization time of a function.

    Args:
        func: Function to measure.
        warmup: Number of warmup iterations (not timed).
        iterations: Number of timed iterations (returns mean).

    Returns:
        Tuple of (result, mean_time_seconds).
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Measure
    times = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)

    return result, sum(times) / len(times)  # type: ignore[return-value]


# Pre-defined lazy modules for common heavy imports
lazy_httpx = LazyModule("httpx")
lazy_docker = LazyModule("docker")
lazy_anthropic = LazyModule("anthropic")


def get_lazy_adapter_registry():
    """
    Get a lazy-loading adapter registry.

    Returns a DeferredRegistry that creates adapters on-demand.
    """

    def http_factory():
        from atp.adapters.http import HTTPAdapter, HTTPAdapterConfig

        return (HTTPAdapter, HTTPAdapterConfig)

    def container_factory():
        from atp.adapters.container import ContainerAdapter, ContainerAdapterConfig

        return (ContainerAdapter, ContainerAdapterConfig)

    def cli_factory():
        from atp.adapters.cli import CLIAdapter, CLIAdapterConfig

        return (CLIAdapter, CLIAdapterConfig)

    def langgraph_factory():
        from atp.adapters.langgraph import LangGraphAdapter, LangGraphAdapterConfig

        return (LangGraphAdapter, LangGraphAdapterConfig)

    def crewai_factory():
        from atp.adapters.crewai import CrewAIAdapter, CrewAIAdapterConfig

        return (CrewAIAdapter, CrewAIAdapterConfig)

    def autogen_factory():
        from atp.adapters.autogen import AutoGenAdapter, AutoGenAdapterConfig

        return (AutoGenAdapter, AutoGenAdapterConfig)

    registry: DeferredRegistry[tuple[Any, Any]] = DeferredRegistry(
        "LazyAdapterRegistry"
    )
    registry.register("http", http_factory)
    registry.register("container", container_factory)
    registry.register("cli", cli_factory)
    registry.register("langgraph", langgraph_factory)
    registry.register("crewai", crewai_factory)
    registry.register("autogen", autogen_factory)

    return registry
