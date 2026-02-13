"""Registry of trace importers for different production systems."""

from __future__ import annotations

from atp.generator.trace_import import TraceImporter

_IMPORTERS: dict[str, type[TraceImporter]] = {}


def register_importer(
    name: str,
    cls: type[TraceImporter],
) -> None:
    """Register a trace importer class.

    Args:
        name: Short name for the importer (e.g. "langsmith").
        cls: TraceImporter subclass.
    """
    _IMPORTERS[name] = cls


def get_importer(name: str, **kwargs: object) -> TraceImporter:
    """Get an importer instance by name.

    Args:
        name: Importer name (e.g. "langsmith", "otel").
        **kwargs: Passed to the importer constructor.

    Returns:
        Instantiated TraceImporter.

    Raises:
        KeyError: If importer name is not registered.
    """
    if name not in _IMPORTERS:
        raise KeyError(f"Unknown importer: {name!r}. Available: {sorted(_IMPORTERS)}")
    return _IMPORTERS[name](**kwargs)  # type: ignore[arg-type]


def list_importers() -> list[str]:
    """List registered importer names."""
    return sorted(_IMPORTERS)


def _register_defaults() -> None:
    """Register built-in importers."""
    from atp.generator.importers.langsmith import (
        LangSmithImporter,
    )
    from atp.generator.importers.opentelemetry import (
        OpenTelemetryImporter,
    )

    register_importer("langsmith", LangSmithImporter)
    register_importer("otel", OpenTelemetryImporter)


_register_defaults()

__all__ = [
    "get_importer",
    "list_importers",
    "register_importer",
]
