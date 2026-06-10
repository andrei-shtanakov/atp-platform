"""Registry for alternate suite *source* formats that load into a TestSuite.

Distinct from ``format_dispatch`` (which runs an entirely different execution
path, e.g. game suites). A *source* format is just another way to author a
normal suite: it parses into a ``TestSuite`` and then runs through the standard
adapter / orchestrator / evaluator / reporter path unchanged. The agent-eval-case
plugin registers its case→TestSuite loader here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atp.loader.models import TestSuite

logger = logging.getLogger(__name__)

# Does this file belong to the source format?
SuiteSourceDetector = Callable[[Path], bool]
# Parse a matched file into a TestSuite.
SuiteSourceLoader = Callable[[Path], "TestSuite"]


class SuiteSourceRegistry:
    """Ordered registry mapping a detected source format to its loader."""

    def __init__(self) -> None:
        self._sources: list[tuple[str, SuiteSourceDetector, SuiteSourceLoader]] = []

    def register(
        self,
        name: str,
        detector: SuiteSourceDetector,
        loader: SuiteSourceLoader,
    ) -> None:
        """Register (or override by name) a source format."""
        self._sources = [(n, d, ldr) for (n, d, ldr) in self._sources if n != name]
        self._sources.append((name, detector, loader))

    def find_loader(self, suite_file: Path) -> SuiteSourceLoader | None:
        """Return the loader for the first source whose detector matches."""
        for name, detector, loader in self._sources:
            try:
                if detector(suite_file):
                    return loader
            except Exception:
                logger.debug(
                    "Suite source detector %r raised on %s; skipping",
                    name,
                    suite_file,
                    exc_info=True,
                )
                continue
        return None

    def names(self) -> list[str]:
        """Return the registered source format names, in registration order."""
        return [name for (name, _d, _ldr) in self._sources]


_registry: SuiteSourceRegistry | None = None


def get_suite_source_registry() -> SuiteSourceRegistry:
    """Return the process-wide suite source registry (singleton)."""
    global _registry
    if _registry is None:
        _registry = SuiteSourceRegistry()
    return _registry
