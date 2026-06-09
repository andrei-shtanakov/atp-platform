"""Registry for non-native suite formats dispatched by ``atp test``.

The ``test`` command natively loads the ATP ``TestSuite`` YAML. Other formats
(game suites today; plugin formats such as ``agent-eval-case`` tomorrow) are
routed here instead of a growing hardcoded chain of ``if _is_*`` branches.

A format registers a *detector* (does this file look like my format?) and a
*handler* (run it end-to-end, return success). The CLI asks the registry for a
handler before falling back to the native loader. Plugins register their formats
in their ``register()`` hook; the CLI registers the built-in game format.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

# Inspect a suite file and report whether it belongs to this format.
SuiteFormatDetector = Callable[[Path], bool]
# Run a matched suite end-to-end. Called with keyword arguments
# (suite_file, verbose, output_format, output_file) and returns success.
# A Coroutine (what `async def` returns) so the result is `asyncio.run`-able.
SuiteFormatHandler = Callable[..., Coroutine[Any, Any, bool]]


class SuiteFormatRegistry:
    """Ordered registry mapping a detected suite format to its handler."""

    def __init__(self) -> None:
        self._formats: list[tuple[str, SuiteFormatDetector, SuiteFormatHandler]] = []

    def register(
        self,
        name: str,
        detector: SuiteFormatDetector,
        handler: SuiteFormatHandler,
    ) -> None:
        """Register (or override by name) a suite format.

        Re-registering an existing name replaces it, letting a plugin override a
        built-in format. The new entry is appended (checked last).

        Args:
            name: Unique format identifier (e.g. ``"game_suite"``).
            detector: Returns True if a file belongs to this format.
            handler: Async callable that runs the suite and returns success.
        """
        self._formats = [(n, d, h) for (n, d, h) in self._formats if n != name]
        self._formats.append((name, detector, handler))

    def find_handler(self, suite_file: Path) -> SuiteFormatHandler | None:
        """Return the handler for the first format whose detector matches.

        A detector that raises is skipped (a malformed file is not this format).

        Args:
            suite_file: Path to the suite file under inspection.

        Returns:
            The matching handler, or None if no format claims the file.
        """
        for _name, detector, handler in self._formats:
            try:
                if detector(suite_file):
                    return handler
            except Exception:
                continue
        return None

    def names(self) -> list[str]:
        """Return the registered format names, in registration order."""
        return [name for (name, _d, _h) in self._formats]


_registry: SuiteFormatRegistry | None = None


def get_suite_format_registry() -> SuiteFormatRegistry:
    """Return the process-wide suite format registry (singleton)."""
    global _registry
    if _registry is None:
        _registry = SuiteFormatRegistry()
    return _registry
