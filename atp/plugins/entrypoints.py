"""Load ATP plugins declared via the ``atp.plugins`` entry-point group.

Each entry point resolves to a zero-argument ``register()`` hook that wires the
plugin into core registries (evaluators, suite formats/sources, etc.). The CLI
calls :func:`load_entrypoint_plugins` once at startup. A plugin that fails to
import or register is logged and skipped — a broken optional plugin must not take
down the CLI.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

ATP_PLUGINS_GROUP = "atp.plugins"

_loaded = False


def load_entrypoint_plugins(*, force: bool = False) -> list[str]:
    """Discover and run ``atp.plugins`` register hooks (once per process).

    Args:
        force: Re-run even if already loaded (mainly for tests).

    Returns:
        Names of the plugins whose register hook ran successfully.
    """
    global _loaded
    if _loaded and not force:
        return []
    _loaded = True

    registered: list[str] = []
    for ep in entry_points(group=ATP_PLUGINS_GROUP):
        try:
            register = ep.load()
            register()
            registered.append(ep.name)
        except Exception:
            logger.warning(
                "Failed to load atp.plugins entry point %r; skipping",
                ep.name,
                exc_info=True,
            )
    return registered
