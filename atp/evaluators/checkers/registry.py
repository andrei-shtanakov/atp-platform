"""Registry of named deterministic checkers (Phase A-1).

A checker is a pure function selected by ``grader: {type: programmatic,
checker: <name>}``. It maps a grader config + the agent's output text to a
uniform :class:`CaseVerdict`. A new capability registers a checker instead of
adding a ``grader.type`` enum value — the core dispatch stays closed.
"""

from collections.abc import Callable
from typing import Any

from atp.core.results import CaseVerdict

Checker = Callable[[dict[str, Any], str | None], CaseVerdict]

_CHECKERS: dict[str, Checker] = {}


def register_checker(name: str, fn: Checker) -> None:
    """Register a checker under ``name``. Raises on a duplicate name."""
    if name in _CHECKERS:
        raise ValueError(f"checker '{name}' already registered")
    _CHECKERS[name] = fn


def get_checker(name: str) -> Checker | None:
    """Return the checker registered under ``name``, or None if unknown."""
    return _CHECKERS.get(name)


def list_checkers() -> list[str]:
    """Return the sorted names of registered checkers."""
    return sorted(_CHECKERS)
