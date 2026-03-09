"""Observer pattern for structured error and event recording.

Inspired by nullclaw's observer vtable. Replaces silent exception
swallowing with structured recording for later inspection.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObservedError:
    """A recorded error with context."""

    error: Exception
    context: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    """A recorded event."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Observer(Protocol):
    """Observer interface for recording errors and events."""

    def record_error(self, error: Exception, context: str = "") -> None: ...

    def record_event(self, name: str, data: dict[str, Any] | None = None) -> None: ...


class LoggingObserver:
    """Observer that logs errors and events via Python logging."""

    def __init__(self, logger_name: str = "atp.observer") -> None:
        self._logger = logging.getLogger(logger_name)

    def record_error(self, error: Exception, context: str = "") -> None:
        self._logger.debug(
            "Suppressed %s in %s: %s",
            type(error).__name__,
            context or "unknown",
            error,
        )

    def record_event(self, name: str, data: dict[str, Any] | None = None) -> None:
        self._logger.debug("Event: %s %s", name, data or {})


class ErrorCollector:
    """Observer that accumulates errors for later inspection."""

    def __init__(self) -> None:
        self._errors: list[ObservedError] = []
        self._events: list[ObservedEvent] = []

    def record_error(self, error: Exception, context: str = "") -> None:
        self._errors.append(ObservedError(error=error, context=context))

    def record_event(self, name: str, data: dict[str, Any] | None = None) -> None:
        self._events.append(ObservedEvent(name=name, data=data or {}))

    @property
    def errors(self) -> Sequence[ObservedError]:
        """All recorded errors."""
        return self._errors

    @property
    def events(self) -> Sequence[ObservedEvent]:
        """All recorded events."""
        return self._events

    @property
    def error_count(self) -> int:
        return len(self._errors)

    def clear(self) -> None:
        self._errors.clear()
        self._events.clear()

    def summary(self) -> dict[str, Any]:
        """Return a summary of recorded errors."""
        by_type: dict[str, int] = {}
        for e in self._errors:
            key = type(e.error).__name__
            by_type[key] = by_type.get(key, 0) + 1
        return {
            "total_errors": len(self._errors),
            "total_events": len(self._events),
            "errors_by_type": by_type,
        }


class CompositeObserver:
    """Observer that delegates to multiple backends."""

    def __init__(self, *observers: Observer) -> None:
        self._observers = list(observers)

    def record_error(self, error: Exception, context: str = "") -> None:
        for obs in self._observers:
            obs.record_error(error, context)

    def record_event(self, name: str, data: dict[str, Any] | None = None) -> None:
        for obs in self._observers:
            obs.record_event(name, data)


# Module-level default observer
_default_observer: Observer = LoggingObserver()


def get_observer() -> Observer:
    """Get the default observer."""
    return _default_observer


def set_observer(observer: Observer) -> None:
    """Set the default observer."""
    global _default_observer
    _default_observer = observer
