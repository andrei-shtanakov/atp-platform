"""Result type for structured error handling.

Provides Success/Failure discriminated union to replace bare
except blocks with explicit error handling.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Success[T]:
    """Successful result containing a value."""

    value: T

    @property
    def is_success(self) -> bool:
        return True

    @property
    def is_failure(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Failure:
    """Failed result containing error information."""

    error: str
    code: str | None = None
    cause: Exception | None = field(default=None, repr=False)

    @property
    def is_success(self) -> bool:
        return False

    @property
    def is_failure(self) -> bool:
        return True


type Result[T] = Success[T] | Failure
"""Discriminated union: either Success[T] or Failure."""
