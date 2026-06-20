"""Request preparation hooks for per-run runtime setup."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from atp.loader.models import TestDefinition
from atp.protocol import ATPRequest


@dataclass(frozen=True)
class PreparedRequest:
    """A prepared request plus optional async cleanup."""

    request: ATPRequest
    cleanup: Callable[[], Awaitable[None]] | None = None


class RequestPreparer(Protocol):
    """Prepare an ATP request before adapter execution."""

    async def prepare(
        self, test: TestDefinition, request: ATPRequest
    ) -> PreparedRequest:
        """Return a prepared request and optional cleanup."""


_REQUEST_PREPARERS: dict[str, RequestPreparer] = {}


def register_request_preparer(name: str, preparer: RequestPreparer) -> None:
    """Register or replace a named request preparer."""
    _REQUEST_PREPARERS[name] = preparer


def unregister_request_preparer(name: str) -> None:
    """Remove a named request preparer if present."""
    _REQUEST_PREPARERS.pop(name, None)


def get_request_preparer(name: str) -> RequestPreparer | None:
    """Return a named preparer, or None."""
    return _REQUEST_PREPARERS.get(name)
