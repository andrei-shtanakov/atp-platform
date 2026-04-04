"""Unified transient auth state store.

Replaces per-module in-memory dicts (_sso_sessions, _saml_sessions)
with a single interface. Currently InMemory only (sufficient for
single-process deployments). For multi-worker setups, implement the
AuthStateStore protocol with a DB or Redis backend.

Used by SSO, SAML, and DeviceFlow routes to store callback/relay state.
"""

from __future__ import annotations

import time
from typing import Any, Protocol


class AuthStateStore(Protocol):
    """Protocol for transient auth state storage.

    Keys are opaque strings (state params, relay states, device codes).
    Values are arbitrary dicts. Entries expire after ttl_seconds.
    """

    async def put(self, key: str, data: dict[str, Any], ttl_seconds: int = 600) -> None:
        """Store data under key with a TTL."""
        ...

    async def get(self, key: str) -> dict[str, Any] | None:
        """Return data for key, or None if missing/expired."""
        ...

    async def pop(self, key: str) -> dict[str, Any] | None:
        """Return and remove data for key, or None if missing/expired."""
        ...


class InMemoryAuthStateStore:
    """In-memory implementation for dev/test/single-process deployments."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[dict[str, Any], float]] = {}

    async def put(self, key: str, data: dict[str, Any], ttl_seconds: int = 600) -> None:
        self._cleanup()
        self._entries[key] = (data, time.monotonic() + ttl_seconds)

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        data, expires_at = entry
        if time.monotonic() >= expires_at:
            del self._entries[key]
            return None
        return data

    async def pop(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        data, expires_at = entry
        if time.monotonic() >= expires_at:
            return None
        return data

    def _cleanup(self) -> None:
        """Remove expired entries (called lazily on put)."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._entries.items() if now >= exp]
        for k in expired:
            del self._entries[k]


# Module-level singleton for simple usage
_default_store: InMemoryAuthStateStore | None = None


def get_auth_state_store() -> InMemoryAuthStateStore:
    """Get the default auth state store singleton.

    Returns InMemoryAuthStateStore. Replace this function to use
    a DB-backed implementation in production multi-worker setups.
    """
    global _default_store
    if _default_store is None:
        _default_store = InMemoryAuthStateStore()
    return _default_store


def reset_auth_state_store() -> None:
    """Reset the singleton (for testing)."""
    global _default_store
    _default_store = None
