"""Helpers for working with database URLs at migration time.

Alembic runs sync, but production ATP uses async drivers
(sqlite+aiosqlite, postgresql+asyncpg). Engines created from the async
URL via sync `create_engine` crash with MissingGreenlet. Strip the
driver suffix so migrations and pre-migration probes can reuse the
same env var the app reads.
"""

from __future__ import annotations


def as_sync_url(url: str) -> str:
    """Return an equivalent URL that works with `sqlalchemy.create_engine`.

    Maps the known async drivers used by this project back to their
    sync counterparts. Other drivers (including unknown ones) pass
    through unchanged so the caller still sees the original error for
    genuinely unsupported URLs.
    """
    # sqlite async driver -> plain sqlite
    url = url.replace("+aiosqlite", "")
    # postgres async driver -> psycopg3 (sync)
    url = url.replace("+asyncpg", "+psycopg")
    return url
