"""Tests for atp.dashboard.migrations.url_helpers."""

from __future__ import annotations

import pytest

from atp.dashboard.migrations.url_helpers import as_sync_url


@pytest.mark.parametrize(
    ("input_url", "expected"),
    [
        # sqlite async -> sync
        (
            "sqlite+aiosqlite:////data/atp.db",
            "sqlite:////data/atp.db",
        ),
        (
            "sqlite+aiosqlite:///:memory:",
            "sqlite:///:memory:",
        ),
        # postgres async -> sync psycopg
        (
            "postgresql+asyncpg://user:pw@host:5432/db",
            "postgresql+psycopg://user:pw@host:5432/db",
        ),
        # already-sync URLs pass through unchanged
        ("sqlite:////data/atp.db", "sqlite:////data/atp.db"),
        (
            "postgresql+psycopg://u:p@h/d",
            "postgresql+psycopg://u:p@h/d",
        ),
        ("postgresql://u:p@h/d", "postgresql://u:p@h/d"),
        # unknown drivers pass through so the caller sees the real error
        ("mysql+aiomysql://u:p@h/d", "mysql+aiomysql://u:p@h/d"),
    ],
)
def test_as_sync_url_normalization(input_url: str, expected: str) -> None:
    assert as_sync_url(input_url) == expected
