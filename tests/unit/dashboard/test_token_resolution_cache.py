"""Tests for the in-process API-token resolution cache.

Covers ``_token_auth_cache`` in
``packages/atp-dashboard/atp/dashboard/v2/rate_limit.py``: a TTL-bounded
LRU that lets ``JWTUserStateMiddleware._resolve_api_token`` skip the
``api_tokens`` SELECT when the same token has been resolved within the
past 30 seconds.

Most tests call ``_resolve_api_token`` directly (it is a static method)
rather than going through the middleware to keep cache-mechanic
assertions tight. The end-to-end auth chain has its own coverage in
``tests/integration/dashboard/test_tsa_auth_gating.py``.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import User
from atp.dashboard.tokens import APIToken
from atp.dashboard.v2.rate_limit import (
    JWTUserStateMiddleware,
    _token_auth_cache,
)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _empty_state() -> dict[str, Any]:
    """Mimic the state dict that ``JWTUserStateMiddleware.__call__`` seeds
    before invoking ``_resolve_api_token``."""
    return {
        "user_id": None,
        "agent_id": None,
        "token_type": None,
        "agent_purpose": None,
    }


@pytest.fixture
async def configured_db(tmp_path: Path) -> AsyncIterator[Database]:
    """Spin up a per-test SQLite ``Database`` and register it globally.

    ``_resolve_api_token`` calls ``get_database()`` to obtain its session
    factory; tests need the module-level singleton wired to a real
    backing store. Uses a file-based SQLite in ``tmp_path`` so multiple
    sessions opened by the implementation see the same rows we seed
    from the test (in-memory SQLite is per-connection and would hide
    inserts).
    """
    db = Database(url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await db.create_tables()
    set_database(db)
    try:
        yield db
    finally:
        await db.close()
        set_database(None)  # type: ignore[arg-type]


async def _seed_user_and_token(
    db: Database,
    *,
    user_id: int,
    token: str,
    agent_id: int | None = None,
    agent_purpose: str | None = None,
    revoked: bool = False,
    expires_at: datetime | None = None,
) -> None:
    """Insert a User + APIToken pair so ``_resolve_api_token`` can find it."""
    async with db.session() as session:
        session.add(
            User(
                id=user_id,
                username=f"u{user_id}",
                email=f"u{user_id}@test.local",
                hashed_password="x",
            )
        )
        session.add(
            APIToken(
                user_id=user_id,
                agent_id=agent_id,
                agent_purpose=agent_purpose,
                name=f"tok-{user_id}",
                token_prefix=token[:12],
                token_hash=_hash(token),
                expires_at=expires_at,
                revoked_at=datetime.now() if revoked else None,
            )
        )
        # ``db.session()`` commits on exit; rely on that.


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cache_hit_skips_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-populated cache entry resolves the token without any DB
    access. We monkeypatch ``get_database`` to fail loudly so a
    regression that introduces a DB call cannot silently pass on a
    test that happens to inherit a configured global ``Database``
    from prior test order."""
    db_calls: list[str] = []

    def _fail_db() -> Any:
        db_calls.append("get_database called on cache hit")
        raise RuntimeError("DB must not be touched on cache hit")

    monkeypatch.setattr("atp.dashboard.database.get_database", _fail_db)

    token = "atp_a_cache_hit_only"
    _token_auth_cache[_hash(token)] = (42, 7, "tournament", None)

    state = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state, token)

    assert db_calls == [], "_resolve_api_token must not touch DB on cache hit"
    assert state["user_id"] == 42
    assert state["agent_id"] == 7
    assert state["agent_purpose"] == "tournament"
    assert state["token_type"] == "api"


@pytest.mark.anyio
async def test_cache_hit_for_user_token_with_no_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-level tokens (agent_id=None, agent_purpose=None) round-trip
    through the cache correctly without DB access."""

    def _fail_db() -> Any:
        raise RuntimeError("DB must not be touched on cache hit")

    monkeypatch.setattr("atp.dashboard.database.get_database", _fail_db)

    token = "atp_u_user_level"
    _token_auth_cache[_hash(token)] = (101, None, None, None)

    state = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state, token)

    assert state["user_id"] == 101
    assert state["agent_id"] is None
    assert state["agent_purpose"] is None
    assert state["token_type"] == "api"


@pytest.mark.anyio
async def test_cached_entry_past_expires_at_is_evicted(
    configured_db: Database,
) -> None:
    """A cache entry whose ``expires_at`` already passed must be
    evicted on hit and force a fresh DB lookup. Without this guard,
    a token that expires within the TTL window would keep resolving
    for up to ~30 s past actual expiry — see Copilot review on PR #99.
    """
    token = "atp_a_cached_then_expired"
    # Pre-seed the cache with an entry whose expires_at is in the past
    # but is otherwise plausible. The DB row does not exist (token was
    # not inserted), so re-resolution must yield empty state.
    _token_auth_cache[_hash(token)] = (
        7,
        99,
        "tournament",
        datetime.now() - timedelta(seconds=1),
    )

    state = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state, token)

    # Cache evicted, DB had no matching row → state stays empty.
    assert state["user_id"] is None
    assert _token_auth_cache.get(_hash(token)) is None


# ---------------------------------------------------------------------------
# Cache miss → DB → populate
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cache_miss_falls_through_to_db_and_populates(
    configured_db: Database,
) -> None:
    """First call with empty cache hits the DB; the resolved triple is
    written back so the next call can short-circuit."""
    token = "atp_a_first_call"
    await _seed_user_and_token(
        configured_db,
        user_id=11,
        token=token,
        agent_id=3,
        agent_purpose="tournament",
    )

    state = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state, token)

    assert state["user_id"] == 11
    assert state["agent_id"] == 3
    assert state["agent_purpose"] == "tournament"
    assert state["token_type"] == "api"

    cached = _token_auth_cache.get(_hash(token))
    # Tokens seeded without expires_at → cached as None.
    assert cached == (11, 3, "tournament", None)


@pytest.mark.anyio
async def test_cache_miss_for_revoked_token_does_not_populate(
    configured_db: Database,
) -> None:
    """Revoked tokens must not enter the cache — otherwise an admin
    revoke followed by an immediate request would still cache the
    pre-revoke triple. The SELECT WHERE clause filters revoked rows;
    asserting cache emptiness here pins that contract."""
    token = "atp_a_revoked"
    await _seed_user_and_token(
        configured_db,
        user_id=12,
        token=token,
        agent_id=4,
        agent_purpose="tournament",
        revoked=True,
    )

    state = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state, token)

    assert state["user_id"] is None
    assert _token_auth_cache.get(_hash(token)) is None


@pytest.mark.anyio
async def test_cache_miss_for_expired_token_does_not_populate(
    configured_db: Database,
) -> None:
    """Expired tokens fall through the early-return inside the DB block
    *before* the cache is populated."""
    token = "atp_a_expired"
    await _seed_user_and_token(
        configured_db,
        user_id=13,
        token=token,
        agent_id=5,
        agent_purpose="tournament",
        expires_at=datetime.now() - timedelta(hours=1),
    )

    state = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state, token)

    assert state["user_id"] is None
    assert _token_auth_cache.get(_hash(token)) is None


# ---------------------------------------------------------------------------
# Subsequent hit confirms cache is doing its job
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_second_call_uses_cache(configured_db: Database) -> None:
    """After a successful resolution, a second call resolves from the
    cache. We prove this by deleting the row between calls — if the
    second call still SELECTed, it would miss and leave state empty."""
    token = "atp_a_second_call"
    await _seed_user_and_token(
        configured_db,
        user_id=21,
        token=token,
        agent_id=8,
        agent_purpose="tournament",
    )

    first = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(first, token)
    assert first["user_id"] == 21

    # Delete the row directly; cache should now serve the answer.
    from sqlalchemy import delete

    async with configured_db.session() as session:
        await session.execute(
            delete(APIToken).where(APIToken.token_hash == _hash(token))
        )

    second = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(second, token)
    assert second["user_id"] == 21  # served from cache despite DB row gone
    assert second["agent_id"] == 8
    assert second["agent_purpose"] == "tournament"


# ---------------------------------------------------------------------------
# Eviction / TTL
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_clear_invalidates_subsequent_calls(
    configured_db: Database,
) -> None:
    """Manually clearing the cache (the operation tests use for
    isolation) forces the next call to hit the DB again — used by
    integration tests to exercise the lookup path after revocation."""
    token = "atp_a_post_clear"
    await _seed_user_and_token(
        configured_db,
        user_id=31,
        token=token,
        agent_id=9,
        agent_purpose="tournament",
    )

    state1 = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state1, token)
    assert _token_auth_cache.get(_hash(token)) is not None

    _token_auth_cache.clear()

    # Re-resolution must work because the row is still in the DB.
    state2 = _empty_state()
    await JWTUserStateMiddleware._resolve_api_token(state2, token)
    assert state2["user_id"] == 31
    assert _token_auth_cache.get(_hash(token)) == (31, 9, "tournament", None)
