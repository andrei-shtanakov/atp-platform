"""Tests for admin-aware JWT TTL selection.

All assertions about concrete minute values rely on the defaults
documented in ``atp.dashboard.auth`` (60 min regular / 720 min admin).
The ``_isolate_ttl_env`` fixture unsets the two env overrides so these
tests do not become environment-dependent when CI sets custom TTLs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    SECRET_KEY,
    access_token_ttl,
    create_access_token,
)


@pytest.fixture(autouse=True)
def _isolate_ttl_env(monkeypatch):
    """Remove any pre-set TTL env vars so the defaults apply.

    ``ACCESS_TOKEN_EXPIRE_MINUTES`` is read once at import time into a
    module-level constant — monkeypatching the env after import does
    not change it, so we also patch that constant directly to the
    documented 60-minute default for the duration of each test.
    """
    monkeypatch.delenv("ATP_TOKEN_EXPIRE_MINUTES", raising=False)
    monkeypatch.delenv("ATP_ADMIN_TOKEN_EXPIRE_MINUTES", raising=False)
    monkeypatch.setattr("atp.dashboard.auth.ACCESS_TOKEN_EXPIRE_MINUTES", 60)


def test_access_token_ttl_defaults_to_60_minutes_for_regular_users():
    assert access_token_ttl(is_admin=False) == timedelta(minutes=60)


def test_access_token_ttl_is_720_minutes_for_admins_by_default():
    assert access_token_ttl(is_admin=True) == timedelta(minutes=720)


def test_access_token_ttl_respects_admin_env_override(monkeypatch):
    monkeypatch.setenv("ATP_ADMIN_TOKEN_EXPIRE_MINUTES", "480")
    from atp.dashboard.auth import _read_admin_ttl

    assert _read_admin_ttl() == 480


def test_create_access_token_uses_admin_ttl_when_is_admin_true():
    token = create_access_token(
        data={"sub": "root", "user_id": 1},
        is_admin=True,
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
    remaining = expires_at - datetime.now(tz=UTC)
    # Wide bracket so the test is not flaky on slow CI.
    assert timedelta(minutes=700) < remaining < timedelta(minutes=730)


def test_create_access_token_uses_regular_ttl_when_is_admin_false():
    token = create_access_token(
        data={"sub": "alice", "user_id": 2},
        is_admin=False,
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
    remaining = expires_at - datetime.now(tz=UTC)
    assert (
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES - 5)
        < remaining
        < timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES + 5)
    )


def test_create_access_token_admin_exp_exceeds_regular_exp_by_10h():
    """Locks in the contract that admin TTL is meaningfully longer."""
    token_admin = create_access_token(
        data={"sub": "root", "user_id": 99}, is_admin=True
    )
    token_regular = create_access_token(
        data={"sub": "alice", "user_id": 100}, is_admin=False
    )
    exp_admin = jwt.decode(token_admin, SECRET_KEY, algorithms=[ALGORITHM])["exp"]
    exp_regular = jwt.decode(token_regular, SECRET_KEY, algorithms=[ALGORITHM])["exp"]
    assert exp_admin - exp_regular > 10 * 3600
