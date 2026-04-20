"""Tests for admin-aware JWT TTL selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

from atp.dashboard.auth import (
    ALGORITHM,
    SECRET_KEY,
    access_token_ttl,
    create_access_token,
)


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
    assert timedelta(minutes=55) < remaining < timedelta(minutes=65)


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
