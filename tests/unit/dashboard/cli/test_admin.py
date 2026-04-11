"""Tests for the admin CLI: create-bot-user + issue-token."""

from __future__ import annotations

from datetime import timedelta

import jwt
import pytest
from sqlalchemy import select

from atp.dashboard.auth import ALGORITHM, SECRET_KEY, create_user
from atp.dashboard.cli.admin import (
    _create_bot_user_impl,
    _issue_token_impl,
    main,
)
from atp.dashboard.database import Database
from atp.dashboard.models import User


@pytest.fixture
async def db(tmp_path):
    """Create an isolated SQLite database for each test."""
    db_path = tmp_path / "admin_cli.db"
    db = Database(url=f"sqlite+aiosqlite:///{db_path}", echo=False)
    await db.create_tables()
    try:
        yield db
    finally:
        await db.close()


@pytest.mark.anyio
async def test_create_bot_user_creates_row_and_returns_token(db):
    token = await _create_bot_user_impl(
        db, username="bot_alice", email=None, token_days=30
    )

    # Token is a valid JWT carrying both user_id (for MCP middleware)
    # and sub (for REST get_current_user).
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert "user_id" in payload
    assert isinstance(payload["user_id"], int)
    assert payload["user_id"] >= 1
    assert payload["sub"] == "bot_alice"

    # User row exists with expected defaults
    async with db.session_factory() as s:
        row = (
            await s.execute(select(User).where(User.username == "bot_alice"))
        ).scalar_one()
        assert row.email == "bot_alice@bot.local"
        assert row.is_active is True
        assert row.is_admin is False
        assert row.id == payload["user_id"]


@pytest.mark.anyio
async def test_create_bot_user_uses_custom_email(db):
    token = await _create_bot_user_impl(
        db, username="bot_bob", email="bob@example.org", token_days=30
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    async with db.session_factory() as s:
        row = (
            await s.execute(select(User).where(User.username == "bot_bob"))
        ).scalar_one()
        assert row.email == "bob@example.org"
        assert row.id == payload["user_id"]


@pytest.mark.anyio
async def test_create_bot_user_idempotent_rotates_token(db):
    # First call creates the user.
    token1 = await _create_bot_user_impl(
        db, username="bot_charlie", email=None, token_days=30
    )
    # Second call finds existing user and issues a new token.
    token2 = await _create_bot_user_impl(
        db, username="bot_charlie", email=None, token_days=30
    )

    payload1 = jwt.decode(token1, SECRET_KEY, algorithms=[ALGORITHM])
    payload2 = jwt.decode(token2, SECRET_KEY, algorithms=[ALGORITHM])

    # Same user_id, but tokens are distinct JWT strings (different exp).
    assert payload1["user_id"] == payload2["user_id"]

    # Only one user row exists.
    async with db.session_factory() as s:
        rows = (
            (await s.execute(select(User).where(User.username == "bot_charlie")))
            .scalars()
            .all()
        )
        assert len(rows) == 1


@pytest.mark.anyio
async def test_issue_token_works_for_existing_user(db):
    # Seed an admin user manually.
    async with db.session_factory() as s:
        await create_user(
            session=s,
            username="andrei",
            email="andrei@example.com",
            password="secret",
            is_admin=True,
        )
        await s.commit()

    token = await _issue_token_impl(db, username="andrei", token_days=7)
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    # Token carries user_id of the admin, not the admin flag itself.
    async with db.session_factory() as s:
        user = (
            await s.execute(select(User).where(User.username == "andrei"))
        ).scalar_one()
    assert payload["user_id"] == user.id
    assert payload["sub"] == "andrei"

    # exp claim reflects the --token-days flag (7d from issuance).
    import datetime as dt

    exp = dt.datetime.fromtimestamp(payload["exp"], tz=dt.UTC)
    delta = exp - dt.datetime.now(tz=dt.UTC)
    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)


@pytest.mark.anyio
async def test_issue_token_raises_on_missing_user(db):
    with pytest.raises(ValueError, match="not found"):
        await _issue_token_impl(db, username="no_such_user", token_days=30)


def test_parser_requires_subcommand():
    """The CLI must refuse invocation without a subcommand."""
    with pytest.raises(SystemExit):
        main([])


def test_parser_rejects_unknown_subcommand():
    """Unknown subcommands must be rejected by argparse."""
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_parser_create_bot_user_requires_username():
    with pytest.raises(SystemExit):
        main(["create-bot-user"])


def test_parser_issue_token_requires_username():
    with pytest.raises(SystemExit):
        main(["issue-token"])
