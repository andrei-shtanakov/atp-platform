"""Admin CLI for creating bot users and issuing long-lived JWTs.

Two subcommands:

- ``create-bot-user`` — create a new User row (non-admin, ``is_active=True``)
  and immediately issue a long-lived JWT for it.
- ``issue-token`` — find an existing user by username and issue a JWT.
  Works for bot users and for admin users alike; admin privileges are
  determined by ``user.is_admin`` in the database, not by any JWT claim.

Both commands print the JWT to ``stdout`` (so it can be captured with
``TOKEN=$(python -m atp.dashboard.cli.admin create-bot-user ... 2>/dev/null)``)
and send informational / diagnostic messages to ``stderr``.

Invocation (inside the production container)::

    docker compose exec platform uv run --no-sync \\
        python -m atp.dashboard.cli.admin create-bot-user --username bot_alice
    docker compose exec platform uv run --no-sync \\
        python -m atp.dashboard.cli.admin issue-token --username bot_alice
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import create_access_token, create_user
from atp.dashboard.database import Database
from atp.dashboard.models import User

_DEFAULT_TOKEN_DAYS = 30


def _open_database() -> Database:
    """Open a Database handle honoring ATP_DATABASE_URL.

    Bypasses ``init_database()`` because the CLI only needs query /
    write access — not the full startup sequence (create_tables,
    _add_missing_columns, _seed_default_roles, _backfill_run_user_id).
    Those side effects are already performed by the platform process
    on startup and duplicating them here fights with an already-running
    container.

    If ``ATP_DATABASE_URL`` is unset, ``Database()`` falls back to the
    historical default ``~/.atp/dashboard.db``. In a container where
    HOME=/app owned by root, that path is not writable by the runtime
    user; the caller must set the env var explicitly in that case.
    """
    url = os.environ.get("ATP_DATABASE_URL")
    return Database(url=url)


async def _get_user_by_username(session: AsyncSession, username: str) -> User | None:
    """Fetch a user by username. Returns None if not found."""
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def _create_bot_user_impl(
    db: Database, username: str, email: str | None, token_days: int
) -> str:
    """Create a bot user + issue its JWT. Returns the token string.

    Idempotent: if a user with the same username already exists, issues
    a fresh token for them instead of raising. This matches the operational
    use case where re-running the command should just rotate the token.
    """
    resolved_email = email or f"{username}@bot.local"
    async with db.session_factory() as session:
        existing = await _get_user_by_username(session, username)
        if existing is not None:
            print(
                f"User {username!r} already exists (id={existing.id}); "
                f"issuing a fresh token.",
                file=sys.stderr,
            )
            user_id = existing.id
        else:
            # Bots do not log in with a password; we set a random one so
            # the hashed_password column is never empty and a stolen row
            # can't be paired with a guessable credential.
            random_password = secrets.token_urlsafe(32)
            user = await create_user(
                session=session,
                username=username,
                email=resolved_email,
                password=random_password,
                is_admin=False,
            )
            await session.commit()
            print(
                f"Created bot user {username!r} (id={user.id}, "
                f"email={resolved_email!r}).",
                file=sys.stderr,
            )
            user_id = user.id

    # Both `user_id` and `sub` are required: the MCP middleware
    # (JWTUserStateMiddleware) reads `user_id`, while REST routes go
    # through `get_current_user` which requires `sub` (username) and
    # looks up the user by name. Tokens without `sub` are treated as
    # unauthenticated by REST handlers.
    return create_access_token(
        {"user_id": user_id, "sub": username},
        expires_delta=timedelta(days=token_days),
    )


async def _issue_token_impl(db: Database, username: str, token_days: int) -> str:
    """Issue a JWT for an existing user. Raises ValueError if not found."""
    async with db.session_factory() as session:
        user = await _get_user_by_username(session, username)
        if user is None:
            raise ValueError(
                f"User {username!r} not found. Use `create-bot-user` to "
                f"create a new bot user, or create an admin via the "
                f"GitHub OAuth flow first."
            )
        print(
            f"Issuing token for user {username!r} (id={user.id}, "
            f"is_admin={user.is_admin}).",
            file=sys.stderr,
        )
        # See note in `_create_bot_user_impl`: REST routes need `sub`.
        return create_access_token(
            {"user_id": user.id, "sub": user.username},
            expires_delta=timedelta(days=token_days),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m atp.dashboard.cli.admin",
        description="Admin CLI for bot-user creation and token issuance.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_bot = subparsers.add_parser(
        "create-bot-user",
        help="Create a new bot user and issue a long-lived JWT.",
    )
    create_bot.add_argument("--username", required=True)
    create_bot.add_argument(
        "--email",
        default=None,
        help="Optional email; defaults to <username>@bot.local.",
    )
    create_bot.add_argument(
        "--token-days",
        type=int,
        default=_DEFAULT_TOKEN_DAYS,
        help=f"Token lifetime in days (default: {_DEFAULT_TOKEN_DAYS}).",
    )

    issue = subparsers.add_parser(
        "issue-token",
        help="Issue a JWT for an existing user (bot OR admin).",
    )
    issue.add_argument("--username", required=True)
    issue.add_argument(
        "--token-days",
        type=int,
        default=_DEFAULT_TOKEN_DAYS,
        help=f"Token lifetime in days (default: {_DEFAULT_TOKEN_DAYS}).",
    )

    return parser


async def _async_main(args: argparse.Namespace) -> int:
    db = _open_database()
    try:
        if args.command == "create-bot-user":
            token = await _create_bot_user_impl(
                db,
                username=args.username,
                email=args.email,
                token_days=args.token_days,
            )
        elif args.command == "issue-token":
            token = await _issue_token_impl(
                db,
                username=args.username,
                token_days=args.token_days,
            )
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await db.close()

    print(token)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
