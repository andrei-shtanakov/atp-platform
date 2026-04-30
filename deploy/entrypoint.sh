#!/bin/sh
# Docker entrypoint for the ATP platform container.
#
# Runs Alembic migrations against ATP_DATABASE_URL before handing off
# to uvicorn. Fails fast on migration error — that's safer than letting
# the app boot with schema drift and surface bugs at request time.
#
# Reconcile script: one-off helper for prod DBs that pre-date Alembic-
# on-deploy. It's an explicit tool, not auto-run: operators invoke it
# manually once after reading docs/runbooks/alembic-enablement.md.
#
# Flow:
#   1. alembic upgrade head   — applies any pending migrations
#   2. exec uvicorn ...       — hands off PID 1 so signals work

set -eu

echo "[entrypoint] Applying Alembic migrations (dashboard) ..."
uv run --no-sync alembic -c alembic.ini -n dashboard upgrade head
echo "[entrypoint] Migrations complete. Starting uvicorn."

exec "$@"
