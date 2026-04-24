#!/bin/bash
# Run on VPS from ~/atp-platform/deploy to recover a prod that got
# deployed with alembic-on-entrypoint BEFORE the one-off reconcile
# was run. Bypasses the entrypoint via ``--entrypoint ""`` so alembic
# doesn't get a chance to fail again.

set -eu

cd "$(dirname "$0")/../../deploy"

echo "==> Backup /data/atp.db"
docker compose run --rm --no-deps --entrypoint "" platform \
    sh -c 'cp /data/atp.db "/data/atp.db.pre-alembic-$(date -u +%Y%m%dT%H%M%SZ)"; ls -la /data/'

echo "==> Run reconcile (rebuilds drifted tables + stamps alembic_version)"
docker compose run --rm --no-deps --entrypoint "" platform \
    uv run --no-sync python /app/scripts/ops/reconcile_prod_schema.py

echo "==> Restart platform container"
docker compose up -d platform

echo "==> Tail platform logs (Ctrl-C when you see 'Application startup complete')"
docker compose logs -f --tail=50 platform
