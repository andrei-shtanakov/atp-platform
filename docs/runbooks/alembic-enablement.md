# Runbook: Enabling Alembic Migrations on Deploy

## Context

Historically the ATP platform container booted with
`init_database()` → `create_all()` + `_add_missing_columns()`, which
does *not* apply Alembic migrations. As the model evolved, prod
drifted from HEAD in ways those helpers can't fix (dropped unique
constraints, nullable → NOT NULL changes, partial indexes, CHECK
constraints). Each drift manifested eventually as a runtime 500.

This change adds `alembic upgrade head` to the container entrypoint so
future deploys apply pending migrations automatically. Before that can
be safe, the prod DB has to be brought to a state Alembic recognizes.

## One-time reconcile

Run once, on the VPS, while the container is running.

### 1. Back up the SQLite DB

```
ssh <VPS>
cd ~/atp-platform/deploy
docker compose exec platform \
  cp /data/atp.db /data/atp.db.pre-alembic-$(date -u +%Y%m%dT%H%M%SZ)
docker compose exec platform ls -la /data/
```

Keep at least one backup until the next successful deploy.

### 2. Run the reconcile script

The script is idempotent. It rebuilds drifted tables, then stamps the
`alembic_version` row at current HEAD so the entrypoint's
`alembic upgrade head` has a valid starting point.

```
docker compose exec platform \
  uv run --no-sync python /app/scripts/ops/reconcile_prod_schema.py
```

Expected output ends with `Reconcile complete.` A re-run should
report each check as `already ... — skip`.

### 3. Deploy the image with entrypoint

Merge the PR, push to `main`, let the deploy workflow rebuild. On
startup the container will print:

```
[entrypoint] Applying Alembic migrations (dashboard) ...
INFO  [alembic.runtime.migration] Running upgrade ... (no-op if stamped to HEAD)
[entrypoint] Migrations complete. Starting uvicorn.
```

If the migration step fails, the container will exit and the
healthcheck will mark it unhealthy. Investigate before rolling
forward — the fail-fast is intentional so drift is never silent.

## Adding a new migration

Standard Alembic flow from a dev machine:

```
uv run alembic -c alembic.ini -n dashboard revision \
  --autogenerate -m "<short description>"
# edit the generated file, run tests locally
uv run alembic -c alembic.ini -n dashboard upgrade head
git add migrations/dashboard/versions/... && git commit
```

No extra deploy step — the entrypoint applies it.

## Rollback

If a deploy applies a bad migration:

1. `docker compose exec platform cp /data/atp.db.pre-alembic-XXX /data/atp.db`
2. Revert the offending migration file on `main` and redeploy.
3. Run the reconcile script again if the migration changed the schema
   before failing.

## If the reconcile script itself fails

The script opens a single transaction and rolls back on any error, so
a half-applied state is not possible. If it aborts:

1. Capture the traceback and the current `PRAGMA table_info(...)`
   output for the affected tables.
2. Add a new branch in the reconcile script's `fix_*` helpers to
   handle the specific drift you observed.
3. Re-run.

Do not edit the `alembic_version` row manually. Let the script
stamp it, or let `alembic stamp` do it from inside the container.
