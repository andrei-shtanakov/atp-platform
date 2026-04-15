# ATP Deploy Notes

Prod lives on a Namecheap VPS. See `.github/workflows/deploy.yml` for the
trigger semantics (push to `main` or manual `workflow_dispatch`).

## Known host constraints

### Namecheap VPS: pre-Nehalem QEMU CPU (no x86_v2 wheels)

`/proc/cpuinfo` reports `QEMU Virtual CPU version 2.5+` — the default
qemu64 model. CPU flags lack SSSE3, SSE4.1, SSE4.2, and POPCNT, so any
Python wheel built targeting the x86_v2 baseline will fail at import
with `Illegal instruction (core dumped)` or a numpy crash.

Current workaround: `numpy>=1.26,<2.0` is pinned in the root
`pyproject.toml`. Do not relax this pin without also verifying the
target host supports x86_v2. If we migrate to a provider with modern
virtualization (host-passthrough or Nehalem+), the pin can be lifted.

Before upgrading numpy or adding any other x86_v2-baseline wheel
(polars, torch, some SIMD-heavy libs), confirm the destination host's
CPU flags with `cat /proc/cpuinfo | grep flags`.

### Deadline worker requires single uvicorn worker

`atp.dashboard.v2.factory:app` runs the tournament deadline worker in
its lifespan. Multiple workers would each run their own deadline loop
and race on `force_resolve_round`. `docker-compose.yml` enforces
`--workers 1`, and `deploy/Dockerfile` CMD keeps the same default for
direct `docker run`.

Lifting this constraint is backlog item I in the MCP Tournament design
(Redis bus + Postgres migration required first).

## Database URL env var

The application reads `ATP_DATABASE_URL`. Alembic migrations now honor
the same variable (preferred), with `ATP_DASHBOARD_DATABASE_URL` as an
explicit override when you need to migrate a different DB than the one
the app is running against.

The URL may use an async driver (`sqlite+aiosqlite://…`,
`postgresql+asyncpg://…`). Migration tooling strips the async suffix
internally so you don't have to maintain two variants.

## Healthcheck

Docker healthcheck uses stdlib `urllib.request`, not `httpx` — the
runtime image's system `python` does not have `httpx` installed (only
the `/app/.venv/bin/python` does). Do not "fix" the healthcheck back to
httpx without also switching the command to use the venv binary.
