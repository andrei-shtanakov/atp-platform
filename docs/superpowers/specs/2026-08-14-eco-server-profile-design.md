# Eco Server Profile — Design

**Date:** 2026-08-14
**Status:** Approved (design review 2026-08-14)
**Scope:** `packages/atp-dashboard` (runtime profile), root `pyproject.toml` (extras)

## 1. Goal

Provide a headless, ecosystem-oriented deployment profile of the ATP dashboard
server — **an API-only benchmark server** — installable and runnable without the
game/tournament stack (`game-environments`, `fastmcp`, `mcp`). The full server
(prod at atp.pr0sto.space) is unchanged and remains the default.

**Eco profile contract:**

> The `eco` profile serves OpenAPI (`/docs`, `/openapi.json`), health,
> auth/token/agent-management, and the Benchmark API. Composition-wise the
> profile is defined subtractively — **full minus tournaments/games/MCP/UI** —
> so every other non-game API router (suites, analytics, marketplace, roles,
> tenants, …) rides along; the guaranteed contract is the benchmark surface,
> the rest is not individually promised. HTML UI is **not part of
> the profile** — consumers are the CLI, the SDK (`atp-platform-sdk`), and
> external UIs (e.g. dispatcher). Tournament, game, and MCP surfaces are
> excluded as a **product boundary**, not merely as an import-breaking
> necessity: `games.py` and `winners_api.py` do not import `game_envs`, yet
> they are excluded because eco has no game-domain API at all.

## 2. Non-goals

- No fork, no new repo, no new package. One codebase, one app factory.
- No core-UI extraction. Eco is deliberately API-only; users lose not just the
  tournament UI but also login/register/benchmark HTML pages. If a
  "benchmark dashboard without games" is ever needed, that is a separate
  design (core UI split).
- No ORM/migration split. Tournament ORM models
  (`atp/dashboard/tournament/models.py`, plus `GameResult`/`TournamentResult`
  in `models.py`) do not import `game_envs`; shared metadata and Alembic stay
  as-is. Eco databases contain empty tournament tables — accepted trade-off
  for a minimal diff.
- No dispatcher work here. Dispatcher consumes the Benchmark API + SDK; after
  merge, file an inbox issue in `../dispatcher` per ADR-ECO-006.

## 3. Configuration

`DashboardConfig` (`atp/dashboard/v2/config.py`) gains:

```python
server_profile: Literal["full", "eco"] = Field(
    default="full",
    description="Server composition profile: full (default) or eco (API-only benchmark server)",
)
```

- Env: `ATP_SERVER_PROFILE` (existing `ATP_` prefix machinery).
- `Literal` gives early validation; unknown values fail at config parse.
- Included in `to_dict()` (`config.py:232`).
- Documented in `.env.example` and in the Environment Variables section of
  `CLAUDE.md`/docs.
- The chosen profile is logged at startup (one INFO line in `create_app`).
- No CLI flag in this iteration; `atp dashboard` reads env config as today.

## 4. Router aggregation split

`atp/dashboard/v2/routes/__init__.py` currently imports **all** routers
eagerly at module import, including `tournament_api` → `tournament/service.py`
→ `game_envs`. Restructure:

```python
def build_core_router() -> APIRouter: ...      # everything except the tournament group
def build_tournament_router() -> APIRouter: ...  # imports inside the function body
def build_router(*, include_tournaments: bool) -> APIRouter:
    router = build_core_router()
    if include_tournaments:
        router.include_router(build_tournament_router())
    return router
```

- Tournament router group: `tournament_api`, `tournament_live`, `games`,
  `el_farol_dashboard`, `builtins_api`, `winners_api`. Their imports move
  **inside** `build_tournament_router()`. Supporting tournament modules such
  as `el_farol_from_tournament` (not a standalone router — it is imported by
  `tournament_live.py` and, function-locally, by `ui.py`) remain reachable
  only through these lazy full-profile imports.
- **The module-level `router` global is removed.** Keeping
  `router = build_router(include_tournaments=True)` for compatibility would
  re-import games on `import atp.dashboard.v2.routes` and defeat the split.
  Known direct importers to migrate in the same PR:
  - `atp/dashboard/v2/__init__.py` (re-export — drop `router` from `__all__`)
  - `atp/dashboard/v2/factory.py:36`
  - `tests/unit/dashboard/test_api_detailed.py`
  - `tests/unit/dashboard/test_leaderboard_endpoint.py`
  - `tests/unit/dashboard/test_api.py`
- Clean API routers such as `agent_management_api.py`, importing only
  tournament ORM models (plain SQLAlchemy, no `game_envs`), remain in core
  untouched. UI routers are **not** core and are imported only inside the
  full-profile branch: `ui.py` imports `tournament.service` at module level
  (`ui.py:1063` — `SUPPORTED_GAMES`, a mid-file `# noqa: E402` import) and
  `admin_ui.py` imports `TournamentService` (`admin_ui.py:29`), so both
  transitively require `game_envs`.

## 5. Factory gating (`atp/dashboard/v2/factory.py`)

**Top-level import cleanup (required for a real eco import):**

- `factory.py:24` `from atp.dashboard.tournament.deadlines import
  run_deadline_worker` — moves into the full-profile branch.
- `factory.py:36` `from atp.dashboard.v2.routes import router as api_router` —
  replaced by `build_router(...)` called inside `create_app()`.

The gate must not be "conditional on the outside" while these module-level
imports keep crashing a minimal install at import time.

**In `create_app()`, profile `eco` skips:**

- MCP sub-apps: import of `atp.dashboard.mcp` (+ tools, auth middleware),
  mounts at `/mcp` and `/mcp-http`, their lifespan contexts.
- Deadline worker task in the combined lifespan.
- `assert_single_worker()` — only guards the deadline worker, so eco supports
  `WEB_CONCURRENCY > 1` (a deliberate benefit; full keeps the assert).
- Tournament router group (`build_router(include_tournaments=False)`).
- All UI: `ui`, `winners_ui`, `admin_ui` routers, Jinja2 templates, static
  mount `/static/v2`, legacy redirects (`/login`, `/register`, `/games`,
  `/analytics`), root redirect. Eco root `/` returns
  `{"profile": "eco", "docs": "/docs"}`.
- The tournament SSE bypass branch in `_is_sse_path` (the `/mcp` branch is
  also irrelevant in eco; the helper collapses accordingly).

**Full-profile guard (fail loud, don't mask):** at the full-composition
boundary, wrap the tournament imports:

```python
try:
    tournament_router = build_tournament_router()
    from atp.dashboard.mcp import ...
except ModuleNotFoundError as exc:
    if exc.name in {"game_envs", "fastmcp", "mcp"}:
        raise RuntimeError(
            "Server profile 'full' requires the tournament stack: "
            "install atp-dashboard[tournaments] or set ATP_SERVER_PROFILE=eco."
        ) from exc
    raise
```

Only the three known optional roots are translated; a `ModuleNotFoundError`
from anywhere else propagates untouched (no masking of internal bugs).

**Module-level default `app` (`factory.py:383`):** `app = create_app()` runs
at import time, and `atp/dashboard/v2/__init__.py` imports it. Consequence,
recorded as an accepted decision: **in a minimal install (no `[tournaments]`),
`import atp.dashboard.v2` requires `ATP_SERVER_PROFILE=eco`**; otherwise the
guard above raises. This is intentional — a full-profile launch without its
stack must fail loudly, even via the module-level convenience app. The wheel
smoke test (see §7) sets the env var accordingly.

## 6. Packaging (PR-2)

`packages/atp-dashboard/pyproject.toml`:

- Move from `dependencies` to a new extra `tournaments`:
  `game-environments>=1.0.0`, `fastmcp>=3.0`, `mcp>=1.28.1,<2` (the direct
  security floor+ceiling moves with fastmcp; verified: `fastmcp`/`mcp` are
  imported only under `atp/dashboard/mcp/`).
- `[tool.uv.sources] game-environments = { workspace = true }` stays; verify
  in the built wheel metadata that the source pin does not leak a runtime
  requirement outside the extra.

Root `pyproject.toml`:

```toml
dashboard = ["atp-dashboard[tournaments]>=1.0.0"]   # byte-compatible behavior
eco-server = ["atp-dashboard>=1.0.0"]
```

- `all` includes the full variant exactly once: the current three lines
  (`atp-dashboard>=1.0.0`, `atp-dashboard[enterprise]`,
  `atp-dashboard[analytics]`) collapse to
  `atp-dashboard[tournaments,enterprise,analytics]>=1.0.0` — no bare
  duplicate that would silently drop the games stack from `all`.
- `uv.lock` regenerated; CI green on both sync profiles.

## 7. Testing

**PR-1 (profile logic):**

- Config: default is `full`; `ATP_SERVER_PROFILE=eco` parses; unknown value
  rejected; `server_profile` present in `to_dict()`.
- Eco route contract (not a brittle full snapshot): required route patterns
  present — there is no bare `/api/v1/runs` root, so assert concrete patterns
  (`/api/v1/benchmarks`, `/api/v1/runs/{run_id}/next-task`,
  `/api/v1/runs/{run_id}/submit`, `/api/v1/tokens`, `/api/v1/agents`, auth);
  forbidden prefixes absent (`/mcp`, `/api/v1/tournaments`,
  `/api/v1/games`, `/ui`, `/static/v2`, `/login`, `/register`, `/games`,
  `/analytics`); `/docs` and `/openapi.json` respond 200.
- Full regression: route set before/after refactor compared via a fixture
  captured once in this PR (guards the router-split refactor itself).
- Import isolation, two levels:
  1. Unit-fast: `sys.modules` poisoning fixture (blocks `game_envs`,
     `fastmcp`, `mcp`) around eco app creation — with the caveat that pytest
     collection may have pre-imported modules;
  2. Authoritative: **subprocess test** — clean interpreter, blocked modules
     (`PYTHONPATH` shim or `-c` with import hooks), `ATP_SERVER_PROFILE=eco`,
     builds the app, asserts route contract.
- Worker policy: eco app boots with `WEB_CONCURRENCY=2`; full raises.
- Full-profile guard: with `game_envs` blocked and profile `full`, the
  `RuntimeError` names the extra and the env var.

**PR-2 (packaging proof):**

- Wheel-level smoke (CI job or slow-marked test): build the wheels, install
  minimal `atp-dashboard` (no `tournaments` extra) into a fresh venv, set
  `ATP_SERVER_PROFILE=eco`, create the app, assert the eco route contract and
  that `import game_envs` / `import fastmcp` fail. This — not the
  `sys.modules` trick — is the proof that optional deps did not leak through
  metadata or transitive imports.

## 8. Delivery plan

- **PR-1** — profile: config field + router split (incl. migration of the
  three tests and `v2/__init__` off the removed `router` global) + factory
  gating + loud full-guard + PR-1 tests.
- **PR-2** — packaging: extras move + root extras mapping + lockfile + wheel
  smoke test.
- **After merge** — inbox issue in `../dispatcher` (ADR-ECO-006): display of
  benchmark runs/leaderboard, consuming the Benchmark API/SDK.

Both PRs follow the repo git workflow: branch → PR → Copilot review → human
merge.
