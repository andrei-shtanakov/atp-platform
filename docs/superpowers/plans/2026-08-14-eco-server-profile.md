# Eco Server Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an API-only `eco` server profile to atp-dashboard that boots and installs without the game/tournament stack, keeping the `full` profile byte-compatible.

**Architecture:** A `server_profile` config field gates composition in `v2/factory.py`; router aggregation splits into `build_core_router()` / `build_tournament_router()` / `build_router(include_tournaments=...)` with tournament imports lazy; PR-2 moves `game-environments`/`fastmcp`/`mcp` into an `atp-dashboard[tournaments]` extra with root-extra compatibility mapping.

**Tech Stack:** FastAPI, pydantic-settings, pytest + anyio, uv workspaces, hatchling.

**Spec:** `docs/superpowers/specs/2026-08-14-eco-server-profile-design.md` (read it first — it records the product boundary and every reviewed decision).

## Global Constraints

- Package management: **uv only** (`uv run`, `uv add`); never pip except inside the PR-2 smoke venv.
- After every change: `uv run ruff format .`, `uv run ruff check .`, `uv run pyrefly check` — fix before moving on.
- Line length 88; type hints everywhere; async tests use anyio.
- Git: branch → PR; **no direct commits to `main`**; merge is done by the human.
- Default profile is `full`: existing behavior (prod, tests) must be unchanged.
- Eco contract (spec §1): OpenAPI + health + auth/token/agent-management + Benchmark API; **no** HTML UI, **no** `/mcp*`, **no** tournament/game routes.
- PR-1 must remove **both** top-level tournament imports in `factory.py` (`run_deadline_worker` at `:24`, aggregated `router` at `:36`) and the module-level `router` global in `routes/__init__.py`.

---

## PR-1 — profile logic (branch `feat/eco-server-profile`)

### Task 1: Branch + spec commit

**Files:**
- Commit: `docs/superpowers/specs/2026-08-14-eco-server-profile-design.md`, `docs/superpowers/plans/2026-08-14-eco-server-profile.md`

- [ ] **Step 1: Create branch**

```bash
git switch main && git pull --ff-only
git switch -c feat/eco-server-profile
```

- [ ] **Step 2: Commit spec + plan**

```bash
git add docs/superpowers/specs/2026-08-14-eco-server-profile-design.md docs/superpowers/plans/2026-08-14-eco-server-profile.md
git commit -m "docs: eco server profile design + plan"
```

### Task 2: `server_profile` config field

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py` (fields near `debug`/`disable_auth`, `to_dict()` at ~line 232)
- Test: `tests/unit/dashboard/test_server_profile_config.py` (create)

**Interfaces:**
- Produces: `DashboardConfig.server_profile: Literal["full", "eco"]` (env `ATP_SERVER_PROFILE`, default `"full"`), present in `to_dict()`.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the ATP_SERVER_PROFILE config field."""

import pytest
from pydantic import ValidationError

from atp.dashboard.v2.config import DashboardConfig


def test_default_profile_is_full() -> None:
    config = DashboardConfig(secret_key="x")
    assert config.server_profile == "full"


def test_env_sets_eco(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATP_SERVER_PROFILE", "eco")
    config = DashboardConfig(secret_key="x")
    assert config.server_profile == "eco"


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValidationError):
        DashboardConfig(secret_key="x", server_profile="turbo")


def test_profile_in_to_dict() -> None:
    config = DashboardConfig(secret_key="x", server_profile="eco")
    assert config.to_dict()["server_profile"] == "eco"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/dashboard/test_server_profile_config.py -v`
Expected: FAIL — `server_profile` unknown field / KeyError in `to_dict`.

- [ ] **Step 3: Implement**

In `config.py`, next to the existing feature flags (`debug`, `disable_auth`); add `Literal` to the existing `typing` import:

```python
server_profile: Literal["full", "eco"] = Field(
    default="full",
    description=(
        "Server composition profile: 'full' (default) or 'eco' "
        "(API-only benchmark server, no tournaments/MCP/UI)"
    ),
)
```

In `to_dict()` add `"server_profile": self.server_profile,`.

- [ ] **Step 4: Run tests, ruff, pyrefly** — all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/config.py tests/unit/dashboard/test_server_profile_config.py
git commit -m "feat(dashboard): add ATP_SERVER_PROFILE config field"
```

### Task 3: Full-route regression fixture (capture BEFORE refactor)

**Files:**
- Create: `tests/fixtures/dashboard_full_api_routes.json`
- Test: `tests/unit/dashboard/test_router_composition.py` (create; extended in Task 4)

**Interfaces:**
- Produces: fixture = sorted list of `"<path>::<METHOD,...>"` strings for the current aggregated `/api` router; helper `route_keys(router) -> list[str]`.

- [ ] **Step 1: Generate fixture from the CURRENT global router**

```bash
uv run python - <<'EOF'
import json
from fastapi.routing import APIRoute
from atp.dashboard.v2.routes import router

keys = sorted(
    f"{r.path}::{','.join(sorted(r.methods))}" if isinstance(r, APIRoute) else r.path
    for r in router.routes
)
with open("tests/fixtures/dashboard_full_api_routes.json", "w") as f:
    json.dump(keys, f, indent=1)
print(len(keys), "routes captured")
EOF
```

- [ ] **Step 2: Write the regression test (fails only after Task 4 if the split loses/renames a route)**

```python
"""Router aggregation regression: the full profile keeps every route."""

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.routing import APIRoute

FIXTURE = Path("tests/fixtures/dashboard_full_api_routes.json")


def route_keys(router: APIRouter) -> list[str]:
    return sorted(
        f"{r.path}::{','.join(sorted(r.methods))}"
        if isinstance(r, APIRoute)
        else r.path
        for r in router.routes
    )


def test_full_router_matches_pre_split_fixture() -> None:
    from atp.dashboard.v2.routes import build_router

    expected = json.loads(FIXTURE.read_text())
    assert route_keys(build_router(include_tournaments=True)) == expected
```

- [ ] **Step 3: Run it** — expected FAIL now (`build_router` does not exist yet). That is the failing test for Task 4.

- [ ] **Step 4: Commit fixture + test**

```bash
git add tests/fixtures/dashboard_full_api_routes.json tests/unit/dashboard/test_router_composition.py
git commit -m "test(dashboard): capture pre-split full API route fixture"
```

### Task 4: Router split + importer migration

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/__init__.py` (drop `router` import at line ~42 and from `__all__`)
- Modify: `tests/unit/dashboard/test_api.py:9`, `tests/unit/dashboard/test_api_detailed.py:18`, `tests/unit/dashboard/test_leaderboard_endpoint.py:222,229`
- Test: `tests/unit/dashboard/test_router_composition.py` (extend)

**Interfaces:**
- Produces: `build_core_router() -> APIRouter`, `build_tournament_router() -> APIRouter`, `build_router(*, include_tournaments: bool) -> APIRouter` in `atp.dashboard.v2.routes`. The module-level `router` global is **deleted**.
- Consumes: `route_keys()` helper from Task 3.

- [ ] **Step 1: Add eco-side tests to `test_router_composition.py`**

```python
def test_core_router_has_no_tournament_routes() -> None:
    from atp.dashboard.v2.routes import build_router

    keys = "\n".join(route_keys(build_router(include_tournaments=False)))
    for forbidden in ("/tournaments", "/games", "/el-farol", "/winners", "/builtins"):
        assert forbidden not in keys
    assert "/benchmarks" in keys
    assert "/runs/{run_id}/next-task" in keys


def test_routes_module_import_does_not_pull_games() -> None:
    """Fast check; the authoritative proof is the subprocess test (Task 6)."""
    import subprocess
    import sys

    code = (
        "import sys; import atp.dashboard.v2.routes; "
        "sys.exit(1 if 'game_envs' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
```

- [ ] **Step 2: Restructure `routes/__init__.py`**

Delete the six tournament-group imports from module level (`builtins_api`,
`el_farol_dashboard`, `games`, `tournament_api`, `tournament_live`,
`winners_api`) — all other router imports stay module-level as today. Replace
the `router = APIRouter()` block with:

```python
def build_core_router() -> APIRouter:
    """Aggregate every non-tournament API router (both profiles)."""
    router = APIRouter()
    router.include_router(auth_router)
    router.include_router(device_auth_router)
    router.include_router(home_router)
    router.include_router(agents_router)
    router.include_router(suites_router)
    router.include_router(tests_router)
    router.include_router(trends_router)
    router.include_router(comparison_router)
    router.include_router(leaderboard_router)
    router.include_router(public_leaderboard_router)
    router.include_router(marketplace_router)
    router.include_router(timeline_router)
    router.include_router(definitions_router)
    router.include_router(upload_router)
    router.include_router(templates_router)
    router.include_router(traces_router)
    router.include_router(metrics_router)
    router.include_router(costs_router)
    router.include_router(budgets_router)
    router.include_router(analytics_router)
    router.include_router(experiments_router)
    router.include_router(tenants_router)
    router.include_router(roles_router)
    router.include_router(sso_router)
    router.include_router(saml_router)
    router.include_router(audit_router)
    router.include_router(users_router)
    router.include_router(websocket_router)
    router.include_router(agent_traces_router)
    router.include_router(catalog_router)
    router.include_router(benchmark_api_router)
    router.include_router(token_api_router)
    router.include_router(agent_management_api_router)
    router.include_router(invite_api_router)
    return router


def build_tournament_router() -> APIRouter:
    """Aggregate tournament/game routers (full profile only).

    Imports live inside the function so the eco profile never imports
    the game stack (game_envs) transitively.
    """
    from atp.dashboard.v2.routes.builtins_api import (
        router as builtins_api_router,
    )
    from atp.dashboard.v2.routes.el_farol_dashboard import (
        router as el_farol_dashboard_router,
    )
    from atp.dashboard.v2.routes.games import router as games_router
    from atp.dashboard.v2.routes.tournament_api import (
        router as tournament_api_router,
    )
    from atp.dashboard.v2.routes.tournament_live import (
        router as tournament_live_router,
    )
    from atp.dashboard.v2.routes.winners_api import (
        router as winners_api_router,
    )

    router = APIRouter()
    router.include_router(games_router)
    router.include_router(el_farol_dashboard_router)
    router.include_router(tournament_api_router)
    router.include_router(tournament_live_router)
    router.include_router(builtins_api_router)
    router.include_router(winners_api_router)
    return router


def build_router(*, include_tournaments: bool) -> APIRouter:
    """Build the aggregated /api router for the requested profile."""
    router = build_core_router()
    if include_tournaments:
        router.include_router(build_tournament_router())
    return router
```

In `__all__`: remove `"router"` and the six tournament router names; add the
three `build_*` functions. Note: the fixture from Task 3 compares route
**sets** (sorted keys), so the tournament group moving to the end of the
include order is fine.

- [ ] **Step 3: Migrate importers**

- `v2/__init__.py`: delete `from atp.dashboard.v2.routes import router` and `"router"` from `__all__`.
- In the three tests, replace `from atp.dashboard.v2.routes import router` with:

```python
from atp.dashboard.v2.routes import build_router

router = build_router(include_tournaments=True)
```

(in `test_leaderboard_endpoint.py` the import is function-local at lines 222 and 229 — same replacement in place.)

- [ ] **Step 4: `factory.py` minimal bridge** — the module still imports the deleted `router`; to keep the branch green mid-refactor, replace line 36 with `from atp.dashboard.v2.routes import build_router` and line 282 with `app.include_router(build_router(include_tournaments=True), prefix="/api")`. (Profile gating lands in Task 5.)

- [ ] **Step 5: Run** `uv run pytest tests/unit/dashboard -v -x` + ruff + pyrefly — all green, including the Task 3 fixture test.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/__init__.py packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py packages/atp-dashboard/atp/dashboard/v2/factory.py tests/unit/dashboard/
git commit -m "refactor(dashboard): split router aggregation, drop global router"
```

### Task 5: Factory gating + loud full-guard

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Test: `tests/unit/dashboard/test_eco_profile_app.py` (create)

**Interfaces:**
- Consumes: `build_router`, `DashboardConfig.server_profile`.
- Produces: `create_app(config)` honoring the profile; eco root returns `{"profile": "eco", "docs": "/docs"}`.

- [ ] **Step 1: Write failing tests**

```python
"""Eco profile composition tests (spec §5, §7)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.factory import create_app

REQUIRED_ECO = [
    "/api/v1/benchmarks",
    "/api/v1/runs/{run_id}/next-task",
    "/api/v1/runs/{run_id}/submit",
    "/api/v1/tokens",
    "/api/v1/agents",
]
FORBIDDEN_ECO_PREFIXES = [
    "/mcp",
    "/api/v1/tournaments",
    "/api/v1/games",
    "/ui",
    "/static/v2",
    "/login",
    "/register",
    "/games",
    "/analytics",
]


def _config(profile: str) -> DashboardConfig:
    return DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        secret_key="test-secret-key",
        disable_auth=True,
        rate_limit_enabled=False,
        server_profile=profile,
    )


def _paths(app: FastAPI) -> set[str]:
    return {getattr(r, "path", "") for r in app.routes}


@pytest.fixture
def eco_app() -> FastAPI:
    return create_app(config=_config("eco"))


def test_eco_route_contract(eco_app: FastAPI) -> None:
    paths = _paths(eco_app)
    for required in REQUIRED_ECO:
        assert required in paths, f"missing {required}"
    for prefix in FORBIDDEN_ECO_PREFIXES:
        assert not any(p == prefix or p.startswith(prefix + "/") for p in paths), (
            f"forbidden {prefix} present"
        )


def test_eco_docs_and_root(eco_app: FastAPI) -> None:
    with TestClient(eco_app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/").json() == {"profile": "eco", "docs": "/docs"}


def test_eco_allows_multiple_workers(
    eco_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with TestClient(eco_app) as client:
        assert client.get("/openapi.json").status_code == 200


def test_full_rejects_multiple_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    app = create_app(config=_config("full"))
    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY=1"):
        with TestClient(app):
            pass


def test_full_profile_still_mounts_everything() -> None:
    paths = _paths(create_app(config=_config("full")))
    assert "/mcp" in paths
    assert "/ui/" in paths or any(p.startswith("/ui") for p in paths)
    assert "/api/v1/tournaments" in paths


def test_full_guard_names_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import atp.dashboard.v2.routes as routes

    def boom() -> None:
        raise ModuleNotFoundError("No module named 'game_envs'", name="game_envs")

    monkeypatch.setattr(routes, "build_tournament_router", boom)
    with pytest.raises(RuntimeError, match=r"atp-dashboard\[tournaments\]"):
        create_app(config=_config("full"))


def test_full_guard_does_not_mask_other_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import atp.dashboard.v2.routes as routes

    def boom() -> None:
        raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")

    monkeypatch.setattr(routes, "build_tournament_router", boom)
    with pytest.raises(ModuleNotFoundError):
        create_app(config=_config("full"))
```

- [ ] **Step 2: Run to verify failure** — eco contract tests fail (tournaments/UI/MCP still mounted).

- [ ] **Step 3: Implement gating in `factory.py`**

Top level: delete `from atp.dashboard.tournament.deadlines import run_deadline_worker` (old line 24); keep `from atp.dashboard.v2.routes import build_router` (from Task 4). Inside `create_app()` after config resolution:

```python
    is_full = config.server_profile == "full"
    logger.info("Creating dashboard app with server profile: %s", config.server_profile)

    mcp_app = mcp_http_app = None
    run_deadline_worker = None
    tournament_event_bus = None
    if is_full:
        try:
            from atp.dashboard.mcp import mcp_server, tournament_event_bus
            from atp.dashboard.mcp import tools as _mcp_tools  # noqa: F401
            from atp.dashboard.mcp.auth import MCPAuthMiddleware
            from atp.dashboard.tournament.deadlines import run_deadline_worker

            api_router = build_router(include_tournaments=True)
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.split(".")[0] in {
                "game_envs",
                "fastmcp",
                "mcp",
            }:
                raise RuntimeError(
                    "Server profile 'full' requires the tournament stack: "
                    "install atp-dashboard[tournaments] or set "
                    "ATP_SERVER_PROFILE=eco."
                ) from exc
            raise
        mcp_app = mcp_server.http_app(transport="sse")
        mcp_http_app = mcp_server.http_app(transport="streamable-http")
    else:
        api_router = build_router(include_tournaments=False)
```

Note the `exc.name.split(".")[0]` — a missing **sub**module of an optional root (e.g. `mcp.types`) is the same "extra not installed" condition. `atp.dashboard.mcp` shadows nothing: its `exc.name` root would be `atp`, so it propagates untouched.

`_combined_lifespan` gates the worker and MCP lifespans:

```python
    @asynccontextmanager
    async def _combined_lifespan(app_: FastAPI) -> AsyncGenerator[None, None]:
        if not is_full:
            async with lifespan(app_):
                yield
            return

        assert_single_worker()
        async with lifespan(app_):
            from atp.dashboard.database import get_database

            session_factory = get_database().session_factory
            shutdown_event = asyncio.Event()
            worker_task = asyncio.create_task(
                run_deadline_worker(
                    session_factory,
                    tournament_event_bus,
                    shutdown_event=shutdown_event,
                )
            )
            try:
                async with mcp_app.router.lifespan_context(app_):
                    async with mcp_http_app.router.lifespan_context(app_):
                        yield
            finally:
                shutdown_event.set()
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.gather(worker_task, return_exceptions=True)
```

Middleware: keep `_slowapi_except_streams` for full; eco has no SSE paths, so:

```python
    if is_full:
        app.add_middleware(_slowapi_except_streams)
    else:
        app.add_middleware(SlowAPIMiddleware)
```

MCP mounts (`/mcp`, `/mcp-http` + `MCPAuthMiddleware`) go inside `if is_full:`. The whole UI block — `ui`/`winners_ui`/`admin_ui` router imports and includes, templates, `/static/v2`, root redirect, legacy redirects — also moves under `if is_full:`; the eco branch instead registers:

```python
    else:

        @app.get("/")
        async def eco_root() -> dict[str, str]:
            """Machine-readable root for the API-only eco profile."""
            return {"profile": "eco", "docs": "/docs"}
```

`app.include_router(api_router, prefix="/api")` stays unconditional. Add `logger = logging.getLogger(__name__)` at module level if absent.

- [ ] **Step 4: Run** `uv run pytest tests/unit/dashboard -v` + ruff + pyrefly — all green (regression fixture from Task 3 must still pass for full).

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/factory.py tests/unit/dashboard/test_eco_profile_app.py
git commit -m "feat(dashboard): eco server profile — API-only composition"
```

### Task 6: Subprocess import-isolation test (authoritative)

**Files:**
- Test: `tests/integration/dashboard/test_eco_import_isolation.py` (create; make sure `tests/integration/dashboard/__init__.py` exists — create empty if not)

**Interfaces:**
- Consumes: `create_app`, `ATP_SERVER_PROFILE` env; blocked roots `game_envs`, `fastmcp`, `mcp`.

- [ ] **Step 1: Write the test**

```python
"""Prove the eco profile never imports the tournament stack.

Runs a CLEAN interpreter with a meta_path blocker installed before any
atp import — pytest-collection pre-imports cannot leak in (spec §7).
"""

import json
import subprocess
import sys
import textwrap

CHILD = textwrap.dedent("""
    import sys

    class _Block:
        BLOCKED = {"game_envs", "fastmcp", "mcp"}

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.BLOCKED:
                raise ModuleNotFoundError(
                    f"No module named {fullname!r}", name=fullname
                )
            return None

    sys.meta_path.insert(0, _Block())

    import os
    os.environ["ATP_SERVER_PROFILE"] = "eco"
    os.environ["ATP_SECRET_KEY"] = "smoke"

    import json
    from atp.dashboard.v2.config import DashboardConfig
    from atp.dashboard.v2.factory import create_app

    app = create_app(
        config=DashboardConfig(
            server_profile="eco",
            secret_key="smoke",
            database_url="sqlite+aiosqlite:///:memory:",
        )
    )
    print(json.dumps(sorted({getattr(r, "path", "") for r in app.routes})))
""")


def test_eco_app_builds_with_tournament_stack_blocked() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    paths = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert "/api/v1/benchmarks" in paths
    assert not any(p.startswith("/mcp") for p in paths)
    assert not any(p.startswith("/api/v1/tournaments") for p in paths)
```

- [ ] **Step 2: Run** `uv run pytest tests/integration/dashboard/test_eco_import_isolation.py -v` — PASS. If it fails, the stderr shows exactly which module chain still reaches the blocked roots; fix the leak in factory/routes, do not weaken the blocker.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/dashboard/test_eco_import_isolation.py
git commit -m "test(dashboard): subprocess proof eco boots without game stack"
```

### Task 7: Docs, full verification, PR-1

**Files:**
- Modify: `.env.example` (server section), `CLAUDE.md` (Environment Variables list)

- [ ] **Step 1: Document the env var**

`.env.example` — add near the other ATP server settings:

```bash
# Server composition profile: full (default) or eco (API-only benchmark
# server: no tournaments, no MCP, no HTML UI)
ATP_SERVER_PROFILE=full
```

`CLAUDE.md` Environment Variables section — add one line:

```markdown
- `ATP_SERVER_PROFILE` - Server composition profile: `full` (default) or `eco` (API-only benchmark server — Benchmark API + auth/tokens/agents; no tournaments, MCP, or HTML UI). See `docs/superpowers/specs/2026-08-14-eco-server-profile-design.md`.
```

(CLAUDE.md edits run the `check_plan_citations.py` pre-commit hook — the added path citation must exist, which it does.)

- [ ] **Step 2: Full verification**

```bash
uv run ruff format . && uv run ruff check .
uv run pyrefly check
uv run pytest tests/ -v -m "not slow"
```

All green; report any pre-existing failures separately, do not fix unrelated code.

- [ ] **Step 3: Commit + push + PR**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: document ATP_SERVER_PROFILE"
git push -u origin feat/eco-server-profile
gh pr create --title "feat(dashboard): eco server profile (API-only benchmark server)" --body "..."
```

PR body: link the spec, state the contract (API-only), note full-profile route set proven unchanged by fixture, and that packaging (extras) lands in PR-2. End with the standard generated-with footer. Then watch Copilot review per repo workflow; fix valid points in-branch.

**STOP: human merges PR-1 before PR-2 starts.**

---

## PR-2 — packaging (branch `feat/eco-server-packaging`, after PR-1 merge)

### Task 8: Move tournament stack to `[tournaments]` extra

**Files:**
- Modify: `packages/atp-dashboard/pyproject.toml`
- Modify: `pyproject.toml` (root — extras `dashboard`, `all`; new `eco-server`)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: extra name `tournaments` on `atp-dashboard`; root extras `dashboard = ["atp-dashboard[tournaments]>=1.0.0"]`, `eco-server = ["atp-dashboard>=1.0.0"]`.

- [ ] **Step 1: `packages/atp-dashboard/pyproject.toml`** — remove from `dependencies`:

```toml
    "fastmcp>=3.0",
    "mcp>=1.28.1,<2",
    "game-environments>=1.0.0",
```

(keep the CVE comment with the moved pin) and add:

```toml
[project.optional-dependencies]
tournaments = [
    "fastmcp>=3.0",
    # Direct floor+ceiling for the transitive `mcp` SDK: >=1.28.1 is the
    # patched version for CVE-2026-59950 / -52869 / -52870; <2 mirrors
    # fastmcp's own ceiling (fastmcp<=3.4.5 requires mcp<2.0). Lift both
    # together when fastmcp ships SDK-v2 support.
    "mcp>=1.28.1,<2",
    "game-environments>=1.0.0",
]
```

(merge into the existing `[project.optional-dependencies]` table with `enterprise`/`analytics`/`postgres`/`dev`). `[tool.uv.sources] game-environments = { workspace = true }` stays.

- [ ] **Step 2: Root `pyproject.toml`** — change:

```toml
dashboard = [
    "atp-dashboard[tournaments]>=1.0.0",
]
eco-server = [
    "atp-dashboard>=1.0.0",
]
```

and in `all` replace the three dashboard lines (`atp-dashboard>=1.0.0`, `atp-dashboard[enterprise]`, `atp-dashboard[analytics]`) with the single `"atp-dashboard[tournaments,enterprise,analytics]>=1.0.0"`.

- [ ] **Step 3: Relock + resync + full test sweep**

```bash
uv lock
uv sync --group dev --all-extras
uv run pytest tests/ -v -m "not slow"
```

Dev workspace still has everything (all-extras), so the full profile and its tests stay green.

- [ ] **Step 4: Commit**

```bash
git switch main && git pull --ff-only && git switch -c feat/eco-server-packaging
git add packages/atp-dashboard/pyproject.toml pyproject.toml uv.lock
git commit -m "build: move game/MCP stack to atp-dashboard[tournaments] extra"
```

(Branch creation happens before the edits in practice — do it first, then edit; listed here in one place for clarity.)

### Task 9: Wheel-level minimal-install smoke test

**Files:**
- Test: `tests/integration/packaging/test_eco_wheel_smoke.py` (create, marked `slow`; create `tests/integration/packaging/__init__.py`)

**Interfaces:**
- Consumes: built wheels of `atp-core` and `atp-dashboard`; PyPI for third-party deps (network required → `slow` marker).

- [ ] **Step 1: Write the test**

```python
"""Wheel-level proof: minimal atp-dashboard install has no game stack.

Builds real wheels, installs WITHOUT the tournaments extra into a fresh
venv, and boots the eco app there. This is the packaging proof the
sys.modules/meta_path tests cannot give (spec §7, PR-2).
"""

import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[3]

CHILD = """
import json
import os

os.environ["ATP_SERVER_PROFILE"] = "eco"
os.environ["ATP_SECRET_KEY"] = "smoke"

for blocked in ("game_envs", "fastmcp", "mcp"):
    try:
        __import__(blocked)
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit(f"{blocked} leaked into the minimal install")

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.factory import create_app

app = create_app(
    config=DashboardConfig(
        server_profile="eco",
        secret_key="smoke",
        database_url="sqlite+aiosqlite:///:memory:",
    )
)
paths = {getattr(r, "path", "") for r in app.routes}
assert "/api/v1/benchmarks" in paths
assert not any(p.startswith(("/mcp", "/api/v1/tournaments")) for p in paths)
print("ECO-WHEEL-SMOKE-OK")
"""


def _run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)  # type: ignore[call-overload]
    assert proc.returncode == 0, f"{cmd}\n{proc.stdout}\n{proc.stderr}"
    return proc


def test_minimal_wheel_install_boots_eco(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _run(["uv", "build", str(REPO / "packages/atp-core"), "-o", str(dist)])
    _run(["uv", "build", str(REPO / "packages/atp-dashboard"), "-o", str(dist)])

    wheel = next(dist.glob("atp_dashboard-*.whl"))
    metadata = subprocess.run(
        [sys.executable, "-m", "zipfile", "-e", str(wheel), str(tmp_path / "w")],
        capture_output=True,
    )
    assert metadata.returncode == 0
    meta_text = next((tmp_path / "w").rglob("METADATA")).read_text()
    assert 'game-environments>=1.0.0; extra == "tournaments"' in meta_text.replace(
        "'", '"'
    )
    for line in meta_text.splitlines():
        if line.startswith("Requires-Dist: game-environments"):
            assert "tournaments" in line, f"unconditional game dep: {line}"

    env_dir = tmp_path / "venv"
    venv.create(env_dir, with_pip=True)
    py = str(env_dir / "bin" / "python")
    _run([py, "-m", "pip", "install", "--find-links", str(dist), str(wheel)])
    proc = _run([py, "-c", CHILD])
    assert "ECO-WHEEL-SMOKE-OK" in proc.stdout
```

- [ ] **Step 2: Run** `uv run pytest tests/integration/packaging/test_eco_wheel_smoke.py -v` (slow, needs network for third-party deps). Expected: PASS. If `game-environments` appears as an unconditional `Requires-Dist`, the `[tool.uv.sources]` workspace pin leaked — fix packaging, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/packaging/
git commit -m "test(packaging): wheel-level eco minimal-install smoke"
```

### Task 10: Verification, PR-2, dispatcher follow-up

- [ ] **Step 1: Full verification** — ruff format/check, `uv run pyrefly check`, `uv run pytest tests/ -v -m "not slow"`, plus the slow smoke test once.

- [ ] **Step 2: Push + PR**

```bash
git push -u origin feat/eco-server-packaging
gh pr create --title "build: atp-dashboard[tournaments] extra + eco-server root extra" --body "..."
```

PR body links the spec §6–7 and PR-1; notes `dashboard`/`all` extras keep byte-identical behavior. Watch Copilot review per repo workflow.

- [ ] **Step 3 (after human merges both PRs): dispatcher inbox issue** — per ADR-ECO-006 (do NOT edit dispatcher files). Use the `repo-inbox` skill, or directly:

```bash
gh issue create --repo andrei-shtanakov/dispatcher \
  --label inbox \
  --title "Display ATP benchmark runs/leaderboard (eco server)" \
  --body "slug: atp-eco-benchmark-view
from: atp-platform

ATP now ships an API-only eco server profile (Benchmark API + auth/tokens/agents,
no UI). Proposal: dispatcher renders benchmark runs and the leaderboard by
consuming that API / atp-platform-sdk. Contract: /api/v1/benchmarks,
/api/v1/runs/{id}/*, leaderboard endpoints; SDK >= 2.0.0."
```

(Check the label exists in dispatcher first: `gh label list --repo andrei-shtanakov/dispatcher`.)
