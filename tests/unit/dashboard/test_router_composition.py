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
        else getattr(r, "path", str(r))
        for r in router.routes
    )


def test_full_router_matches_pre_split_fixture() -> None:
    from atp.dashboard.v2.routes import build_router

    expected = json.loads(FIXTURE.read_text())
    assert route_keys(build_router(include_tournaments=True)) == expected


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
