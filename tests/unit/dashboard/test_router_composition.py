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
