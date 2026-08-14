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
