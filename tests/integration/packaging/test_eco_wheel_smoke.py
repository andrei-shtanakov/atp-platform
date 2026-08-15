"""Wheel-level proof: minimal atp-dashboard install has no game stack.

Builds real wheels, installs WITHOUT the tournaments extra into a fresh
venv, and boots the eco app there. This is the packaging proof the
sys.modules/meta_path tests cannot give (spec §7, PR-2).
"""

import subprocess
import sys
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

from fastapi.testclient import TestClient

from atp.dashboard.v2.config import DashboardConfig
from atp.dashboard.v2.factory import create_app

app = create_app(
    config=DashboardConfig(
        server_profile="eco",
        secret_key="smoke",
        database_url="sqlite+aiosqlite:///:memory:",
    )
)
with TestClient(app) as client:
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = set(spec.json()["paths"])
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

    # Use a standalone uv project so this packaging test obeys the repository's
    # uv-only policy. Add both local wheels in one resolution: PyPI hosts an
    # unrelated, higher-version "atp-core", which must not satisfy the
    # dashboard wheel's atp-core dependency during this smoke test.
    smoke_project = tmp_path / "smoke-project"
    _run(
        [
            "uv",
            "init",
            "--bare",
            "--no-workspace",
            "--vcs",
            "none",
            str(smoke_project),
        ]
    )
    core_wheel = next(dist.glob("atp_core-*.whl"))
    _run(
        [
            "uv",
            "add",
            "--project",
            str(smoke_project),
            str(core_wheel),
            str(wheel),
        ]
    )
    # cwd must not be the repo root: `python -c` prepends cwd to sys.path,
    # and the repo's own atp/ namespace package would shadow the wheel
    # actually installed in the standalone project's environment.
    proc = _run(
        [
            "uv",
            "run",
            "--project",
            str(smoke_project),
            "python",
            "-c",
            CHILD,
        ],
        cwd=str(tmp_path),
    )
    assert "ECO-WHEEL-SMOKE-OK" in proc.stdout
