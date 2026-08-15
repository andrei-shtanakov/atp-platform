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

    env_dir = tmp_path / "venv"
    # symlinks=True: uv-managed CPython on macOS resolves its shared lib via
    # @executable_path/../lib/libpython3.12.dylib; a copied (non-symlinked)
    # interpreter loses that relative path and aborts at startup.
    venv.create(env_dir, with_pip=True, symlinks=True)
    py = str(env_dir / "bin" / "python")
    # Install the local atp-core wheel by explicit path FIRST. PyPI hosts an
    # unrelated "atp-core" package (an "Attested Transport Protocol" SDK) at
    # a higher version; atp-dashboard's unpinned `atp-core>=1.0.0` requirement
    # would otherwise resolve against that impostor instead of our wheel.
    # With our atp-core already installed and satisfying the constraint, pip
    # never needs to consult the index for it.
    core_wheel = next(dist.glob("atp_core-*.whl"))
    _run([py, "-m", "pip", "install", str(core_wheel)])
    _run([py, "-m", "pip", "install", "--find-links", str(dist), str(wheel)])
    # cwd must not be the repo root: `python -c` prepends cwd to sys.path,
    # and the repo's own atp/ namespace package would shadow the wheel
    # actually installed in the venv's site-packages.
    proc = _run([py, "-c", CHILD], cwd=str(tmp_path))
    assert "ECO-WHEEL-SMOKE-OK" in proc.stdout
