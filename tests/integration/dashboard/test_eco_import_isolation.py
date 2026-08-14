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
