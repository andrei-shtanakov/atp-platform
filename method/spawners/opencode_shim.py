#!/usr/bin/env python3
"""opencode spawner shim — `opencode run --format json` (GLM via opencode).

Non-routable API/CLI baseline row. Reads OPENCODE_MODEL (e.g. glm-5.1; the
shim prefixes the `opencode/` provider) and OPENCODE_BIN (default "opencode").
Auth is opencode's own (operator's OPENCODE_GLM_API_KEY / `opencode auth`).

Each invocation gets an ISOLATED opencode data dir (XDG_DATA_HOME → temp,
with auth.json seeded from the operator's real one): concurrent harness runs
otherwise contend on the shared SQLite state in ~/.local/share/opencode/ —
"Error: Unexpected error database is locked" killed/hung 20-40% of case-runs
in the 2026-07-02/03 sweeps and masqueraded first as model weakness, then as
a provider stall. The temp dir is removed after the run.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from _cli_common import run  # pyrefly: ignore[missing-import]


def _argv(binary: list[str], model: str, prompt: str) -> list[str]:
    return [*binary, "run", "--format", "json", "-m", model, prompt]


def _parse(stdout: str) -> tuple[str, int | None, int | None]:
    """Concat `type:text` part.text; tokens from the `step_finish` event."""
    text = ""
    in_tok: int | None = None
    out_tok: int | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "text":
            text += (event.get("part") or {}).get("text", "")
        elif event.get("type") == "step_finish":
            tokens = (event.get("part") or {}).get("tokens") or {}
            in_tok = tokens.get("input")
            out_tok = tokens.get("output")
    return text, in_tok, out_tok


def _isolated_data_home() -> str:
    """Create a temp XDG_DATA_HOME seeded with opencode's auth.json.

    opencode resolves its state dir as ``$XDG_DATA_HOME/opencode`` (verified
    on 1.17.5, incl. darwin). A fresh dir per invocation means each subprocess
    gets its own SQLite files — no cross-run lock contention. Credentials live
    in the same dir, so the operator's auth.json is copied in; everything else
    (session DBs, caches) starts empty on purpose.
    """
    src = (
        Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        / "opencode"
        / "auth.json"
    )
    tmp = tempfile.mkdtemp(prefix="atp-opencode-")
    try:
        dst = Path(tmp) / "opencode"
        dst.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst / "auth.json")
    except OSError:
        # Don't leak the temp dir if seeding fails.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp


def main() -> int:
    """Run one opencode invocation inside an isolated data home.

    The prior ``XDG_DATA_HOME`` is restored afterwards so the mutation
    cannot outlive the run (matters for in-process reuse/tests).
    """
    prior = os.environ.get("XDG_DATA_HOME")
    data_home = _isolated_data_home()
    os.environ["XDG_DATA_HOME"] = data_home
    try:
        return run(
            bin_env="OPENCODE_BIN",
            default_bin="opencode",
            model_env="OPENCODE_MODEL",
            default_provider="opencode",
            argv=_argv,
            parse_output=_parse,
        )
    finally:
        if prior is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = prior
        shutil.rmtree(data_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
