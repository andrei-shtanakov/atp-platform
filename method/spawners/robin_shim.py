#!/usr/bin/env python3
"""Robin spawner shim for ATP's CLI adapter — the ecosystem Q&A agent (robin-runtime).

Contract: read an ATPRequest JSON from stdin, run Robin non-interactively
(`python -m robin.agent "<question>"` in the sibling robin-runtime repo), normalize
its stdout into an ATPResponse JSON on stdout. Any error becomes a status:"failed"
response — never a crash (same contract as every other shim here).

Why this exists (stage 4 of the Robin self-improvement loop — proposal
devtools/proposals/2026-07-10-robin-self-improvement.md): answer failures logged by
robin-runtime (gaps.jsonl) graduate into regression eval cases; the suite
examples/test_suites/robin_regression.yaml gates changes to Robin's prompt, tools
and retrieval. Robin is NOT a coding agent — it is deliberately absent from
method/agents-catalog.toml (that roster feeds arbiter's routing).

Unlike the coding-agent shims this one does not use the shared prompt envelope:
Robin answers the question verbatim (task.description), grounded in its own KB.

Robin's output is plain text: `Q: …`, `A: …` (when ANTHROPIC_API_KEY is set),
`(cost $N)`, then `Grounding sources (N):` with `path:line: text` lines. Without a
key Robin runs retrieve-only and still prints the grounding block — source-pattern
assertions keep working; answer-text assertions need the key.

Env:
  ROBIN_DIR — robin-runtime checkout (default: sibling of this repo);
  ROBIN_BIN — interpreter command (default "uv run python", shlex-split; run
              with cwd=ROBIN_DIR, so uv resolves robin-runtime's venv from
              the working directory — no --project flag is passed).
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REQUEST_TIMEOUT_S = 300.0
_COST_RE = re.compile(r"\(cost \$([0-9.]+)\)")


def _fail(task_id: str, error: str) -> int:
    """Emit a status=failed ATPResponse (the adapter reads it off stdout)."""
    sys.stdout.write(
        json.dumps(
            {
                "version": "1.0",
                "task_id": task_id,
                "status": "failed",
                "artifacts": [],
                "metrics": {},
                "error": error[:2000],
            }
        )
    )
    return 0


def _robin_dir() -> Path:
    default = Path(__file__).resolve().parents[2].parent / "robin-runtime"
    return Path(os.environ.get("ROBIN_DIR", str(default)))


def main() -> int:
    """Read ATPRequest from stdin, invoke Robin, emit ATPResponse to stdout."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    question = ((request.get("task") or {}).get("description") or "").strip()
    if not question:
        return _fail(task_id, "empty task.description — nothing to ask Robin")

    robin_dir = _robin_dir()
    if not robin_dir.is_dir():
        return _fail(task_id, f"robin-runtime not found at {robin_dir} (set ROBIN_DIR)")

    binary = shlex.split(os.environ.get("ROBIN_BIN", "uv run python"))
    argv = [*binary, "-m", "robin.agent", question]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            timeout=REQUEST_TIMEOUT_S,
            cwd=robin_dir,
        )
    except subprocess.TimeoutExpired:
        return _fail(task_id, f"robin timed out after {REQUEST_TIMEOUT_S}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return _fail(task_id, f"robin invocation error: {exc}")

    stdout = proc.stdout.decode(errors="replace")
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[:2000]
        return _fail(task_id, f"robin failed (rc={proc.returncode}): {err}")
    if not stdout.strip():
        return _fail(task_id, "robin produced no output")

    cost_match = _COST_RE.search(stdout)
    response = {
        "version": "1.0",
        "task_id": task_id,
        "status": "completed",
        "artifacts": [
            {
                "type": "file",
                "path": "answer.md",
                "content": stdout,
                "content_type": "text/markdown",
            }
        ],
        "metrics": {
            # Robin prints dollars, not tokens — tokens stay null (unknown).
            "total_tokens": None,
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": float(cost_match.group(1)) if cost_match else None,
        },
    }
    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
