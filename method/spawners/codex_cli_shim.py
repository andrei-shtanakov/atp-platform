#!/usr/bin/env python3
"""codex_cli spawner shim for ATP's CLI adapter — REAL `codex exec` (OpenAI).

Contract: read an ATPRequest JSON from stdin, run the agent-under-test
(`codex exec ...`) non-interactively, normalize its output into an ATPResponse
JSON on stdout. agent_id is set by which adapter-config the suite selects, not
here.

Why this exists (R-07 agent matrix):
  - the matrix needs more *routable* CLI agents than just `claude_code`. `codex`
    (OpenAI's product CLI) is a second real product harness, reached through its
    own `codex exec` non-interactive path — a peer to claude_code, not a baseline.

Guardrails (so this stays a fair matrix row):
  - SAME prompt envelope as every other shim. All shims draw from the single
    shared source (`atp_method.envelopes`), so they cannot drift apart.
  - The model is NOT hardcoded: codex uses its configured default unless
    CODEX_MODEL is set, so we never pin a model id that may not exist.

Invocation (determined empirically against codex-cli 0.139):
  codex exec --skip-git-repo-check --sandbox read-only --color never \
    --output-last-message <tmpfile> [-m CODEX_MODEL] <prompt>
  The final assistant message is read back from <tmpfile>. read-only sandbox runs
  unattended (no prompt/hang); our task is pure text-out with no side effects.

Binary from `CODEX_BIN` (default "codex"), shlex.split so a fake binary like
"python .../fake_codex.py" works in tests. Model from `CODEX_MODEL` (unset =>
codex's configured default; -m is simply not passed).

Tokens/cost are reported NULL: `codex exec --output-last-message` writes only the
final message, not a machine-readable usage block, so we don't fabricate counts
(null = unknown, per the contract).
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from atp_method.envelopes import build_prompt, get_envelope

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
CODEX_MODEL = os.environ.get("CODEX_MODEL")
REQUEST_TIMEOUT_S = 600.0


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


def main() -> int:
    """Read ATPRequest from stdin, invoke codex, emit ATPResponse to stdout."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        # Invalid/empty stdin must still produce a contract-shaped failed
        # response, not crash the shim.
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    prompt = build_prompt(request, get_envelope("review"))

    # codex writes the final assistant message to this file; we read it back.
    fd, tmp_path = tempfile.mkstemp(prefix="codex_msg_", suffix=".txt")
    os.close(fd)
    try:
        argv = [
            *shlex.split(CODEX_BIN),
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            tmp_path,
        ]
        if CODEX_MODEL:
            argv += ["-m", CODEX_MODEL]
        argv.append(prompt)

        try:
            proc = subprocess.run(argv, capture_output=True, timeout=REQUEST_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            return _fail(task_id, f"codex invocation error: {exc}")

        message = Path(tmp_path).read_text() if Path(tmp_path).exists() else ""
        if proc.returncode != 0 or not message.strip():
            err = proc.stderr.decode(errors="replace")[:2000]
            return _fail(
                task_id,
                f"codex exec failed (rc={proc.returncode}): {err}",
            )

        response = {
            "version": "1.0",
            "task_id": task_id,
            "status": "completed",
            "artifacts": [
                {
                    "type": "file",
                    "path": "review.md",
                    "content": message,
                    "content_type": "text/markdown",
                }
            ],
            "metrics": {
                # `codex exec --output-last-message` reports no usage block, so
                # token counts are unknown (null), not fabricated.
                "total_tokens": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
            },
        }
        sys.stdout.write(json.dumps(response))
        return 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
