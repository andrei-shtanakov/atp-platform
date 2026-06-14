#!/usr/bin/env python3
"""claude_code spawner shim for ATP's CLI adapter.

Contract: read an ATPRequest JSON from stdin, run the agent-under-test
(`claude -p <prompt> --output-format json`), normalize its output into an
ATPResponse JSON on stdout. agent_id is set by which adapter-config the suite
selects, not here.

PINNED (R-07 Phase 1 guardrail #1 — measure the agent, not a random prompt):
  model            = DEFAULT_MODEL    (override via CLAUDE_MODEL)
  prompt-envelope  = atp_method.envelopes.get_envelope("review") (the shared,
                     spawner-agnostic contract; changing it is a new agent)
Record the exact invocation in each case's provenance.
"""

import json
import os
import shlex
import subprocess
import sys

from atp_method.envelopes import DEFAULT_MODEL, build_prompt, get_envelope

MODEL = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")


def main() -> int:
    """Read ATPRequest from stdin, invoke claude, emit ATPResponse to stdout."""
    request = json.loads(sys.stdin.read())
    prompt = build_prompt(request, get_envelope("review"))
    cmd = shlex.split(CLAUDE_BIN) + [
        "-p",
        prompt,
        "--model",
        MODEL,
        "--output-format",
        "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        sys.stdout.write(
            json.dumps(
                {
                    "version": "1.0",
                    "task_id": request.get("task_id", ""),
                    "status": "failed",
                    "artifacts": [],
                    "metrics": {},
                    "error": proc.stderr.decode()[:2000],
                }
            )
        )
        return 0
    out = json.loads(proc.stdout.decode())
    usage = out.get("usage") or {}
    in_tok: int | None = usage.get("input_tokens")
    out_tok: int | None = usage.get("output_tokens")
    has_tokens = in_tok is not None or out_tok is not None
    total: int | None = (in_tok or 0) + (out_tok or 0) if has_tokens else None
    response = {
        "version": "1.0",
        "task_id": request.get("task_id", ""),
        "status": "completed",
        "artifacts": [
            {
                "type": "file",
                "path": "review.md",
                "content": out.get("result", ""),
                "content_type": "text/markdown",
            }
        ],
        "metrics": {
            "total_tokens": total,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": out.get("total_cost_usd"),
        },
    }
    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
