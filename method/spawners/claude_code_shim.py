#!/usr/bin/env python3
"""claude_code spawner shim for ATP's CLI adapter.

Contract: read an ATPRequest JSON from stdin, run the agent-under-test
(`claude -p <prompt> --output-format json`), normalize its output into an
ATPResponse JSON on stdout. agent_id is set by which adapter-config the suite
selects, not here.

PINNED (R-07 Phase 1 guardrail #1 — measure the agent, not a random prompt):
  model            = claude-opus-4-8   (override via CLAUDE_MODEL)
  prompt-envelope  = REVIEW_ENVELOPE below (verbatim; changes are a new agent)
Record the exact invocation in each case's provenance.
"""

import json
import os
import shlex
import subprocess
import sys

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

REVIEW_ENVELOPE = (
    "You are a senior code reviewer. Review the material below. Report each issue "
    "with rule_id, file:line, severity, and a concrete fix. Do not invent issues. "
    "Output only the review.\n\n{task}"
)


def _build_prompt(request: dict) -> str:
    """Build the full review prompt from an ATPRequest dict."""
    task = request.get("task") or {}
    body = task.get("description", "")
    for art in (request.get("context") or {}).get("artifacts", []) or []:
        if art.get("content"):
            body += f"\n\n--- {art.get('id', 'artifact')} ---\n{art['content']}"
    return REVIEW_ENVELOPE.format(task=body)


def main() -> int:
    """Read ATPRequest from stdin, invoke claude, emit ATPResponse to stdout."""
    request = json.loads(sys.stdin.read())
    prompt = _build_prompt(request)
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
                    "status": "error",
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
