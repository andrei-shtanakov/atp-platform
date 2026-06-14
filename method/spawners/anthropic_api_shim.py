#!/usr/bin/env python3
"""anthropic_api spawner shim for ATP's CLI adapter — the RAW-API ablation baseline.

Contract: read an ATPRequest JSON from stdin, call the Anthropic Messages API
directly (no Claude Code product harness), normalize the result into an
ATPResponse JSON on stdout. agent_id is set by the run wiring, not here.

Why this exists (R-07 Phase-1b Ticket B, "product harness vs raw API"):
  - `claude_code_shim.py` reaches the model THROUGH `claude -p`, which carries
    Claude Code's system prompt / tool scaffolding — a *product harness*.
  - this shim hits the bare Messages API with ONLY the shared REVIEW_ENVELOPE as
    the user turn — the *raw model*.
The two are deliberately NOT equalizable: that gap is the thing being measured.

Guardrails (so this stays a fair baseline, not "a different agent"):
  - SAME model + SAME prompt envelope as the CLI shim (imported, not re-typed).
  - NO assistant prefill and NO system prompt — prefilling the agent-under-test
    would make the ablation meaningless.
  - `anthropic_api` is a LABELED BASELINE row only; it must never substitute the
    CLI `agent_id` in arbiter routing.

Needs `ANTHROPIC_API_KEY` (absent in some envs — the run harness skips this agent
with a clear message when the key is missing).
"""

import json
import os
import sys

# Reuse the CLI shim's pinned model + prompt envelope verbatim so the only
# difference between the two agents is harness-vs-raw-API (sys.path[0] is this
# script's dir when run as `python method/spawners/anthropic_api_shim.py`).
from claude_code_shim import MODEL, _build_prompt

MAX_TOKENS = int(os.environ.get("API_MAX_TOKENS", "4096"))


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
    """Read ATPRequest from stdin, call the Messages API, emit ATPResponse."""
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except (ValueError, TypeError) as exc:
        # Invalid/empty stdin must still produce a contract-shaped failed
        # response, not crash the shim.
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fail(task_id, "ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - env guard
        return _fail(task_id, f"anthropic SDK not installed: {exc}")

    prompt = _build_prompt(request)
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - any API error becomes a failed run
        return _fail(task_id, f"anthropic API error: {exc}")

    text = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    )
    # usage / its fields may be absent depending on SDK/response shape; default
    # to 0 rather than crashing on an attribute error.
    usage = getattr(msg, "usage", None)
    in_tok = getattr(usage, "input_tokens", None) or 0
    out_tok = getattr(usage, "output_tokens", None) or 0
    response = {
        "version": "1.0",
        "task_id": task_id,
        "status": "completed",
        "artifacts": [
            {
                "type": "file",
                "path": "review.md",
                "content": text,
                "content_type": "text/markdown",
            }
        ],
        "metrics": {
            "total_tokens": in_tok + out_tok,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            # No cost field on the raw API response; the CLI shim gets real cost
            # from `claude --output-format json`. Left null for the baseline.
            "cost_usd": None,
        },
    }
    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
