#!/usr/bin/env python3
"""ollama spawner shim for ATP's CLI adapter — local-model matrix agents.

Contract: read an ATPRequest JSON from stdin, call a local Ollama model via its
HTTP `/api/chat` endpoint, normalize the result into an ATPResponse JSON on
stdout. agent_id (and the concrete model) is set by the run wiring, not here.

Why this exists (R-07 agent matrix):
  - the cloud agents (claude_code, anthropic_api) saturate the deterministic
    findings_match gate, so they carry little routing signal.
  - a *weak* local model (e.g. llama3.2:1b) fails the harder axis levels, which
    creates the separation the matrix needs to prove the tube carries signal.

Guardrails (so this stays a fair matrix row):
  - SAME prompt envelope as every other shim. All shims draw from the single
    shared source (`atp_method.envelopes`), so they cannot drift apart.
  - NO assistant prefill and NO system prompt — the envelope is the whole turn.
  - cost_usd is reported NULL: local inference is free, but "free" must not be
    confused with a real cost number (the harness treats null as unknown,
    matching the raw-API baseline).

Stdlib only — no new dependency: the model is reached with `urllib.request`.

Needs `OLLAMA_MODEL` (no default — the run harness sets it per agent_id and skips
this agent with a clear message when it is missing). Host comes from
`OLLAMA_HOST` (default http://localhost:11434).
"""

import json
import os
import sys
import urllib.error
import urllib.request

from atp_method.envelopes import build_prompt, get_envelope

DEFAULT_HOST = "http://localhost:11434"
# Generous: local models on a CPU box can be slow to first token.
REQUEST_TIMEOUT_S = 300.0


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


def _call_ollama(host: str, model: str, prompt: str) -> dict:
    """POST a single user turn to Ollama's /api/chat and return the parsed JSON.

    Raises urllib/JSON errors to the caller, which turns them into a failed
    ATPResponse — this helper never swallows them so it stays unit-testable.
    """
    url = f"{host.rstrip('/')}/api/chat"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def build_response(request: dict, raw: dict) -> dict:
    """Normalize an Ollama /api/chat payload into an ATPResponse dict."""
    task_id = request.get("task_id", "")
    text = raw["message"]["content"]
    # Token counts are present on a non-streamed /api/chat response, but be
    # defensive: default each to 0 and report None for the total only when
    # NEITHER field is present (so "unknown" stays distinguishable from 0).
    in_tok = raw.get("prompt_eval_count")
    out_tok = raw.get("eval_count")
    if in_tok is None and out_tok is None:
        total: int | None = None
    else:
        total = (in_tok or 0) + (out_tok or 0)
    return {
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
            "total_tokens": total,
            "input_tokens": in_tok or 0,
            "output_tokens": out_tok or 0,
            # Local inference is free, but report null (unknown) rather than 0.0
            # so the harness never confuses "free" with a measured cost.
            "cost_usd": None,
        },
    }


def main() -> int:
    """Read ATPRequest from stdin, call Ollama, emit ATPResponse."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        # Invalid/empty stdin must still produce a contract-shaped failed
        # response, not crash the shim.
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        return _fail(task_id, "OLLAMA_MODEL not set")
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)

    prompt = build_prompt(request, get_envelope("review"))
    try:
        raw = _call_ollama(host, model, prompt)
        response = build_response(request, raw)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(task_id, f"ollama request error: {exc}")
    except (ValueError, TypeError, KeyError) as exc:
        return _fail(task_id, f"ollama response error: {exc}")

    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
