#!/usr/bin/env python3
"""deepseek spawner shim for ATP's CLI adapter — DeepSeek API (OpenAI-compatible).

Contract: read an ATPRequest JSON from stdin, call DeepSeek's OpenAI-compatible
chat-completions endpoint, normalize the result into an ATPResponse JSON on
stdout. agent_id is set by the run wiring, not here.

Why this exists (R-07 agent matrix):
  - DeepSeek ships NO product CLI, so this is an API-direct agent — a LABELED API
    BASELINE row (like anthropic_api). It must never substitute a CLI agent_id in
    arbiter routing; it widens the model spread, not the set of routable harnesses.

Guardrails (so this stays a fair matrix row):
  - SAME prompt envelope as every other shim. All shims draw from the single
    shared source (`atp_method.envelopes`), so they cannot drift apart.
  - NO assistant prefill and NO system prompt — the envelope is the whole turn.
  - cost_usd is reported NULL (unknown): the raw API returns token usage but no
    cost number, and the harness treats null as unknown (never confused with 0).

Stdlib only — no new dependency: the API is reached with `urllib.request`.

Needs `DEEPSEEK_API_KEY` (required; the run harness skips this agent with a clear
message when it is missing). Model from `DEEPSEEK_MODEL` (default deepseek-chat),
host base from `DEEPSEEK_HOST` (default https://api.deepseek.com).
"""

import json
import os
import sys
import urllib.error
import urllib.request

from atp_method.envelopes import build_prompt, get_envelope

DEFAULT_HOST = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
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


def _call_deepseek(host: str, key: str, model: str, prompt: str) -> dict:
    """POST a single user turn to DeepSeek's chat/completions and return the JSON.

    Raises urllib/JSON errors to the caller, which turns them into a failed
    ATPResponse — this helper never swallows them so it stays unit-testable.
    """
    url = f"{host.rstrip('/')}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def build_response(request: dict, raw: dict) -> dict:
    """Normalize a DeepSeek chat-completions payload into an ATPResponse dict."""
    task_id = request.get("task_id", "")
    text = raw["choices"][0]["message"]["content"]
    # Usage is present on a non-streamed response, but be defensive: a missing
    # count stays None (unknown) rather than collapsing to 0. The total is None
    # only when BOTH fields are absent (matching the local-model shim semantics).
    usage = raw.get("usage") or {}
    in_tok = usage.get("prompt_tokens")
    out_tok = usage.get("completion_tokens")
    total = (
        (in_tok or 0) + (out_tok or 0)
        if in_tok is not None or out_tok is not None
        else None
    )
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
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            # Raw API returns no cost number; report null (unknown) rather than 0.
            "cost_usd": None,
        },
    }


def main() -> int:
    """Read ATPRequest from stdin, call DeepSeek, emit ATPResponse."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        # Invalid/empty stdin must still produce a contract-shaped failed
        # response, not crash the shim.
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return _fail(task_id, "DEEPSEEK_API_KEY not set")
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    host = os.environ.get("DEEPSEEK_HOST", DEFAULT_HOST)

    prompt = build_prompt(request, get_envelope("review"))
    try:
        raw = _call_deepseek(host, key, model, prompt)
        response = build_response(request, raw)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(task_id, f"deepseek request error: {exc}")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return _fail(task_id, f"deepseek response error: {exc}")

    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
