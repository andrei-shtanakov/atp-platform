#!/usr/bin/env python3
"""Shared OpenAI-compatible spawner logic for ATP's CLI adapter.

Provider-agnostic core: read an ATPRequest JSON from stdin, call an
OpenAI-compatible ``/v1/chat/completions`` endpoint, normalize the result into
an ATPResponse JSON on stdout. A thin per-provider shim calls ``run(prefix,
default_host)`` — the prefix selects the ``{prefix}_API_KEY`` / ``{prefix}_HOST``
/ ``{prefix}_MODEL`` env vars. Mirrors ``deepseek_shim.py`` (kept separate so
working Tier-1 code is not churned). Stdlib only — no new dependency.
"""

import json
import os
import sys
import urllib.error
import urllib.request

from atp_method.envelopes import build_prompt, get_envelope

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


def _call(host: str, key: str, model: str, prompt: str) -> dict:
    """POST a single user turn and return the parsed JSON (errors propagate)."""
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
    """Normalize an OpenAI-compatible chat-completions payload into ATPResponse."""
    task_id = request.get("task_id", "")
    text = raw["choices"][0]["message"]["content"]
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
            "cost_usd": None,
        },
    }


def run(prefix: str, default_host: str) -> int:
    """Drive one OpenAI-compatible provider selected by ``prefix`` env vars."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        return _fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    key = os.environ.get(f"{prefix}_API_KEY")
    if not key:
        return _fail(task_id, f"{prefix}_API_KEY not set")
    model = os.environ.get(f"{prefix}_MODEL")
    if not model:
        return _fail(task_id, f"{prefix}_MODEL not set")
    host = os.environ.get(f"{prefix}_HOST", default_host)

    prompt = build_prompt(request, get_envelope("review"))
    try:
        raw = _call(host, key, model, prompt)
        response = build_response(request, raw)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(task_id, f"{prefix.lower()} request error: {exc}")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return _fail(task_id, f"{prefix.lower()} response error: {exc}")

    sys.stdout.write(json.dumps(response))
    return 0
