#!/usr/bin/env python3
"""Shared CLI-spawner logic for ATP's CLI adapter (Tier-2 agents).

A thin per-tool shim supplies an argv template + a JSONL parser; this module
runs the subprocess with a hard timeout and normalizes the result into the
ATPResponse contract on stdout. Any error (bad stdin, missing model, non-zero
exit, timeout, empty output) becomes a status:"failed" response — never a crash.
Mirrors codex_cli_shim.py; stdlib + ``atp_method`` (workspace), no new dependency.
"""

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable

from atp_method.envelopes import build_prompt, get_envelope

REQUEST_TIMEOUT_S = 600.0


def fail(task_id: str, error: str) -> int:
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


def build_response(
    task_id: str, text: str, in_tok: int | None, out_tok: int | None
) -> dict:
    """Normalize a CLI run into a completed ATPResponse dict."""
    total = in_tok + out_tok if in_tok is not None and out_tok is not None else None
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


def model_arg(model: str, default_provider: str) -> str:
    """A bare model id gets the tool's provider prefix; an already
    provider-qualified id (contains '/') passes through unchanged."""
    return model if "/" in model else f"{default_provider}/{model}"


def run(
    *,
    bin_env: str,
    default_bin: str,
    model_env: str,
    default_provider: str,
    argv: Callable[[list[str], str, str], list[str]],
    parse_output: Callable[[str], tuple[str, int | None, int | None]],
) -> int:
    """Drive one CLI tool. ``argv(binary_tokens, model, prompt) -> list[str]``;
    ``parse_output(stdout) -> (text, input_tokens, output_tokens)``."""
    raw_in = sys.stdin.read()
    try:
        request = json.loads(raw_in)
    except (ValueError, TypeError) as exc:
        return fail("", f"invalid ATPRequest JSON on stdin: {exc}")
    task_id = request.get("task_id", "")

    model = os.environ.get(model_env)
    if not model:
        return fail(task_id, f"{model_env} not set")
    binary = shlex.split(os.environ.get(bin_env, default_bin)) or [default_bin]

    prompt = build_prompt(request, get_envelope("review"))
    cmd = argv(binary, model_arg(model, default_provider), prompt)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=REQUEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return fail(task_id, f"{binary[0]} timed out after {REQUEST_TIMEOUT_S}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return fail(task_id, f"{binary[0]} invocation error: {exc}")

    if proc.returncode != 0:
        return fail(
            task_id,
            f"{binary[0]} failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:2000]}",
        )
    text, in_tok, out_tok = parse_output(proc.stdout.decode(errors="replace"))
    if not text.strip():
        return fail(task_id, f"{binary[0]} produced no output text")

    sys.stdout.write(json.dumps(build_response(task_id, text, in_tok, out_tok)))
    return 0
