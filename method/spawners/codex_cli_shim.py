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

Token usage is captured from `--json`: codex emits JSONL events on stdout,
including a `turn.completed` event carrying a `usage` block with input/output
token counts. We parse those for budget-aware routing. Cost ($) is reported NULL:
codex exposes no dollar figure, so we don't fabricate it (null = unknown).
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


def _parse_usage(
    stdout: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Token counts from codex --json JSONL; each None if absent.

    Returns (input_tokens, output_tokens, cached_input_tokens,
    reasoning_output_tokens). Codex emits a `turn.completed` event whose `usage`
    block carries these (e.g. {"type": "turn.completed", "usage":
    {"input_tokens": N, "output_tokens": M, "cached_input_tokens": C,
    "reasoning_output_tokens": R}}). We scan all events and take the last counts.

    `cached_input_tokens` is a subset of `input_tokens` and
    `reasoning_output_tokens` a subset of `output_tokens` (per OpenAI usage
    convention) — both are breakdowns, surfaced for transparency but NOT added to
    any total by callers.
    """
    in_tok = out_tok = cached_in = reasoning_out = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        info = ev.get("usage") or ev.get("info") or ev
        if isinstance(info, dict) and (
            "input_tokens" in info or "output_tokens" in info
        ):
            in_tok = info.get("input_tokens", in_tok)
            out_tok = info.get("output_tokens", out_tok)
            cached_in = info.get("cached_input_tokens", cached_in)
            reasoning_out = info.get("reasoning_output_tokens", reasoning_out)
    return in_tok, out_tok, cached_in, reasoning_out


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
            "--json",
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

        # The last-message file is produced by an external CLI; read it as UTF-8
        # with replacement so non-UTF-8 bytes can't raise before we emit a
        # contract-shaped response.
        message = (
            Path(tmp_path).read_text(encoding="utf-8", errors="replace")
            if Path(tmp_path).exists()
            else ""
        )
        if proc.returncode != 0 or not message.strip():
            err = proc.stderr.decode(errors="replace")[:2000]
            return _fail(
                task_id,
                f"codex exec failed (rc={proc.returncode}): {err}",
            )

        in_tok, out_tok, cached_in, reasoning_out = _parse_usage(
            proc.stdout.decode(errors="replace")
        )
        # `cached_input_tokens` is a subset of `input_tokens` (ignored in total);
        # `output_tokens` already includes reasoning per OpenAI usage convention,
        # so `reasoning_output_tokens` is surfaced for transparency but NOT
        # re-added. total = input + output, and ONLY when both are known
        # (else None) — a partial sum would be misleading ("both known else
        # unknown"). codex's turn.completed.usage carries both together.
        total = in_tok + out_tok if in_tok is not None and out_tok is not None else None
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
                # Token counts come from codex's --json `turn.completed` usage
                # event; cost ($) is not exposed by codex, so it stays null.
                # cached_input/reasoning_output are subset breakdowns (not added
                # to total), surfaced for future budget calc transparency.
                "total_tokens": total,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_input_tokens": cached_in,
                "reasoning_output_tokens": reasoning_out,
                "cost_usd": None,
            },
        }
        sys.stdout.write(json.dumps(response))
        return 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
