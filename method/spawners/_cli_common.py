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
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from atp_method.envelopes import build_prompt, get_envelope

REQUEST_TIMEOUT_S = 600.0


def _dump_raw(task_id: str, stdout: bytes | None, stderr: bytes | None) -> None:
    """Persist raw subprocess streams for post-hoc diagnosis.

    Paid runs are expensive; the streams are the only evidence when a run
    comes back empty (e.g. the provider stalls and the hard timeout reaps
    the process). No-op unless ``ATP_SHIM_RAW_DIR`` is set (the harness
    points it at ``<out-dir>/raw/<agent>``). Diagnostics must never break
    the run, so any OSError is swallowed.
    """
    raw_dir = os.environ.get("ATP_SHIM_RAW_DIR")
    if not raw_dir:
        return
    try:
        out = Path(raw_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", task_id or "unknown")
        (out / f"{stem}.stdout").write_bytes(stdout or b"")
        (out / f"{stem}.stderr").write_bytes(stderr or b"")
    except OSError:
        pass


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


def corpus_workspace(request: dict) -> str | None:
    """Materialized corpus root for a read_only_corpus run, else None.

    Path A (CLI corpus grounding): the corpus preparer materializes the
    verified corpus and sets ``context.workspace_path``. A native-tools CLI
    runs with cwd at that root instead of ATP's HTTP file_read endpoint.
    Both markers must be present — workspace_path alone may mean any future
    workspace-carrying run mode.
    """
    task = request.get("task") or {}
    input_data = task.get("input_data") or {}
    if input_data.get("run_mode") != "read_only_corpus":
        return None
    context = request.get("context") or {}
    workspace = context.get("workspace_path")
    return str(workspace) if workspace else None


def normalize_citation_paths(text: str, workspace: str) -> str:
    """Rewrite absolute corpus paths in model output to corpus-relative.

    The citation_grounding grader keys on corpus-relative paths
    (``policy-current.md``); a CLI reading files under cwd may cite the
    absolute path instead. String-level replace keeps this format-agnostic
    (works inside JSON strings without parsing the whole artifact).
    """
    prefix = workspace.rstrip("/") + "/"
    return text.replace(prefix, "")


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
    corpus_args: list[str] | None = None,
    corpus_env: dict[str, str] | None = None,
) -> int:
    """Drive one CLI tool. ``argv(binary_tokens, model, prompt) -> list[str]``;
    ``parse_output(stdout) -> (text, input_tokens, output_tokens)``.

    Path A corpus mode: on a ``read_only_corpus`` run the subprocess cwd is
    the materialized corpus root, ``corpus_args`` (the CLI's read-only
    confinement flags) are inserted just before the prompt — argv callbacks
    MUST place the prompt as the final element — and ``corpus_env`` entries
    override the subprocess environment (for CLIs whose confinement is
    config-based rather than flag-based). Citation paths in the parsed
    output are rewritten to corpus-relative.
    """
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

    workspace = corpus_workspace(request)
    try:
        prompt = build_prompt(request, get_envelope("review"))
        cmd = argv(binary, model_arg(model, default_provider), prompt)
        if workspace and corpus_args:
            cmd = [*cmd[:-1], *corpus_args, cmd[-1]]
    except Exception as exc:  # noqa: BLE001 — contract: any error → failed
        return fail(task_id, f"{binary[0]} command build error: {exc}")
    sub_env = {**os.environ, **corpus_env} if workspace and corpus_env else None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=REQUEST_TIMEOUT_S,
            cwd=workspace or None,
            env=sub_env,
        )
    except subprocess.TimeoutExpired as exc:
        # Partial streams are the only evidence of a hung provider request.
        _dump_raw(task_id, exc.stdout, exc.stderr)
        return fail(task_id, f"{binary[0]} timed out after {REQUEST_TIMEOUT_S}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return fail(task_id, f"{binary[0]} invocation error: {exc}")

    _dump_raw(task_id, proc.stdout, proc.stderr)
    if proc.returncode != 0:
        return fail(
            task_id,
            f"{binary[0]} failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:2000]}",
        )
    try:
        text, in_tok, out_tok = parse_output(proc.stdout.decode(errors="replace"))
    except Exception as exc:  # noqa: BLE001 — contract: any error → failed
        return fail(
            task_id,
            f"{binary[0]} output parse error: {exc} (stdout {len(proc.stdout)} bytes)",
        )
    if not text.strip():
        # Keep the stderr tail: without it an empty run is undiagnosable
        # (rate-limit vs stall vs tool-loop break all look identical).
        stderr_tail = proc.stderr.decode(errors="replace")[-500:]
        return fail(
            task_id,
            f"{binary[0]} produced no output text; stderr tail: {stderr_tail!r}",
        )
    if workspace:
        # The CLI reads files under cwd and may cite absolute paths; the
        # citation_grounding grader keys on corpus-relative ones.
        text = normalize_citation_paths(text, workspace)

    sys.stdout.write(json.dumps(build_response(task_id, text, in_tok, out_tok)))
    return 0
