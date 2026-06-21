#!/usr/bin/env python3
"""anthropic_api spawner shim for ATP's CLI adapter — the RAW-API ablation baseline.

Contract: read an ATPRequest JSON from stdin, call the Anthropic Messages API
directly (no Claude Code product harness), normalize the result into an
ATPResponse JSON on stdout. agent_id is set by the run wiring, not here.

Why this exists (R-07 Phase-1b Ticket B, "product harness vs raw API"):
  - the CLI shim reaches the model THROUGH `claude -p`, which carries Claude
    Code's system prompt / tool scaffolding — a *product harness*.
  - this shim hits the bare Messages API with ONLY the shared review envelope as
    the user turn — the *raw model*.
The two are deliberately NOT equalizable: that gap is the thing being measured.

Guardrails (so this stays a fair baseline, not "a different agent"):
  - SAME model + SAME prompt envelope as the CLI shim. Both draw from the single
    shared source (`atp_method.envelopes`), so they cannot drift apart.
  - NO assistant prefill and NO system prompt — prefilling the agent-under-test
    would make the ablation meaningless.
  - `anthropic_api` is a LABELED BASELINE row only; it must never substitute the
    CLI `agent_id` in arbiter routing.

Needs `ANTHROPIC_API_KEY` (absent in some envs — the run harness skips this agent
with a clear message when the key is missing).
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Pull the pinned model + prompt envelope from the single shared source so the
# only difference between this shim and the CLI shim is harness-vs-raw-API.
from atp_method.envelopes import DEFAULT_MODEL, build_prompt, get_envelope

MODEL = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
MAX_TOKENS = int(os.environ.get("API_MAX_TOKENS", "4096"))
MAX_TOOL_ITERATIONS = int(os.environ.get("ANTHROPIC_TOOL_MAX_ITERATIONS", "8"))
DEBUG_IO_DIR = os.environ.get("ATP_METHOD_DEBUG_IO_DIR")
DEBUG_IO_TIMESTAMP = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _safe_task_id(task_id: str) -> str:
    """Render a task id as one filesystem path segment."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id)
    return safe or "unknown-task"


def _debug_write_text(task_id: str, name: str, text: str) -> None:
    """Write debug text when ATP_METHOD_DEBUG_IO_DIR is set."""
    if not DEBUG_IO_DIR:
        return
    root = Path(DEBUG_IO_DIR)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{DEBUG_IO_TIMESTAMP}-{_safe_task_id(task_id)}-{name}"
    (root / filename).write_text(text, encoding="utf-8")


def _debug_write_json(task_id: str, name: str, data: object) -> None:
    """Write debug JSON when ATP_METHOD_DEBUG_IO_DIR is set."""
    _debug_write_text(
        task_id,
        name,
        json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n",
    )


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


def _tool_enabled(request: dict) -> bool:
    constraints = request.get("constraints") or {}
    context = request.get("context") or {}
    allowed = constraints.get("allowed_tools") or []
    return "file_read" in allowed and bool(context.get("tools_endpoint"))


def _file_read_tool() -> dict:
    return {
        "name": "file_read",
        "description": "Read one UTF-8 text or markdown file from the corpus.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }


def _block_to_dict(block) -> dict:  # type: ignore[no-untyped-def]
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    return {"type": str(block_type or "unknown")}


def _extract_text(content) -> str:  # type: ignore[no-untyped-def]
    return "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )


def _usage_tokens(msg) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    usage = getattr(msg, "usage", None)
    return (
        getattr(usage, "input_tokens", None) or 0,
        getattr(usage, "output_tokens", None) or 0,
    )


def _message_debug_snapshot(msg) -> dict:  # type: ignore[no-untyped-def]
    """JSON-serializable view of an Anthropic message object."""
    in_tok, out_tok = _usage_tokens(msg)
    return {
        "content": [_block_to_dict(block) for block in getattr(msg, "content", [])],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def _call_tool_sync(endpoint: str, tool: str, input_data: dict, task_id: str) -> dict:
    from method.spawners._tool_client import call_tool

    return asyncio.run(call_tool(endpoint, tool, input_data, task_id=task_id))


def _emit_tool_event(task_id: str, tool: str, input_data: dict, response: dict) -> None:
    payload = {
        "tool": tool,
        "input": input_data,
        "status": response.get("status"),
    }
    if "output" in response:
        payload["output"] = response.get("output")
    if "error" in response:
        payload["error"] = response.get("error")
    sys.stderr.write(
        json.dumps(
            {
                "event_type": "tool_call",
                "task_id": task_id,
                "payload": payload,
            }
        )
        + "\n"
    )


def _emit_success(
    task_id: str, text: str, input_tokens: int, output_tokens: int
) -> int:
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
            "total_tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            # No cost field on the raw API response; the CLI shim gets real cost
            # from `claude --output-format json`. Left null for the baseline.
            "cost_usd": None,
        },
    }
    _debug_write_text(task_id, "final_output.txt", text)
    _debug_write_json(task_id, "atp_response.json", response)
    sys.stdout.write(json.dumps(response))
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

    prompt = build_prompt(request, get_envelope("review"))
    _debug_write_text(task_id, "prompt.txt", prompt)
    try:
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": prompt}]
        if _tool_enabled(request):
            text, in_tok, out_tok = _run_tool_loop(client, request, messages)
            return _emit_success(task_id, text, in_tok, out_tok)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )
    except Exception as exc:  # noqa: BLE001 - any API error becomes a failed run
        return _fail(task_id, f"anthropic API error: {exc}")

    _debug_write_json(task_id, "raw_response.json", _message_debug_snapshot(msg))
    in_tok, out_tok = _usage_tokens(msg)
    return _emit_success(task_id, _extract_text(msg.content), in_tok, out_tok)


def _run_tool_loop(
    client,
    request: dict,
    messages: list[dict],  # type: ignore[no-untyped-def]
) -> tuple[str, int, int]:
    """Run a bounded Anthropic tool loop for file_read."""
    task_id = request.get("task_id", "")
    endpoint = (request.get("context") or {}).get("tools_endpoint")
    input_tokens = 0
    output_tokens = 0
    debug_steps = []
    for _ in range(MAX_TOOL_ITERATIONS):
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=[_file_read_tool()],
        )
        step_debug = {"assistant": _message_debug_snapshot(msg), "tool_results": []}
        in_tok, out_tok = _usage_tokens(msg)
        input_tokens += in_tok
        output_tokens += out_tok
        tool_blocks = [
            block for block in msg.content if getattr(block, "type", None) == "tool_use"
        ]
        if not tool_blocks:
            debug_steps.append(step_debug)
            _debug_write_json(task_id, "raw_tool_loop.json", {"steps": debug_steps})
            return _extract_text(msg.content), input_tokens, output_tokens

        messages.append(
            {
                "role": "assistant",
                "content": [_block_to_dict(block) for block in msg.content],
            }
        )
        results = []
        for block in tool_blocks:
            tool_name = getattr(block, "name", "")
            tool_input = getattr(block, "input", {}) or {}
            if tool_name != "file_read":
                tool_response = {
                    "status": "error",
                    "error": f"unsupported tool: {tool_name}",
                }
            else:
                try:
                    tool_response = _call_tool_sync(
                        endpoint, tool_name, tool_input, task_id
                    )
                except Exception as exc:  # noqa: BLE001
                    tool_response = {"status": "error", "error": str(exc)}
            _emit_tool_event(task_id, tool_name, tool_input, tool_response)
            step_debug["tool_results"].append(
                {
                    "tool": tool_name,
                    "input": tool_input,
                    "response": tool_response,
                }
            )
            content = (
                json.dumps(tool_response.get("output"))
                if tool_response.get("status") == "success"
                else json.dumps({"error": tool_response.get("error")})
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "id", ""),
                    "content": content,
                    "is_error": tool_response.get("status") != "success",
                }
            )
        messages.append({"role": "user", "content": results})
        debug_steps.append(step_debug)
    _debug_write_json(task_id, "raw_tool_loop.json", {"steps": debug_steps})
    raise RuntimeError(f"tool loop exceeded {MAX_TOOL_ITERATIONS} iterations")


if __name__ == "__main__":
    raise SystemExit(main())
