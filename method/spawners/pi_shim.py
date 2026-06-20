#!/usr/bin/env python3
"""pi (earendil-works) spawner shim — `pi -p --mode json`.

Non-routable CLI row. Reads PI_MODEL (e.g. gpt-5; the shim prefixes the
`openai/` provider, since a bare id routes pi to an unauthed azure provider)
and PI_BIN (default "pi"). pi authenticates via its own session (no key here).
`--no-prompt-templates` keeps the run lean; the shared runner's hard timeout
guards pi's agentic behavior (it can otherwise not terminate).
"""

import json

from _cli_common import run  # pyrefly: ignore[missing-import]


def _argv(binary: list[str], model: str, prompt: str) -> list[str]:
    return [
        *binary,
        "-p",
        "--mode",
        "json",
        "--no-prompt-templates",
        "--model",
        model,
        prompt,
    ]


def _parse(stdout: str) -> tuple[str, int | None, int | None]:
    """Last non-empty assistant message's content text + usage."""
    text = ""
    in_tok: int | None = None
    out_tok: int | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        parts = message.get("content") or []
        joined = "".join(
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        )
        if joined:
            text = joined  # message_end carries the full content
        usage = message.get("usage") or {}
        if usage.get("input") is not None:
            in_tok = usage.get("input")
        if usage.get("output") is not None:
            out_tok = usage.get("output")
    return text, in_tok, out_tok


if __name__ == "__main__":
    raise SystemExit(
        run(
            bin_env="PI_BIN",
            default_bin="pi",
            model_env="PI_MODEL",
            default_provider="openai",
            argv=_argv,
            parse_output=_parse,
        )
    )
