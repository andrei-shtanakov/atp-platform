#!/usr/bin/env python3
"""opencode spawner shim — `opencode run --format json` (GLM via opencode).

Non-routable API/CLI baseline row. Reads OPENCODE_MODEL (e.g. glm-5.1; the
shim prefixes the `opencode/` provider) and OPENCODE_BIN (default "opencode").
Auth is opencode's own (operator's OPENCODE_GLM_API_KEY / `opencode auth`).
"""

import json

from _cli_common import run  # pyrefly: ignore[missing-import]


def _argv(binary: list[str], model: str, prompt: str) -> list[str]:
    return [*binary, "run", "--format", "json", "-m", model, prompt]


def _parse(stdout: str) -> tuple[str, int | None, int | None]:
    """Concat `type:text` part.text; tokens from the `step_finish` event."""
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
        if event.get("type") == "text":
            text += (event.get("part") or {}).get("text", "")
        elif event.get("type") == "step_finish":
            tokens = (event.get("part") or {}).get("tokens") or {}
            in_tok = tokens.get("input")
            out_tok = tokens.get("output")
    return text, in_tok, out_tok


if __name__ == "__main__":
    raise SystemExit(
        run(
            bin_env="OPENCODE_BIN",
            default_bin="opencode",
            model_env="OPENCODE_MODEL",
            default_provider="opencode",
            argv=_argv,
            parse_output=_parse,
        )
    )
