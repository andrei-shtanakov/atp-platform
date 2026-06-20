#!/usr/bin/env python3
"""qwen (DashScope) spawner shim — OpenAI-compatible endpoint.

API baseline row: never substitutes a CLI agent in routing. Reads
QWEN_API_KEY / QWEN_HOST / QWEN_MODEL.
"""

from _openai_compat import run  # pyrefly: ignore[missing-import]

if __name__ == "__main__":
    raise SystemExit(
        run("QWEN", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    )
