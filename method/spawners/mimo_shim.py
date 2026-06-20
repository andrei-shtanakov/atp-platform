#!/usr/bin/env python3
"""mimo (Xiaomi MiMo) spawner shim — OpenAI-compatible endpoint.

API baseline row (like deepseek): never substitutes a CLI agent in routing.
Reads MIMO_API_KEY / MIMO_HOST / MIMO_MODEL.
"""

from _openai_compat import run  # pyrefly: ignore[missing-import]

if __name__ == "__main__":
    raise SystemExit(run("MIMO", "https://token-plan-sgp.xiaomimimo.com/v1"))
