#!/usr/bin/env python3
"""Deterministic stand-in for `claude -p --output-format json`.

Ignores the prompt; emits the canned envelope the real CLI produces so shim
tests run offline. The review text echoes a fixed finding so the shim's
normalization is assertable.
"""

import json
import sys

_ = sys.argv
_ = sys.stdin.read() if not sys.stdin.isatty() else ""

print(
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": "SEC-011 violation at app.py:12 (severity: critical) — raw SQL "
            "built via f-string. Fix: use a parameterized query.",
            "total_cost_usd": 0.0123,
            "usage": {"input_tokens": 800, "output_tokens": 120},
            "num_turns": 1,
        }
    )
)
