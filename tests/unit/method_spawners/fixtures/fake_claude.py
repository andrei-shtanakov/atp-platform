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

_FINDINGS = (
    '[{"rule_id": "sql-injection", "file": "app.py",'
    ' "anchor": "query = f\\"SELECT",'
    ' "severity": "critical", "fix": "use a parameterized query"}]'
)

print(
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": _FINDINGS,
            "total_cost_usd": 0.0123,
            "usage": {
                "input_tokens": 800,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 5000,
                "output_tokens": 120,
            },
            "num_turns": 1,
        }
    )
)
