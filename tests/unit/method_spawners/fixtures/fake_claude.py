#!/usr/bin/env python3
"""Deterministic stand-in for `claude -p --output-format json`.

Ignores the prompt; emits the canned envelope the real CLI produces so shim
tests run offline. The review text echoes a fixed finding so the shim's
normalization is assertable.
"""

import json
import os
import sys

_ = sys.stdin.read() if not sys.stdin.isatty() else ""

# Record the invocation (argv + cwd) so shim tests can assert corpus
# confinement flags and working directory without a real CLI.
_log = os.environ.get("ATP_FAKE_CLAUDE_LOG")
if _log:
    with open(_log, "w", encoding="utf-8") as fh:
        json.dump({"argv": sys.argv[1:], "cwd": os.getcwd()}, fh)

_FINDINGS = (
    '[{"rule_id": "sql-injection", "file": "app.py",'
    ' "anchor": "query = f\\"SELECT",'
    ' "severity": "critical", "fix": "use a parameterized query"}]'
)

# Corpus-mode prompts carry the envelope's corpus block; emit a citation
# with an ABSOLUTE path (as a native-tools CLI would) so the shim's
# corpus-relative normalization is observable in tests.
_prompt = " ".join(sys.argv[1:])
if "Read-only corpus" in _prompt:
    _result = json.dumps(
        {"citations": [{"path": os.path.join(os.getcwd(), "policy-current.md")}]}
    )
else:
    _result = _FINDINGS

print(
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": _result,
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
