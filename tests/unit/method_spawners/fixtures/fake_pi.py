#!/usr/bin/env python3
"""Deterministic stand-in for `pi -p --mode json`.

Records the invocation (argv + cwd) to ATP_FAKE_PI_LOG when set, then emits
pi-style JSONL: a message_end event whose content text is either the canned
review findings or (for corpus-mode prompts) a citation carrying an ABSOLUTE
path, so the shim's corpus-relative normalization is observable in tests.
"""

import json
import os
import sys

argv = sys.argv[1:]

log = os.environ.get("ATP_FAKE_PI_LOG")
if log:
    with open(log, "w", encoding="utf-8") as fh:
        json.dump({"argv": argv, "cwd": os.getcwd()}, fh)

prompt = " ".join(argv)
if "Read-only corpus" in prompt:
    text = json.dumps(
        {"citations": [{"path": os.path.join(os.getcwd(), "policy-current.md")}]}
    )
else:
    text = "[]"

print(
    json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "usage": {"input": 12, "output": 3, "totalTokens": 15},
            },
        }
    )
)
