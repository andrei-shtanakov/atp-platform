#!/usr/bin/env python3
"""Deterministic stand-in for `opencode run --format json`.

Records the invocation (argv + cwd + the XDG_CONFIG_HOME it sees) to
ATP_FAKE_OPENCODE_LOG when set, then emits opencode-style JSONL: a `text`
event whose text is either canned findings or (for corpus-mode prompts) a
citation carrying an ABSOLUTE path, so the shim's corpus-relative
normalization is observable in tests.
"""

import json
import os
import sys

argv = sys.argv[1:]

log = os.environ.get("ATP_FAKE_OPENCODE_LOG")
if log:
    # Read the injected config NOW — the shim removes its temp config home
    # after the run, so the test can only see it through this snapshot.
    cfg_home = os.environ.get("XDG_CONFIG_HOME")
    config = None
    if cfg_home:
        cfg_path = os.path.join(cfg_home, "opencode", "opencode.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                config = json.load(fh)
    with open(log, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "argv": argv,
                "cwd": os.getcwd(),
                "xdg_config_home": cfg_home,
                "config": config,
            },
            fh,
        )

prompt = " ".join(argv)
if "Read-only corpus" in prompt:
    text = json.dumps(
        {"citations": [{"path": os.path.join(os.getcwd(), "policy-current.md")}]}
    )
else:
    text = "[]"

print(json.dumps({"type": "text", "part": {"text": text}}))
print(
    json.dumps({"type": "step_finish", "part": {"tokens": {"input": 10, "output": 5}}})
)
