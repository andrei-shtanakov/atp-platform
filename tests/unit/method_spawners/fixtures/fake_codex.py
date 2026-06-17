#!/usr/bin/env python3
"""Deterministic stand-in for `codex exec ... --output-last-message <FILE>`.

Ignores the prompt; honors --output-last-message by WRITING a canned review text
to that file (mirroring how the real codex writes its final assistant message),
then exits 0. Lets the codex_cli shim be tested offline.

Set FAKE_CODEX_FAIL=1 to exercise the failure path: exit non-zero, write nothing
to the message file, and emit a canned error to stderr.
"""

import os
import sys

_FINDINGS = (
    '[{"rule_id": "sql-injection", "file": "app.py",'
    ' "anchor": "query = f\\"SELECT",'
    ' "severity": "critical", "fix": "use a parameterized query"}]'
)


def _output_path(argv: list[str]) -> str | None:
    """Return the value following --output-last-message, if present."""
    for i, arg in enumerate(argv):
        if arg == "--output-last-message" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main() -> int:
    argv = sys.argv[1:]
    if os.environ.get("FAKE_CODEX_FAIL") == "1":
        sys.stderr.write("fake codex: simulated failure\n")
        return 3
    out_path = _output_path(argv)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(_FINDINGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
