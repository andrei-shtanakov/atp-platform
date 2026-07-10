#!/usr/bin/env python3
"""Deterministic stand-in for `python -m robin.agent "<question>"`.

Prints Robin-shaped output (Q:/A:/(cost $)/Grounding sources) so the robin shim
can be tested offline. The question is the last argv element (after `-m robin.agent`).

Env switches:
  FAKE_ROBIN_FAIL=1        — exit non-zero with a canned stderr;
  FAKE_ROBIN_RETRIEVE=1    — retrieve-only shape: no A:/cost, zero sources.
"""

import os
import sys


def main() -> int:
    if os.environ.get("FAKE_ROBIN_FAIL") == "1":
        sys.stderr.write("robin exploded: no vault mounted\n")
        return 3
    question = sys.argv[-1] if len(sys.argv) > 1 else "?"
    print(f"Q: {question}")
    if os.environ.get("FAKE_ROBIN_RETRIEVE") == "1":
        print("(retrieve-only: no ANTHROPIC_API_KEY — showing grounding sources)")
        print("Grounding sources (0):")
        return 0
    print()
    print("A: The SSOT is atp-platform/method/agents-catalog.toml.")
    print()
    print("(cost $0.0123)")
    print("Grounding sources (2):")
    print("  authored/decisions/2026-07-01-adr-eco-003-agent-catalog.md:22: catalog")
    print("  Maestro@a1b2c3d: 2026-07-10 t: routing fix (1 file changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
