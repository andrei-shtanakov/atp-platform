"""Minimal git grep wrapper for static architectural guard tests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GrepMatch:
    path: str
    line_number: int
    content: str


def grep_pattern(pattern: str, paths: list[str]) -> list[GrepMatch]:
    """Run git grep and return structured matches. Empty list if no matches."""
    cmd = ["git", "grep", "-n", "-E", pattern, "--", *paths]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("git not available — static guards need git grep")

    if result.returncode == 1:
        return []  # git grep exits 1 when no matches
    if result.returncode != 0:
        raise RuntimeError(f"git grep failed: {result.stderr}")

    matches: list[GrepMatch] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path, lineno, content = parts
        matches.append(GrepMatch(path=path, line_number=int(lineno), content=content))
    return matches
