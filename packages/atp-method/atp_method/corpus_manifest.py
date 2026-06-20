"""Helpers for generating stable corpus SHA-256 manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from atp_method.corpus import TEXT_SUFFIXES, normalize_lf


def generate_manifest(root: Path, paths: list[str] | None = None) -> str:
    """Return stable ``manifest.sha256`` content for text corpus files."""
    root = root.resolve()
    selected = (
        [root / path for path in paths]
        if paths is not None
        else sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        )
    )
    lines: list[str] = []
    for path in selected:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"manifest path escapes root: {path}")
        rel = resolved.relative_to(root).as_posix()
        text = normalize_lf(resolved.read_text(encoding="utf-8"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lines.append(f"{digest}  {rel}")
    return "\n".join(sorted(lines)) + ("\n" if lines else "")
