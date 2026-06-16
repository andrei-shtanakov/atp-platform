"""A deterministic, single-node subset of JSONPath: `$`, `.key`, `[index]`.

No wildcards, recursion, or filters — those produce multi-node matches, which
are ambiguous for a deterministic grader (ADR-007). A path resolves to exactly
one node or it is "not found"; anything outside the grammar raises InvalidPath.
"""

from __future__ import annotations

import re
from typing import Any

_SEGMENT = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)|\[(\d+)\]")


class InvalidPath(ValueError):
    """Raised when a path is outside the supported single-node grammar."""


def resolve(data: Any, path: str) -> tuple[bool, Any]:
    """Resolve ``path`` against ``data``.

    Returns ``(found, value)``. ``found`` is False when a key/index along the
    path is missing. Raises :class:`InvalidPath` for unsupported syntax.
    """
    if not isinstance(path, str) or not path.startswith("$"):
        raise InvalidPath(f"path must start with '$': {path!r}")
    rest = path[1:]
    pos = 0
    current: Any = data
    found = True
    for m in _SEGMENT.finditer(rest):
        if m.start() != pos:
            raise InvalidPath(f"unsupported syntax in path: {path!r}")
        pos = m.end()
        key, idx = m.group(1), m.group(2)
        if not found:
            continue
        if key is not None:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
        else:
            i = int(idx)
            if isinstance(current, list) and 0 <= i < len(current):
                current = current[i]
            else:
                found = False
    if pos != len(rest):
        raise InvalidPath(f"unsupported syntax in path: {path!r}")
    return (found, current if found else None)
