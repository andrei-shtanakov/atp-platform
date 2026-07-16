"""Canonical JSON + content hashing for openprose.receipt.v1.

Implements the contract's canonical form (method/contract/openprose/receipt.md):
object keys sorted, no whitespace, UTF-8 strings not ASCII-escaped, integers
only. Written against the contract text; open-prose's reference canonical.py
was consulted, not copied. stdlib-only.
"""

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Serialize ``value`` to the contract's canonical byte form.

    Raises ValueError on floats/NaN/Infinity (invalid in a receipt — they break
    hash portability) and TypeError on non-JSON types or non-string keys.
    """
    # bool is a subclass of int — it must be checked first.
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if value is None:
        return b"null"
    if isinstance(value, int):
        return str(value).encode()
    if isinstance(value, float):
        raise ValueError("floats are invalid in a canonical receipt")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).encode()
    if isinstance(value, list):
        return b"[" + b",".join(canonical_json(v) for v in value) + b"]"
    if isinstance(value, dict):
        parts: list[bytes] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("receipt object keys must be strings")
            parts.append(canonical_json(key) + b":" + canonical_json(value[key]))
        return b"{" + b",".join(parts) + b"}"
    raise TypeError(f"unsupported type in receipt: {type(value).__name__}")


def content_hash(receipt: dict[str, Any]) -> str:
    """Chain identity: sha256 over the canonical form sans ``content_hash``.

    The hash covers every other field — including ``prev`` (each receipt
    commits to the whole chain behind it) and unknown fields (append-frozen:
    ignored semantically, hashed as received).
    """
    body = {k: v for k, v in receipt.items() if k != "content_hash"}
    return "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()
