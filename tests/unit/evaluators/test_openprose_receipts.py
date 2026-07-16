"""Unit tests for openprose_receipts: canonical form, ledger reader, checker.

Contract: method/contract/openprose/receipt.md (openprose.receipt.v1).
"""

import pytest

from atp.evaluators.openprose_receipts.canonical import canonical_json, content_hash


class TestCanonicalJson:
    def test_sorts_object_keys(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_no_whitespace_nested_containers(self) -> None:
        value = {"outer": {"z": [1, 2], "a": None}, "flag": True}
        assert canonical_json(value) == b'{"flag":true,"outer":{"a":null,"z":[1,2]}}'

    def test_utf8_not_ascii_escaped(self) -> None:
        assert canonical_json({"k": "приём"}) == '{"k":"приём"}'.encode()

    def test_string_json_escapes_preserved(self) -> None:
        assert canonical_json('a"b\n') == b'"a\\"b\\n"'

    def test_rejects_float(self) -> None:
        with pytest.raises(ValueError):
            canonical_json({"tokens": 1.5})

    def test_rejects_nan_and_infinity(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError):
                canonical_json(bad)

    def test_bool_is_not_an_int(self) -> None:
        assert canonical_json(True) == b"true"

    def test_rejects_non_string_keys(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({1: "x"})


class TestContentHash:
    def test_excludes_content_hash_field_itself(self) -> None:
        receipt = {"v": "openprose.receipt.v1", "prev": None}
        h = content_hash(receipt)
        assert h == content_hash({**receipt, "content_hash": "sha256:bogus"})
        assert h.startswith("sha256:")
        assert len(h) == 71  # "sha256:" + 64 hex

    def test_unknown_fields_participate_in_hash(self) -> None:
        base = {"v": "openprose.receipt.v1", "prev": None}
        assert content_hash(base) != content_hash({**base, "future_field": 1})

    def test_hash_covers_prev(self) -> None:
        base = {"v": "openprose.receipt.v1", "prev": None}
        assert content_hash(base) != content_hash({**base, "prev": "sha256:aa"})
