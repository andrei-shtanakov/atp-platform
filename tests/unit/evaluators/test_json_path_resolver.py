"""Single-node JSONPath subset resolver."""

import pytest

from atp.evaluators.json_path.resolver import InvalidPath, resolve


def test_root() -> None:
    assert resolve({"a": 1}, "$") == (True, {"a": 1})


def test_key() -> None:
    assert resolve({"a": 1}, "$.a") == (True, 1)


def test_nested_key_and_index() -> None:
    data = {"reqs": [{"d": None}, {"d": "x"}]}
    assert resolve(data, "$.reqs[1].d") == (True, "x")
    assert resolve(data, "$.reqs[0].d") == (True, None)


def test_missing_key_or_index_not_found() -> None:
    assert resolve({"a": 1}, "$.b") == (False, None)
    assert resolve({"a": [1]}, "$.a[5]") == (False, None)


def test_root_index() -> None:
    assert resolve([10, 20, 30], "$[1]") == (True, 20)


def test_unsupported_syntax_raises() -> None:
    for bad in ["a", "$.a[*]", "$..a", "$.a[?(@.x)]", "$.a.", "$[0]extra", "$.a[-1]"]:
        with pytest.raises(InvalidPath):
            resolve({"a": 1}, bad)
