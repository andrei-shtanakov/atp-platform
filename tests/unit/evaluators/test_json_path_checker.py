"""json_path checker: text JSON -> CaseVerdict."""

import json

from atp.evaluators.json_path.checker import json_path_check


def _cfg(**over) -> dict:
    base = {
        "assertions": [
            {"path": "$.requirements[1].deadline", "op": "equals", "expected": None}
        ]
    }
    base.update(over)
    return base


def test_pass_when_assertion_holds() -> None:
    text = json.dumps({"requirements": [{"deadline": "30d"}, {"deadline": None}]})
    v = json_path_check(_cfg(), text)
    assert v.critical_pass is True
    assert v.malformed is False


def test_fail_when_fabricated() -> None:
    text = json.dumps({"requirements": [{"deadline": "30d"}, {"deadline": "soon"}]})
    v = json_path_check(_cfg(), text)
    assert v.critical_pass is False
    assert v.malformed is False


def test_unparseable_is_malformed() -> None:
    v = json_path_check(_cfg(), "not json at all")
    assert v.malformed is True
    assert v.critical_pass is False


def test_none_text_is_malformed() -> None:
    v = json_path_check(_cfg(), None)
    assert v.malformed is True


def test_schema_violation_is_malformed() -> None:
    cfg = _cfg(schema={"type": "object", "required": ["requirements"]})
    v = json_path_check(cfg, json.dumps({"other": 1}))
    assert v.malformed is True


def test_absent_op() -> None:
    cfg = _cfg(assertions=[{"path": "$.requirements[0].deadline", "op": "absent"}])
    text = json.dumps({"requirements": [{"deadline": "x"}]})
    assert json_path_check(cfg, text).critical_pass is False
    text2 = json.dumps({"requirements": [{}]})
    assert json_path_check(cfg, text2).critical_pass is True


def test_contains_string() -> None:
    text = json.dumps({"requirements": [{"deadline": "in 30d"}, {"deadline": None}]})
    cfg = _cfg(
        assertions=[
            {"path": "$.requirements[0].deadline", "op": "contains", "expected": "30d"}
        ]
    )
    assert json_path_check(cfg, text).critical_pass is True
    miss = _cfg(
        assertions=[
            {"path": "$.requirements[0].deadline", "op": "contains", "expected": "zz"}
        ]
    )
    assert json_path_check(miss, text).critical_pass is False


def test_contains_list() -> None:
    text = json.dumps({"tags": ["a", "b", "c"]})
    hit = _cfg(assertions=[{"path": "$.tags", "op": "contains", "expected": "b"}])
    miss = _cfg(assertions=[{"path": "$.tags", "op": "contains", "expected": "z"}])
    assert json_path_check(hit, text).critical_pass is True
    assert json_path_check(miss, text).critical_pass is False


def test_contains_none_expected_no_crash() -> None:
    text = json.dumps({"s": "abc"})
    cfg = _cfg(assertions=[{"path": "$.s", "op": "contains", "expected": None}])
    v = json_path_check(cfg, text)
    assert v.critical_pass is False and v.malformed is False


def test_invalid_schema_is_malformed() -> None:
    import json as _json

    cfg = _cfg(schema={"type": "not_a_real_type"})
    v = json_path_check(
        cfg, _json.dumps({"requirements": [{"deadline": None}, {"deadline": None}]})
    )
    assert v.malformed is True
    assert v.critical_pass is False


def test_empty_or_missing_assertions_is_malformed() -> None:
    import json as _json

    text = _json.dumps({"x": 1})
    assert json_path_check({"assertions": []}, text).malformed is True
    assert json_path_check({}, text).malformed is True


def test_multi_or_bad_path_fails_assertion_not_crash() -> None:
    cfg = _cfg(
        assertions=[{"path": "$.requirements[*]", "op": "equals", "expected": 1}]
    )
    v = json_path_check(cfg, json.dumps({"requirements": []}))
    assert v.critical_pass is False
    assert v.malformed is False
