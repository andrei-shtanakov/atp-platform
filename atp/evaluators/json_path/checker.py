"""json_path checker: parse agent JSON text, optionally schema-validate, run
single-node path assertions, and map to a uniform CaseVerdict.

Mirrors atp/evaluators/findings/checker.py: parse+validate+assert in one
checker, with malformed (not gradeable) distinct from a failed assertion.
"""

import json
from typing import Any

import jsonschema

from atp.core.results import CaseVerdict
from atp.evaluators.json_path.resolver import InvalidPath, resolve

JSON_PATH_CHECKER_VERSION = "json_path@1"


def _assertion_holds(data: Any, assertion: dict[str, Any]) -> bool:
    """Evaluate one single-node assertion. A bad/multi path fails (no crash)."""
    op = assertion.get("op")
    try:
        found, value = resolve(data, assertion.get("path", ""))
    except InvalidPath:
        return False
    if op == "absent":
        return not found
    if not found:
        return False
    if op == "equals":
        return value == assertion.get("expected")
    if op == "contains":
        expected = assertion.get("expected")
        if isinstance(value, str):
            return isinstance(expected, str) and expected in value
        if isinstance(value, list):
            return expected in value
        return False
    return False


def json_path_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Grade JSON text against config.assertions (+ optional config.schema)."""
    if text is None:
        return _malformed("no agent output")
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return _malformed("output is not valid JSON")

    schema = config.get("schema")
    if schema:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            return _malformed(f"schema violation: {e.message}")
        except jsonschema.SchemaError as e:
            return _malformed(f"invalid schema: {e.message}")

    assertions = config.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        # A json_path gate with no assertions is a misconfiguration, not a pass —
        # never let a critical gate vacuously succeed.
        return _malformed("no assertions configured")
    results = [{"assertion": a, "ok": _assertion_holds(data, a)} for a in assertions]
    passed = all(r["ok"] for r in results)
    return CaseVerdict(
        critical_pass=passed,
        malformed=False,
        details={"results": results, "n": len(assertions)},
        grader_version=JSON_PATH_CHECKER_VERSION,
    )


def _malformed(reason: str) -> CaseVerdict:
    return CaseVerdict(
        critical_pass=False,
        malformed=True,
        details={"reason": reason},
        grader_version=JSON_PATH_CHECKER_VERSION,
    )
