"""Regression tests for storage._json_safe.

Bedrock (boto3) trace artifacts embed datetime objects; storing them in the
JSON ``data_json`` column used to abort the whole result save with
"Object of type datetime is not JSON serializable".
"""

import json
from datetime import UTC, datetime

from atp.dashboard.storage import _json_safe


def test_json_safe_passes_through_plain_data() -> None:
    value = {"a": 1, "b": ["x", "y"], "c": {"nested": True}}
    assert _json_safe(value) == value


def test_json_safe_handles_none() -> None:
    assert _json_safe(None) is None


def test_json_safe_converts_datetime_to_string() -> None:
    dt = datetime(2026, 6, 13, 9, 9, 43, tzinfo=UTC)
    result = _json_safe({"eventTime": dt})
    # datetime is coerced to its string form, not left as a datetime object.
    assert isinstance(result["eventTime"], str)
    assert "2026-06-13" in result["eventTime"]


def test_json_safe_result_is_json_serializable() -> None:
    # Mirrors a Bedrock trace artifact: nested datetimes inside a list.
    trace = {
        "traces": [
            {"agentId": "I1OA6XHJYA", "eventTime": datetime(2026, 6, 13, tzinfo=UTC)}
        ]
    }
    safe = _json_safe(trace)
    # The whole point: the coerced value must serialize without raising.
    json.dumps(safe)
