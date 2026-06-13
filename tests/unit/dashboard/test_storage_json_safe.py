"""Regression tests for storage._json_safe / _json_default.

Bedrock (boto3) trace artifacts embed datetime objects; storing them in the
JSON ``data_json`` column used to abort the whole result save with
"Object of type datetime is not JSON serializable".
"""

import json
from datetime import UTC, date, datetime

from atp.dashboard.storage import _json_safe


def test_json_safe_passes_through_plain_data() -> None:
    value = {"a": 1, "b": ["x", "y"], "c": {"nested": True}}
    assert _json_safe(value) == value


def test_json_safe_handles_none() -> None:
    assert _json_safe(None) is None


def test_json_safe_datetime_uses_isoformat() -> None:
    dt = datetime(2026, 6, 13, 9, 9, 43, tzinfo=UTC)
    result = _json_safe({"eventTime": dt})
    # ISO-8601 (T-separated), not str(datetime) (space-separated) — matches the
    # rest of the dashboard's JS consumers.
    assert result["eventTime"] == dt.isoformat()
    assert "T" in result["eventTime"]


def test_json_safe_date_uses_isoformat() -> None:
    result = _json_safe({"d": date(2026, 6, 13)})
    assert result["d"] == "2026-06-13"


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
