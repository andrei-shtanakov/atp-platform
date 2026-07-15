"""Tests for the UsageCapture seam (ADR-ECO-003e M0)."""

import json
from pathlib import Path

from atp.cost.capture import (
    CAPTURE_PATH_ENV,
    JsonlUsageCapture,
    NullUsageCapture,
    UsageRecord,
    capture_from_env,
    usage_from_metrics,
)
from atp.cost.cloud_pricer import PerClassUsage
from atp.protocol import Metrics


def make_record(call_id: str = "call-1") -> UsageRecord:
    return UsageRecord(
        call_id=call_id,
        timestamp="2026-07-15T00:00:00+00:00",
        adapter_type="cli",
        status="completed",
        model=None,
        provider=None,
        usage=PerClassUsage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            usage_source="measured",
        ),
        reported_cost_usd=None,
        test_id="test-1",
    )


class TestUsageFromMetrics:
    def test_none_metrics_gives_none(self) -> None:
        assert usage_from_metrics(None) is None

    def test_all_token_fields_absent_gives_none(self) -> None:
        assert usage_from_metrics(Metrics(wall_time_seconds=1.0)) is None

    def test_partial_fields_fill_zero(self) -> None:
        usage = usage_from_metrics(Metrics(input_tokens=5))
        assert usage == PerClassUsage(
            input_tokens=5,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            usage_source="measured",
        )

    def test_all_four_classes_pass_through(self) -> None:
        usage = usage_from_metrics(
            Metrics(
                input_tokens=1,
                output_tokens=2,
                cache_creation_tokens=3,
                cache_read_tokens=4,
            )
        )
        assert usage is not None
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_tokens,
            usage.cache_read_tokens,
        ) == (1, 2, 3, 4)


class TestJsonlUsageCapture:
    def test_appends_one_json_line_per_record(self, tmp_path: Path) -> None:
        sink = JsonlUsageCapture(tmp_path / "usage.jsonl")
        sink.record_usage(make_record("a"))
        sink.record_usage(make_record("b"))
        lines = (tmp_path / "usage.jsonl").read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["call_id"] == "a"
        assert first["usage"]["input_tokens"] == 10

    def test_idempotent_on_call_id(self, tmp_path: Path) -> None:
        sink = JsonlUsageCapture(tmp_path / "usage.jsonl")
        sink.record_usage(make_record("a"))
        sink.record_usage(make_record("a"))
        lines = (tmp_path / "usage.jsonl").read_text().splitlines()
        assert len(lines) == 1

    def test_none_usage_serializes_as_null(self, tmp_path: Path) -> None:
        sink = JsonlUsageCapture(tmp_path / "usage.jsonl")
        record = UsageRecord(
            call_id="c",
            timestamp="2026-07-15T00:00:00+00:00",
            adapter_type="http",
            status="failed",
            model=None,
            provider=None,
            usage=None,
            reported_cost_usd=None,
            test_id=None,
        )
        sink.record_usage(record)
        row = json.loads((tmp_path / "usage.jsonl").read_text())
        assert row["usage"] is None

    def test_io_error_is_swallowed(self, tmp_path: Path) -> None:
        # Point the sink at a path whose parent is a *file* -> open() fails.
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        sink = JsonlUsageCapture(blocker / "usage.jsonl")
        sink.record_usage(make_record("a"))  # must not raise


class TestCaptureFromEnv:
    def test_unset_env_gives_null_capture(self, monkeypatch: object) -> None:
        monkeypatch.delenv(CAPTURE_PATH_ENV, raising=False)  # type: ignore[attr-defined]
        assert isinstance(capture_from_env(), NullUsageCapture)

    def test_set_env_gives_jsonl_capture(
        self, monkeypatch: object, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(  # type: ignore[attr-defined]
            CAPTURE_PATH_ENV, str(tmp_path / "u.jsonl")
        )
        assert isinstance(capture_from_env(), JsonlUsageCapture)
