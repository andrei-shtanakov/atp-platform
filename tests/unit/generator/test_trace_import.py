"""Tests for trace import functionality.

Covers TraceRecord, TraceImporter base class, LangSmith importer,
OpenTelemetry importer, importer registry, and CLI integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from atp.generator.trace_import import (
    TraceImporter,
    TraceRecord,
    _deduplicate,
    _trace_to_test,
)

# ---------------------------------------------------------------------------
# TraceRecord tests
# ---------------------------------------------------------------------------


class TestTraceRecord:
    """Tests for the TraceRecord dataclass."""

    def test_basic_creation(self) -> None:
        rec = TraceRecord(
            trace_id="abc123",
            input_text="What is 2+2?",
            output_text="4",
        )
        assert rec.trace_id == "abc123"
        assert rec.input_text == "What is 2+2?"
        assert rec.output_text == "4"
        assert rec.status == "completed"
        assert rec.tags == []
        assert rec.tool_calls == []
        assert rec.duration_ms is None

    def test_content_hash_deterministic(self) -> None:
        r1 = TraceRecord(trace_id="a", input_text="hello", output_text="world")
        r2 = TraceRecord(trace_id="b", input_text="hello", output_text="different")
        assert r1.content_hash == r2.content_hash

    def test_content_hash_differs_for_different_input(self) -> None:
        r1 = TraceRecord(trace_id="a", input_text="hello", output_text="world")
        r2 = TraceRecord(trace_id="a", input_text="goodbye", output_text="world")
        assert r1.content_hash != r2.content_hash

    def test_metadata_and_tags(self) -> None:
        rec = TraceRecord(
            trace_id="x",
            input_text="test",
            output_text="out",
            metadata={"key": "val"},
            tags=["prod", "v2"],
            tool_calls=["search", "write"],
            duration_ms=1500.0,
            status="error",
        )
        assert rec.metadata == {"key": "val"}
        assert rec.tags == ["prod", "v2"]
        assert rec.tool_calls == ["search", "write"]
        assert rec.duration_ms == 1500.0
        assert rec.status == "error"


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for _deduplicate helper."""

    def test_removes_duplicates(self) -> None:
        traces = [
            TraceRecord(trace_id="1", input_text="same", output_text="a"),
            TraceRecord(trace_id="2", input_text="same", output_text="b"),
            TraceRecord(trace_id="3", input_text="different", output_text="c"),
        ]
        result = _deduplicate(traces)
        assert len(result) == 2
        assert result[0].trace_id == "1"
        assert result[1].trace_id == "3"

    def test_empty_list(self) -> None:
        assert _deduplicate([]) == []

    def test_no_duplicates(self) -> None:
        traces = [
            TraceRecord(trace_id="1", input_text="a", output_text="x"),
            TraceRecord(trace_id="2", input_text="b", output_text="y"),
        ]
        result = _deduplicate(traces)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _trace_to_test tests
# ---------------------------------------------------------------------------


class TestTraceToTest:
    """Tests for _trace_to_test helper."""

    def test_basic_conversion(self) -> None:
        rec = TraceRecord(
            trace_id="abc123def456",
            input_text="Write a hello world program",
            output_text="print('hello')",
        )
        test = _trace_to_test(rec, "trace-0001")
        assert test.id == "trace-0001"
        assert test.name == "Trace abc123def456"
        assert test.task.description == "Write a hello world program"
        assert "trace-import" in test.tags
        assert "regression" in test.tags

    def test_completed_adds_behavior_assertion(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="done",
            status="completed",
        )
        test = _trace_to_test(rec, "trace-0001")
        assert any(a.type == "behavior" for a in test.assertions)

    def test_error_status_no_behavior_assertion(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="err",
            status="error",
        )
        test = _trace_to_test(rec, "trace-0001")
        assert not any(a.type == "behavior" for a in test.assertions)

    def test_extra_tags(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="out",
            tags=["prod"],
        )
        test = _trace_to_test(rec, "trace-0001", extra_tags=["ci"])
        assert "ci" in test.tags
        assert "prod" in test.tags

    def test_tool_calls_in_tags(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="out",
            tool_calls=["search", "write_file"],
        )
        test = _trace_to_test(rec, "trace-0001")
        assert "uses:search" in test.tags
        assert "uses:write_file" in test.tags

    def test_duration_sets_timeout(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="out",
            duration_ms=10000.0,  # 10 seconds
        )
        test = _trace_to_test(rec, "trace-0001")
        # 10s * 3 = 30s, clamped to min 60
        assert test.constraints.timeout_seconds == 60

    def test_long_duration_timeout(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="out",
            duration_ms=120000.0,  # 120 seconds
        )
        test = _trace_to_test(rec, "trace-0001")
        # 120s * 3 = 360
        assert test.constraints.timeout_seconds == 360

    def test_metadata_as_input_data(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="out",
            metadata={"key": "val"},
        )
        test = _trace_to_test(rec, "trace-0001")
        assert test.task.input_data == {"key": "val"}

    def test_empty_metadata_no_input_data(self) -> None:
        rec = TraceRecord(
            trace_id="t1",
            input_text="task",
            output_text="out",
            metadata={},
        )
        test = _trace_to_test(rec, "trace-0001")
        assert test.task.input_data is None


# ---------------------------------------------------------------------------
# TraceImporter.import_traces tests
# ---------------------------------------------------------------------------


class ConcreteImporter(TraceImporter):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test"

    async def fetch_traces(
        self, *, limit: int = 50, **kwargs: Any
    ) -> list[TraceRecord]:
        return []


class TestImportTraces:
    """Tests for TraceImporter.import_traces."""

    def test_creates_suite(self) -> None:
        importer = ConcreteImporter()
        traces = [
            TraceRecord(
                trace_id="t1",
                input_text="task1",
                output_text="out1",
            ),
            TraceRecord(
                trace_id="t2",
                input_text="task2",
                output_text="out2",
            ),
        ]
        suite = importer.import_traces(traces, suite_name="my-suite")
        assert suite.name == "my-suite"
        assert len(suite.tests) == 2

    def test_deduplication(self) -> None:
        importer = ConcreteImporter()
        traces = [
            TraceRecord(
                trace_id="t1",
                input_text="same",
                output_text="a",
            ),
            TraceRecord(
                trace_id="t2",
                input_text="same",
                output_text="b",
            ),
        ]
        suite = importer.import_traces(traces, deduplicate=True)
        assert len(suite.tests) == 1

    def test_no_deduplication(self) -> None:
        importer = ConcreteImporter()
        traces = [
            TraceRecord(
                trace_id="t1",
                input_text="same",
                output_text="a",
            ),
            TraceRecord(
                trace_id="t2",
                input_text="same",
                output_text="b",
            ),
        ]
        suite = importer.import_traces(traces, deduplicate=False)
        assert len(suite.tests) == 2

    def test_extra_tags_applied(self) -> None:
        importer = ConcreteImporter()
        traces = [
            TraceRecord(
                trace_id="t1",
                input_text="task",
                output_text="out",
            ),
        ]
        suite = importer.import_traces(traces, tags=["ci", "nightly"])
        test = suite.tests[0]
        assert "ci" in test.tags
        assert "nightly" in test.tags

    def test_empty_traces(self) -> None:
        importer = ConcreteImporter()
        suite = importer.import_traces([])
        assert len(suite.tests) == 0

    def test_custom_description(self) -> None:
        importer = ConcreteImporter()
        traces = [
            TraceRecord(
                trace_id="t1",
                input_text="task",
                output_text="out",
            ),
        ]
        suite = importer.import_traces(traces, suite_description="Custom desc")
        assert suite.description == "Custom desc"


# ---------------------------------------------------------------------------
# LangSmith importer tests
# ---------------------------------------------------------------------------


class TestLangSmithImporter:
    """Tests for the LangSmith trace importer."""

    @pytest.fixture()
    def sample_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "run-001",
                "inputs": {"input": "Summarize this document"},
                "outputs": {"output": "Summary: ..."},
                "status": "success",
                "total_time_ms": 5000,
                "tags": ["prod"],
                "child_runs": [
                    {"run_type": "tool", "name": "search"},
                ],
            },
            {
                "id": "run-002",
                "inputs": {"question": "What is ATP?"},
                "outputs": {"answer": "Agent Test Platform"},
                "status": "success",
                "latency": 2.5,
                "tags": [],
                "child_runs": [],
            },
        ]

    @pytest.mark.anyio()
    async def test_fetch_traces_list_response(
        self, sample_runs: list[dict[str, Any]]
    ) -> None:
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        importer = LangSmithImporter(api_key="test-key", project="test-project")

        with patch.object(importer, "_fetch_runs", return_value=sample_runs):
            records = await importer.fetch_traces(limit=10)

        assert len(records) == 2
        assert records[0].trace_id == "run-001"
        assert records[0].input_text == "Summarize this document"
        assert records[0].output_text == "Summary: ..."
        assert records[0].status == "completed"
        assert records[0].duration_ms == 5000
        assert "search" in records[0].tool_calls
        assert "prod" in records[0].tags

    @pytest.mark.anyio()
    async def test_fetch_traces_second_run(
        self, sample_runs: list[dict[str, Any]]
    ) -> None:
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        importer = LangSmithImporter(api_key="key")

        with patch.object(importer, "_fetch_runs", return_value=sample_runs):
            records = await importer.fetch_traces(limit=10)

        assert len(records) == 2
        assert records[1].trace_id == "run-002"
        assert records[1].input_text == "What is ATP?"
        assert records[1].output_text == "Agent Test Platform"
        assert records[1].duration_ms == 2500.0

    @pytest.mark.anyio()
    async def test_fetch_traces_empty_response(self) -> None:
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        importer = LangSmithImporter(api_key="key")

        with patch.object(importer, "_fetch_runs", return_value=[]):
            records = await importer.fetch_traces(limit=10)

        assert records == []

    def test_run_to_record_latency_conversion(self) -> None:
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        run = {
            "id": "run-x",
            "inputs": "raw text",
            "outputs": "raw output",
            "latency": 3.0,
        }
        rec = LangSmithImporter._run_to_record(run)
        assert rec.duration_ms == 3000.0
        assert rec.input_text == "raw text"
        assert rec.output_text == "raw output"

    def test_name_property(self) -> None:
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        importer = LangSmithImporter(api_key="k")
        assert importer.name == "langsmith"


# ---------------------------------------------------------------------------
# OpenTelemetry importer tests
# ---------------------------------------------------------------------------


class TestOpenTelemetryImporter:
    """Tests for the OpenTelemetry trace importer."""

    @pytest.fixture()
    def otlp_nested_data(self) -> dict[str, Any]:
        return {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "abc123",
                                    "spanId": "span-1",
                                    "name": "agent.run",
                                    "parentSpanId": "",
                                    "startTimeUnixNano": 1000000000,
                                    "endTimeUnixNano": 3000000000,
                                    "status": {"code": 1},
                                    "attributes": [
                                        {
                                            "key": "input",
                                            "value": {"stringValue": "Hello"},
                                        },
                                        {
                                            "key": "output",
                                            "value": {"stringValue": "Hi!"},
                                        },
                                    ],
                                },
                                {
                                    "traceId": "abc123",
                                    "spanId": "span-2",
                                    "name": "tool.search",
                                    "parentSpanId": "span-1",
                                    "startTimeUnixNano": 1500000000,
                                    "endTimeUnixNano": 2000000000,
                                    "status": {"code": 1},
                                    "attributes": [],
                                },
                            ]
                        }
                    ]
                }
            ]
        }

    @pytest.fixture()
    def flat_spans_data(self) -> list[dict[str, Any]]:
        return [
            {
                "traceId": "trace-flat-1",
                "spanId": "s1",
                "name": "query",
                "parentSpanId": "",
                "startTimeUnixNano": 0,
                "endTimeUnixNano": 5000000000,
                "status": {},
                "attributes": [
                    {
                        "key": "gen_ai.prompt",
                        "value": {"stringValue": "What is AI?"},
                    },
                    {
                        "key": "gen_ai.completion",
                        "value": {"stringValue": "AI is artificial intelligence"},
                    },
                ],
            }
        ]

    @pytest.mark.anyio()
    async def test_nested_otlp_format(
        self, otlp_nested_data: dict[str, Any], tmp_path: Path
    ) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        trace_file = tmp_path / "traces.json"
        trace_file.write_text(json.dumps(otlp_nested_data))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=10)

        # Should only get root span (span-1)
        assert len(records) == 1
        assert records[0].trace_id == "abc123"
        assert records[0].input_text == "Hello"
        assert records[0].output_text == "Hi!"
        assert records[0].duration_ms == 2000.0
        assert records[0].status == "completed"

    @pytest.mark.anyio()
    async def test_flat_spans_format(
        self, flat_spans_data: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        trace_file = tmp_path / "flat.json"
        trace_file.write_text(json.dumps(flat_spans_data))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=10)

        assert len(records) == 1
        assert records[0].input_text == "What is AI?"
        assert records[0].output_text == "AI is artificial intelligence"
        assert records[0].duration_ms == 5000.0

    @pytest.mark.anyio()
    async def test_error_status(self, tmp_path: Path) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        data = [
            {
                "traceId": "err-trace",
                "spanId": "s1",
                "name": "failed_op",
                "parentSpanId": "",
                "status": {"code": 2},
                "attributes": [
                    {
                        "key": "input",
                        "value": {"stringValue": "do something"},
                    },
                ],
            }
        ]
        trace_file = tmp_path / "err.json"
        trace_file.write_text(json.dumps(data))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=10)

        assert len(records) == 1
        assert records[0].status == "error"

    @pytest.mark.anyio()
    async def test_missing_file_path_raises(self) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        importer = OpenTelemetryImporter()
        with pytest.raises(ValueError, match="file_path is required"):
            await importer.fetch_traces(limit=10)

    @pytest.mark.anyio()
    async def test_limit_respected(self, tmp_path: Path) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        spans = [
            {
                "traceId": f"t{i}",
                "spanId": f"s{i}",
                "name": f"op{i}",
                "parentSpanId": "",
                "attributes": [
                    {
                        "key": "input",
                        "value": {"stringValue": f"task {i}"},
                    },
                ],
            }
            for i in range(10)
        ]
        trace_file = tmp_path / "many.json"
        trace_file.write_text(json.dumps(spans))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=3)
        assert len(records) == 3

    @pytest.mark.anyio()
    async def test_spans_key_format(self, tmp_path: Path) -> None:
        """Test {"spans": [...]} format."""
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        data = {
            "spans": [
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "name": "op1",
                    "parentSpanId": "",
                    "attributes": [],
                }
            ]
        }
        trace_file = tmp_path / "spans_key.json"
        trace_file.write_text(json.dumps(data))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=10)
        assert len(records) == 1

    def test_name_property(self) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        importer = OpenTelemetryImporter()
        assert importer.name == "otel"


# ---------------------------------------------------------------------------
# Importer registry tests
# ---------------------------------------------------------------------------


class TestImporterRegistry:
    """Tests for the importer registry."""

    def test_get_langsmith(self) -> None:
        from atp.generator.importers import get_importer
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        importer = get_importer("langsmith", api_key="k", project="p")
        assert isinstance(importer, LangSmithImporter)

    def test_get_otel(self) -> None:
        from atp.generator.importers import get_importer
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        importer = get_importer("otel", file_path="f.json")
        assert isinstance(importer, OpenTelemetryImporter)

    def test_unknown_importer_raises(self) -> None:
        from atp.generator.importers import get_importer

        with pytest.raises(KeyError, match="Unknown importer"):
            get_importer("nonexistent")

    def test_list_importers(self) -> None:
        from atp.generator.importers import list_importers

        names = list_importers()
        assert "langsmith" in names
        assert "otel" in names


# ---------------------------------------------------------------------------
# End-to-end integration: fetch -> import -> suite
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests for the full trace import pipeline."""

    @pytest.mark.anyio()
    async def test_langsmith_to_suite(self) -> None:
        from atp.generator.importers.langsmith import (
            LangSmithImporter,
        )

        importer = LangSmithImporter(api_key="key", project="proj")

        mock_runs = [
            {
                "id": f"run-{i}",
                "inputs": {"input": f"Task {i}"},
                "outputs": {"output": f"Result {i}"},
                "status": "success",
                "total_time_ms": 1000 * (i + 1),
                "tags": [],
                "child_runs": [],
            }
            for i in range(5)
        ]

        with patch.object(importer, "_fetch_runs", return_value=mock_runs):
            records = await importer.fetch_traces(limit=5)

        suite = importer.import_traces(
            records,
            suite_name="regression-suite",
            tags=["nightly"],
        )

        assert suite.name == "regression-suite"
        assert len(suite.tests) == 5
        for test in suite.tests:
            assert "nightly" in test.tags
            assert "trace-import" in test.tags
            assert test.id.startswith("trace-")

    @pytest.mark.anyio()
    async def test_otel_to_suite(self, tmp_path: Path) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        spans = [
            {
                "traceId": f"trace-{i}",
                "spanId": f"span-{i}",
                "name": f"operation_{i}",
                "parentSpanId": "",
                "startTimeUnixNano": 1000000000 * i,
                "endTimeUnixNano": 1000000000 * (i + 1),
                "status": {"code": 1},
                "attributes": [
                    {
                        "key": "input",
                        "value": {"stringValue": f"Do task {i}"},
                    },
                    {
                        "key": "output",
                        "value": {"stringValue": f"Done {i}"},
                    },
                ],
            }
            for i in range(3)
        ]

        trace_file = tmp_path / "traces.json"
        trace_file.write_text(json.dumps(spans))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=10)
        suite = importer.import_traces(records)

        assert len(suite.tests) == 3
        assert suite.tests[0].task.description == "Do task 0"

    @pytest.mark.anyio()
    async def test_dedup_across_pipeline(self, tmp_path: Path) -> None:
        from atp.generator.importers.opentelemetry import (
            OpenTelemetryImporter,
        )

        # Two spans with identical input
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "op",
                "parentSpanId": "",
                "attributes": [
                    {
                        "key": "input",
                        "value": {"stringValue": "same task"},
                    },
                ],
            },
            {
                "traceId": "t2",
                "spanId": "s2",
                "name": "op",
                "parentSpanId": "",
                "attributes": [
                    {
                        "key": "input",
                        "value": {"stringValue": "same task"},
                    },
                ],
            },
        ]

        trace_file = tmp_path / "dup.json"
        trace_file.write_text(json.dumps(spans))

        importer = OpenTelemetryImporter(file_path=str(trace_file))
        records = await importer.fetch_traces(limit=10)
        suite = importer.import_traces(records, deduplicate=True)

        assert len(suite.tests) == 1

    def test_suite_can_convert_to_yaml(self) -> None:
        """Verify generated suite is valid for YAML export."""
        from atp.generator.writer import YAMLWriter

        importer = ConcreteImporter()
        traces = [
            TraceRecord(
                trace_id="t1",
                input_text="Build a calculator",
                output_text="done",
            ),
        ]
        suite = importer.import_traces(traces)
        writer = YAMLWriter()
        yaml_str = writer.to_yaml(suite)

        assert "Build a calculator" in yaml_str
        assert "trace-0001" in yaml_str
