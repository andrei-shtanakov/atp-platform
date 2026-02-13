"""OpenTelemetry trace importer.

Reads OTLP JSON export files and converts spans into
TraceRecord objects for ATP test generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atp.generator.trace_import import TraceImporter, TraceRecord


class OpenTelemetryImporter(TraceImporter):
    """Import traces from OpenTelemetry OTLP JSON exports.

    Args:
        file_path: Path to the OTLP JSON file.
    """

    def __init__(
        self,
        *,
        file_path: str = "",
    ) -> None:
        self._file_path = file_path

    @property
    def name(self) -> str:
        """Importer name."""
        return "otel"

    async def fetch_traces(
        self,
        *,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[TraceRecord]:
        """Read OTLP JSON file and convert to TraceRecords.

        Args:
            limit: Maximum number of traces to return.
            **kwargs: Ignored for file-based imports.

        Returns:
            List of TraceRecord objects.
        """
        file_path = kwargs.get("file_path", self._file_path)
        if not file_path:
            raise ValueError("file_path is required for OpenTelemetry import")

        raw_spans = self._load_spans(str(file_path))
        root_spans = self._extract_root_spans(raw_spans)
        records = [self._span_to_record(s) for s in root_spans[:limit]]
        return records

    @staticmethod
    def _load_spans(file_path: str) -> list[dict[str, Any]]:
        """Load spans from an OTLP JSON export file.

        Supports both the nested OTLP format and flat span arrays.
        """
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        spans: list[dict[str, Any]] = []

        if isinstance(data, list):
            # Flat list of spans
            spans = data
        elif isinstance(data, dict):
            # OTLP nested format
            for resource_span in data.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    spans.extend(scope_span.get("spans", []))
            # Also handle {"spans": [...]} format
            if not spans and "spans" in data:
                spans = data["spans"]
        return spans

    @staticmethod
    def _extract_root_spans(
        spans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter to root spans (no parent or empty parent)."""
        root: list[dict[str, Any]] = []
        for span in spans:
            parent = span.get("parentSpanId", "")
            if not parent or parent == "0" * len(parent):
                root.append(span)
        # If no root spans found, return all spans
        return root if root else spans

    @staticmethod
    def _get_attribute(
        span: dict[str, Any],
        key: str,
        default: str = "",
    ) -> str:
        """Extract an attribute value from a span."""
        for attr in span.get("attributes", []):
            attr_key = attr.get("key", "")
            if attr_key == key:
                value = attr.get("value", {})
                if isinstance(value, dict):
                    return str(
                        value.get(
                            "stringValue",
                            value.get("intValue", default),
                        )
                    )
                return str(value)
        return default

    @classmethod
    def _span_to_record(
        cls,
        span: dict[str, Any],
    ) -> TraceRecord:
        """Convert an OTLP span to a TraceRecord."""
        trace_id = span.get("traceId", span.get("trace_id", ""))
        span_id = span.get("spanId", span.get("span_id", ""))
        name = span.get("name", "")

        # Extract input/output from attributes
        input_text = (
            cls._get_attribute(span, "input")
            or cls._get_attribute(span, "llm.input")
            or cls._get_attribute(span, "gen_ai.prompt")
        )
        if not input_text:
            input_text = name or f"Span {span_id}"

        output_text = (
            cls._get_attribute(span, "output")
            or cls._get_attribute(span, "llm.output")
            or cls._get_attribute(span, "gen_ai.completion")
        )

        # Duration from start/end timestamps (nanoseconds)
        duration_ms: float | None = None
        start = span.get(
            "startTimeUnixNano",
            span.get("start_time_unix_nano"),
        )
        end = span.get("endTimeUnixNano", span.get("end_time_unix_nano"))
        if start is not None and end is not None:
            try:
                duration_ms = (int(end) - int(start)) / 1_000_000
            except (ValueError, TypeError):
                pass

        # Status
        span_status = span.get("status", {})
        if isinstance(span_status, dict):
            code = span_status.get("code", 0)
            status = "error" if code == 2 else "completed"
        else:
            status = "completed"

        # Tags from span name
        tags: list[str] = []
        if name:
            tags.append(f"span:{name}")

        record_id = trace_id or span_id or "unknown"

        metadata: dict[str, Any] = {}
        if name:
            metadata["span_name"] = name

        return TraceRecord(
            trace_id=record_id,
            input_text=input_text,
            output_text=output_text,
            metadata=metadata,
            tags=tags,
            duration_ms=duration_ms,
            status=status,
        )
