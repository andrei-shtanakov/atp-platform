"""Base classes for importing production traces into ATP test suites."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from atp.generator.core import TestGenerator, TestSuiteData
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
)


@dataclass
class TraceRecord:
    """A single trace record extracted from a production system.

    Represents the minimal input/output pair needed to generate
    a regression test from a production trace.
    """

    trace_id: str
    input_text: str
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    status: str = "completed"
    tool_calls: list[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """Hash for deduplication based on input content."""
        return hashlib.sha256(self.input_text.encode("utf-8")).hexdigest()[:16]


class TraceImporter(ABC):
    """Abstract base class for trace importers.

    Subclasses implement fetching traces from a specific source
    (LangSmith, OpenTelemetry, etc.) and converting them to
    a list of TraceRecord objects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Importer name used in the registry."""

    @abstractmethod
    async def fetch_traces(
        self,
        *,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[TraceRecord]:
        """Fetch traces from the source.

        Args:
            limit: Maximum number of traces to fetch.
            **kwargs: Source-specific parameters.

        Returns:
            List of TraceRecord objects.
        """

    def import_traces(
        self,
        traces: list[TraceRecord],
        *,
        suite_name: str = "trace-regression",
        suite_description: str | None = None,
        deduplicate: bool = True,
        tags: list[str] | None = None,
    ) -> TestSuiteData:
        """Convert trace records into an ATP test suite.

        Args:
            traces: List of TraceRecord objects.
            suite_name: Name for the generated suite.
            suite_description: Optional description.
            deduplicate: Skip similar inputs by content hash.
            tags: Extra tags to add to every test.

        Returns:
            TestSuiteData ready to be saved as YAML.
        """
        if deduplicate:
            traces = _deduplicate(traces)

        generator = TestGenerator()
        suite = generator.create_suite(
            name=suite_name,
            description=suite_description
            or f"Generated from {len(traces)} production traces",
        )

        for trace in traces:
            test_id = generator.generate_test_id(suite, prefix="trace")
            test = _trace_to_test(trace, test_id, tags)
            generator.add_test(suite, test)

        return suite


def _deduplicate(traces: list[TraceRecord]) -> list[TraceRecord]:
    """Remove traces with duplicate input content."""
    seen: set[str] = set()
    unique: list[TraceRecord] = []
    for t in traces:
        h = t.content_hash
        if h not in seen:
            seen.add(h)
            unique.append(t)
    return unique


def _trace_to_test(
    trace: TraceRecord,
    test_id: str,
    extra_tags: list[str] | None = None,
) -> TestDefinition:
    """Convert a single TraceRecord to a TestDefinition."""
    all_tags: set[str] = {"trace-import", "regression"}
    all_tags.update(trace.tags)
    if extra_tags:
        all_tags.update(extra_tags)

    for tool in trace.tool_calls:
        all_tags.add(f"uses:{tool}")

    assertions: list[Assertion] = []
    if trace.status == "completed":
        assertions.append(
            Assertion(
                type="behavior",
                config={"check": "completed_successfully"},
            )
        )

    constraints = Constraints()
    if trace.duration_ms is not None:
        timeout = max(int(trace.duration_ms / 1000 * 3), 60)
        constraints = Constraints(timeout_seconds=timeout)

    return TestDefinition(
        id=test_id,
        name=f"Trace {trace.trace_id[:12]}",
        description=(f"Auto-generated from production trace {trace.trace_id}"),
        tags=sorted(all_tags),
        task=TaskDefinition(
            description=trace.input_text,
            input_data=(trace.metadata if trace.metadata else None),
            expected_artifacts=None,
        ),
        constraints=constraints,
        assertions=assertions,
    )
