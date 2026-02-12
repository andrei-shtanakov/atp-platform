"""Call recorder for tracking tool calls."""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from atp.mock_tools.models import ToolCallRecord


class CallRecorder:
    """Thread-safe recorder for tool calls."""

    def __init__(self) -> None:
        """Initialize the call recorder."""
        self._records: list[ToolCallRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        tool: str,
        input_data: dict[str, Any] | str | None,
        output: dict[str, Any] | str | None,
        error: str | None,
        status: str,
        duration_ms: float,
        task_id: str | None = None,
    ) -> ToolCallRecord:
        """Record a tool call.

        Args:
            tool: Tool name
            input_data: Tool input
            output: Tool output
            error: Error message if any
            status: Call status
            duration_ms: Call duration in milliseconds
            task_id: Optional task ID

        Returns:
            The recorded ToolCallRecord
        """
        record = ToolCallRecord(
            timestamp=datetime.now(UTC),
            tool=tool,
            input=input_data,
            output=output,
            error=error,
            status=status,
            duration_ms=duration_ms,
            task_id=task_id,
        )

        with self._lock:
            self._records.append(record)

        return record

    def get_records(
        self,
        tool: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
    ) -> list[ToolCallRecord]:
        """Get recorded calls with optional filtering.

        Args:
            tool: Filter by tool name
            task_id: Filter by task ID
            limit: Maximum number of records to return

        Returns:
            List of matching records
        """
        with self._lock:
            records = list(self._records)

        if tool:
            records = [r for r in records if r.tool == tool]

        if task_id:
            records = [r for r in records if r.task_id == task_id]

        if limit:
            records = records[-limit:]

        return records

    def get_call_count(self, tool: str | None = None) -> int:
        """Get number of recorded calls.

        Args:
            tool: Optional tool name to filter by

        Returns:
            Number of matching calls
        """
        with self._lock:
            if tool:
                return sum(1 for r in self._records if r.tool == tool)
            return len(self._records)

    def clear(self) -> int:
        """Clear all recorded calls.

        Returns:
            Number of records cleared
        """
        with self._lock:
            count = len(self._records)
            self._records = []
            return count

    def __iter__(self) -> Iterator[ToolCallRecord]:
        """Iterate over recorded calls."""
        with self._lock:
            records = list(self._records)
        yield from records

    def __len__(self) -> int:
        """Return number of recorded calls."""
        with self._lock:
            return len(self._records)

    def __bool__(self) -> bool:
        """Return True to indicate recorder exists (even if empty)."""
        return True
