"""Trace storage backends for persisting execution traces."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from atp.tracing.models import Trace, TraceSummary

logger = logging.getLogger(__name__)

DEFAULT_TRACES_DIR = Path.home() / ".atp" / "traces"


class TraceStorage(ABC):
    """Abstract base class for trace storage backends."""

    @abstractmethod
    def save(self, trace: Trace) -> None:
        """Save a trace.

        Args:
            trace: Trace to persist.
        """

    @abstractmethod
    def load(self, trace_id: str) -> Trace | None:
        """Load a trace by ID.

        Args:
            trace_id: Unique trace identifier.

        Returns:
            The trace, or None if not found.
        """

    @abstractmethod
    def list_traces(
        self,
        test_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TraceSummary]:
        """List trace summaries with optional filtering.

        Args:
            test_id: Filter by test ID.
            status: Filter by status.
            limit: Maximum number of results.

        Returns:
            List of trace summaries.
        """

    @abstractmethod
    def delete(self, trace_id: str) -> bool:
        """Delete a trace.

        Args:
            trace_id: Trace to delete.

        Returns:
            True if deleted, False if not found.
        """


class FileTraceStorage(TraceStorage):
    """File-based trace storage using JSON files.

    Stores each trace as a JSON file in the configured directory.
    Default location: ``~/.atp/traces/``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize file-based storage.

        Args:
            base_dir: Directory for trace files.
                Defaults to ``~/.atp/traces/``.
        """
        self._base_dir = base_dir or DEFAULT_TRACES_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        """Return the storage directory."""
        return self._base_dir

    def _trace_path(self, trace_id: str) -> Path:
        """Get the file path for a trace ID."""
        safe_id = trace_id.replace("/", "_").replace("\\", "_")
        return self._base_dir / f"{safe_id}.json"

    def save(self, trace: Trace) -> None:
        """Save a trace to a JSON file."""
        path = self._trace_path(trace.trace_id)
        data = trace.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug("Saved trace %s to %s", trace.trace_id, path)

    def load(self, trace_id: str) -> Trace | None:
        """Load a trace from a JSON file."""
        path = self._trace_path(trace_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Trace.model_validate(data)
        except Exception:
            logger.exception("Failed to load trace %s", trace_id)
            return None

    def list_traces(
        self,
        test_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TraceSummary]:
        """List traces from the storage directory."""
        summaries: list[TraceSummary] = []

        if not self._base_dir.exists():
            return summaries

        files = sorted(
            self._base_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for path in files:
            if len(summaries) >= limit:
                break
            try:
                data = json.loads(path.read_text())
                trace = Trace.model_validate(data)

                if test_id and trace.test_id != test_id:
                    continue
                if status and trace.status != status:
                    continue

                summaries.append(TraceSummary.from_trace(trace))
            except Exception:
                logger.debug("Skipping invalid trace file: %s", path)
                continue

        return summaries

    def delete(self, trace_id: str) -> bool:
        """Delete a trace file."""
        path = self._trace_path(trace_id)
        if path.exists():
            path.unlink()
            return True
        return False


def get_default_storage() -> FileTraceStorage:
    """Return the default file-based trace storage."""
    return FileTraceStorage()
