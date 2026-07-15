"""Runtime usage capture — the mandatory UsageCapture seam (ADR-ECO-003e D2).

M0 is the observe-only slice: it records per-call token usage (or its
absence) at the runner⟷adapter boundary. BudgetControl (reserve/settle,
003e D1/D3) builds on the same records later.

This module must stay dependency-light (stdlib + PerClassUsage): downstream
repos vendor a pinned copy per the 003e D2 vendoring rule.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from atp.cost.cloud_pricer import PerClassUsage

if TYPE_CHECKING:
    from atp.protocol import Metrics

logger = logging.getLogger(__name__)

CAPTURE_PATH_ENV = "ATP_USAGE_CAPTURE_PATH"


@dataclass(frozen=True)
class UsageRecord:
    """One captured adapter call, with or without token usage.

    ``usage is None`` means the adapter path produced no token accounting —
    that absence is itself the Action №0 signal, so it is recorded, not
    skipped. ``model``/``provider`` are None until adapters plumb real ids
    (003e adoption, M1).
    """

    call_id: str
    timestamp: str
    adapter_type: str
    status: str
    model: str | None
    provider: str | None
    usage: PerClassUsage | None
    reported_cost_usd: float | None
    test_id: str | None = None


class UsageCapture(Protocol):
    """The mandatory usage-capture seam (ADR-ECO-003e D2)."""

    def record_usage(self, record: UsageRecord) -> None:
        """Record one adapter call. Must never raise."""
        ...


class NullUsageCapture:
    """Sink used when no capture path is configured."""

    def record_usage(self, record: UsageRecord) -> None:
        """Discard the record."""


class JsonlUsageCapture:
    """Append-only JSONL sink, idempotent on call_id within the process."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    def record_usage(self, record: UsageRecord) -> None:
        """Append the record as one JSON line; swallow and log IO errors."""
        with self._lock:
            if record.call_id in self._seen:
                return
            self._seen.add(record.call_id)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(record)) + "\n")
            except OSError as e:
                logger.warning("Usage capture write failed: %s", e)


def usage_from_metrics(metrics: Metrics | None) -> PerClassUsage | None:
    """Convert response Metrics into PerClassUsage.

    Returns None when no token class is reported at all (the "path does
    not capture usage" case); fills absent classes with 0 otherwise.
    """
    if metrics is None:
        return None
    fields = (
        metrics.input_tokens,
        metrics.output_tokens,
        metrics.cache_creation_tokens,
        metrics.cache_read_tokens,
    )
    if all(f is None for f in fields):
        return None
    return PerClassUsage(
        input_tokens=metrics.input_tokens or 0,
        output_tokens=metrics.output_tokens or 0,
        cache_creation_tokens=metrics.cache_creation_tokens or 0,
        cache_read_tokens=metrics.cache_read_tokens or 0,
        usage_source="measured",
    )


def capture_from_env() -> UsageCapture:
    """Build the process-wide capture sink from ATP_USAGE_CAPTURE_PATH."""
    path = os.environ.get(CAPTURE_PATH_ENV)
    if path:
        return JsonlUsageCapture(Path(path))
    return NullUsageCapture()
