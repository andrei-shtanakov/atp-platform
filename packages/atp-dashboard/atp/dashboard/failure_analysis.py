"""Failure-cause aggregation for the dashboard execution history view.

Pure, synchronous helpers that summarize why a set of run results failed.
The input objects are duck-typed: only ``.response_status`` (str) and
``.error`` (str | None) are read, so any object (including the ``RunResult``
ORM model) can be passed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Protocol

from pydantic import BaseModel

COMPLETED_STATUS = "completed"
MAX_SAMPLE_ERRORS = 3

_WHITESPACE_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")


class RunResultLike(Protocol):
    """Structural type for objects accepted by failure aggregation."""

    response_status: str
    error: str | None


class FailureCause(BaseModel):
    """One non-completed status and its representative error messages.

    Attributes:
        status: The ``response_status`` value (e.g. "failed", "timeout").
        count: Number of runs with this status.
        sample_errors: Up to three distinct, original (non-normalized) error
            strings representing the distinct error clusters for this status.
    """

    status: str
    count: int
    sample_errors: list[str]


class FailureBreakdown(BaseModel):
    """Aggregated failure summary for a list of run results.

    Attributes:
        total_runs: Total number of runs supplied.
        failed_runs: Number of runs whose status is not "completed".
        causes: Failure causes sorted by ``count`` descending.
    """

    total_runs: int
    failed_runs: int
    causes: list[FailureCause]


def _normalize_error(error: str | None) -> str:
    """Normalize an error string so near-identical messages cluster together.

    Lowercases, collapses runs of whitespace, and replaces digit runs with
    ``N`` so messages differing only by numbers/paths group as one cluster.
    A ``None`` or empty error normalizes to an empty string.
    """
    if not error:
        return ""
    collapsed = _WHITESPACE_RE.sub(" ", error.strip()).lower()
    return _DIGITS_RE.sub("N", collapsed)


def compute_failure_breakdown(
    run_results: list[RunResultLike],
) -> FailureBreakdown:
    """Aggregate failure causes for a list of run results.

    Counts non-"completed" runs grouped by ``response_status``. Within each
    status, error strings are normalized and clustered; up to three distinct
    representative (original) error messages are kept per cause. Causes are
    sorted by count descending. Runs with status "completed" contribute
    nothing to the causes.

    Args:
        run_results: Objects exposing ``.response_status`` and ``.error``.

    Returns:
        A :class:`FailureBreakdown` summarizing the failures.
    """
    total_runs = len(run_results)
    status_counts: Counter[str] = Counter()
    samples_by_status: dict[str, dict[str, str]] = defaultdict(dict)

    for run in run_results:
        status = run.response_status
        if status == COMPLETED_STATUS:
            continue

        status_counts[status] += 1
        seen = samples_by_status[status]
        if len(seen) < MAX_SAMPLE_ERRORS:
            key = _normalize_error(run.error)
            if key not in seen:
                seen[key] = run.error or ""

    causes = [
        FailureCause(
            status=status,
            count=count,
            sample_errors=list(samples_by_status[status].values()),
        )
        for status, count in status_counts.most_common()
    ]

    return FailureBreakdown(
        total_runs=total_runs,
        failed_runs=sum(status_counts.values()),
        causes=causes,
    )
