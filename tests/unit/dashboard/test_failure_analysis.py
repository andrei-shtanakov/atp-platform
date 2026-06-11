"""Unit tests for dashboard failure-cause aggregation."""

from __future__ import annotations

from types import SimpleNamespace

from atp.dashboard.failure_analysis import (
    FailureBreakdown,
    FailureCause,
    compute_failure_breakdown,
)


def _run(response_status: str, error: str | None = None) -> SimpleNamespace:
    """Build a lightweight stand-in for a RunResult ORM object."""
    return SimpleNamespace(response_status=response_status, error=error)


def test_empty_input() -> None:
    result = compute_failure_breakdown([])

    assert isinstance(result, FailureBreakdown)
    assert result.total_runs == 0
    assert result.failed_runs == 0
    assert result.causes == []


def test_all_completed_has_no_causes() -> None:
    runs = [_run("completed") for _ in range(4)]

    result = compute_failure_breakdown(runs)

    assert result.total_runs == 4
    assert result.failed_runs == 0
    assert result.causes == []


def test_mixed_counts() -> None:
    runs = [
        _run("completed"),
        _run("completed"),
        _run("completed"),
        _run("timeout", "timed out after 30s"),
        _run("timeout", "timed out after 45s"),
        _run("failed", "boom"),
    ]

    result = compute_failure_breakdown(runs)

    assert result.total_runs == 6
    assert result.failed_runs == 3

    causes_by_status = {c.status: c for c in result.causes}
    assert set(causes_by_status) == {"timeout", "failed"}

    timeout = causes_by_status["timeout"]
    assert timeout.count == 2
    # Two timeouts differing only by digits cluster into one sample.
    assert len(timeout.sample_errors) == 1

    failed = causes_by_status["failed"]
    assert failed.count == 1
    assert failed.sample_errors == ["boom"]


def test_error_normalization_and_sample_cap() -> None:
    # Five timeouts that all normalize to the same cluster.
    runs = [_run("timeout", f"timed out after {n}s") for n in (10, 20, 30, 40, 50)]

    result = compute_failure_breakdown(runs)

    assert result.failed_runs == 5
    assert len(result.causes) == 1

    cause = result.causes[0]
    assert isinstance(cause, FailureCause)
    assert cause.status == "timeout"
    assert cause.count == 5
    # All cluster to one normalized key -> a single distinct sample.
    assert len(cause.sample_errors) == 1
    assert cause.sample_errors[0].startswith("timed out after")


def test_distinct_clusters_capped_at_three() -> None:
    runs = [
        _run("failed", "error code 1: disk full"),
        _run("failed", "connection refused"),
        _run("failed", "permission denied"),
        _run("failed", "out of memory"),
    ]

    result = compute_failure_breakdown(runs)

    cause = result.causes[0]
    assert cause.status == "failed"
    assert cause.count == 4
    # Four distinct error clusters, but only 3 representatives are kept.
    assert len(cause.sample_errors) == 3


def test_causes_sorted_by_count_desc() -> None:
    runs = (
        [_run("failed", "a") for _ in range(1)]
        + [_run("timeout", "b") for _ in range(3)]
        + [_run("cancelled", "c") for _ in range(2)]
    )

    result = compute_failure_breakdown(runs)

    counts = [c.count for c in result.causes]
    assert counts == sorted(counts, reverse=True)
    assert result.causes[0].status == "timeout"
    assert result.failed_runs == 6


def test_none_error_handled() -> None:
    runs = [_run("failed", None), _run("failed", None)]

    result = compute_failure_breakdown(runs)

    cause = result.causes[0]
    assert cause.count == 2
    assert cause.sample_errors == [""]
