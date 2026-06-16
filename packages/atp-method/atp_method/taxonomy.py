"""The one taxonomy registry mapping internal `task_type` to the arbiter export
`benchmark_id` (ADR-006 direction #3).

The internal store / CLI / dashboard speak `task_type` (the arbiter `TaskType`
canon, e.g. "review"); `benchmark_id` (e.g. "code-review") exists only on the
`report_benchmark-v1` export to arbiter and is derived here at the sink. Keeping
the map in one place stops "benchmark" leaking back onto the internal store.
"""

# task_type (internal) -> benchmark_id (arbiter export key).
TASK_TYPE_TO_BENCHMARK_ID: dict[str, str] = {
    "review": "code-review",  # arbiter TaskType::Review (ordinal 5)
    "req-extraction": "req-extraction",  # requirement-extraction calibration vertical
}


def benchmark_id_for(task_type: str) -> str:
    """Return the arbiter export benchmark_id for an internal task_type."""
    try:
        return TASK_TYPE_TO_BENCHMARK_ID[task_type]
    except KeyError as exc:
        known = sorted(TASK_TYPE_TO_BENCHMARK_ID)
        raise ValueError(f"unknown task_type {task_type!r}; known: {known}") from exc
