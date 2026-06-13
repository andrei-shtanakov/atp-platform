"""report_benchmark reporter — aggregate a method-family run into a
`report_benchmark-v1` payload for the arbiter MCP `report_benchmark` tool."""

from typing import Any

from atp.reporters.base import Reporter, SuiteReport

_AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]
PAYLOAD_VERSION = "1.0.0"


def _breakpoint(case_results: list[dict[str, Any]]) -> str | None:
    """Lowest axis_level at which the critical_check first fails (the routing
    signal). None if every level passed."""
    failed = [c["axis_level"] for c in case_results if not c["critical_pass"]]
    if not failed:
        return None
    return min(failed, key=lambda a: _AXIS_ORDER.index(a) if a in _AXIS_ORDER else 99)


def build_report_benchmark_payload(
    *,
    run_id: str,
    benchmark_id: str,
    agent_id: str,
    ts: str,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-case results into the report_benchmark-v1 contract.

    score = critical_pass_rate (weighted equally; axis weighting is Phase-1b).

    The payload satisfies the Request branch of report_benchmark-v1.schema.json.
    Note: ``score_components`` values must all be numbers (schema constraint).
    ``breakpoint_axis_level`` is therefore surfaced as a top-level string field.
    """
    n = len(case_results)
    crit_pass = sum(1 for c in case_results if c["critical_pass"])
    pass_rate = round(crit_pass / n, 6) if n else 0.0
    mean_rubric = (
        round(sum(c["rubric_score"] for c in case_results) / n, 6) if n else 0.0
    )
    per_task = [
        {
            "task_index": i,
            "task_type": "review",
            "score": round(c["rubric_score"] if c["critical_pass"] else 0.0, 6),
            "tokens_used": c["tokens"],
            "duration_seconds": c["duration_seconds"],
            "error_class": c["error_class"],
        }
        for i, c in enumerate(case_results)
    ]
    bp = _breakpoint(case_results)
    payload: dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "agent_id": agent_id,
        "ts": ts,
        "score": pass_rate,
        # score_components values must all be numbers (schema constraint)
        "score_components": {
            "critical_pass_rate": pass_rate,
            "mean_rubric": mean_rubric,
        },
        # breakpoint surfaced at top level (string; Request allows additionalProperties)
        "total_tokens": sum(c["tokens"] for c in case_results),
        "total_cost_usd": round(sum(c["cost_usd"] for c in case_results), 6),
        "duration_seconds": round(sum(c["duration_seconds"] for c in case_results), 6),
        "per_task": per_task,
        "per_task_total_count": n,
        "per_task_truncated": False,
    }
    if bp is not None:
        payload["breakpoint_axis_level"] = bp
    return payload


class BenchmarkReporter(Reporter):
    """Reporter that builds a report_benchmark-v1 payload.

    This reporter wraps ``build_report_benchmark_payload`` with the standard
    ``Reporter`` interface so it can be registered in the reporter registry.
    It does not consume a ``SuiteReport`` directly; callers that need the
    arbiter payload should use ``build_report_benchmark_payload`` directly.
    """

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "report_benchmark"

    def report(self, report: SuiteReport) -> None:
        """No-op: use ``build_report_benchmark_payload`` for payload generation.

        This stub satisfies the ``Reporter`` ABC; the actual payload builder
        is the module-level function ``build_report_benchmark_payload``.
        """
