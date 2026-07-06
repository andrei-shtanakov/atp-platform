"""report_benchmark reporter — aggregate a method-family run into a
`report_benchmark-v1` payload for the arbiter MCP `report_benchmark` tool."""

from typing import Any

from atp.reporters.base import Reporter, SuiteReport

_AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]
PAYLOAD_VERSION = "1.0.0"
_REPORT_ERROR_CLASSES = {"timeout", "crash", "test_failure", "other"}
_TEST_FAILURE_STATUSES = {"failed", "no_run"}


def _breakpoint(case_results: list[dict[str, Any]]) -> str | None:
    """Lowest axis_level at which the critical_check first fails (the routing
    signal). None if every level passed."""
    failed = [c["axis_level"] for c in case_results if not c["critical_pass"]]
    if not failed:
        return None
    return min(failed, key=lambda a: _AXIS_ORDER.index(a) if a in _AXIS_ORDER else 99)


def _bp_ordinal(bp: str | None) -> int:
    """Numeric breakpoint: higher = holds longer = better. ``None`` (never
    breaks) sorts above every real level; an unknown level maps to 0 (worst)."""
    if bp is None:
        return len(_AXIS_ORDER)  # 5
    return _AXIS_ORDER.index(bp) if bp in _AXIS_ORDER else 0


def _rank_score(
    pass_rate: float, bp_ordinal: int, malformed_rate: float, n: int
) -> float:
    """critical_pass_rate plus a sub-1/N tiebreaker (breakpoint + malformed).

    ``rank_score = cpr + (t - 1)/(N + 1)`` with ``t ∈ [0, 1]``: bounded so it
    can never cross a genuine 1/N critical-pass gap, and ``<= pass_rate <= 1.0``
    by construction (fixes the ceiling clamp). See the design spec.

    Preconditions: ``malformed_rate ∈ [0, 1]`` and ``bp_ordinal ∈ [0,
    len(_AXIS_ORDER)]`` (both hold for real payloads) ⇒ ``t ∈ [0, 1]``. The
    result is NOT rounded: the genuine-difference margin ``1/(N(N+1))`` would be
    eroded by 6-decimal rounding for N ≳ 1000, so full float precision keeps the
    ordering invariant unconditional.
    """
    if n == 0:
        return 0.0
    t = 0.75 * (bp_ordinal / len(_AXIS_ORDER)) + 0.25 * (1.0 - malformed_rate)
    return pass_rate + (t - 1.0) / (n + 1)


def normalize_report_error_class(error_class: Any) -> str | None:
    """Map run statuses to the report_benchmark-v1 error_class enum."""
    if error_class is None:
        return None
    value = str(error_class)
    if value in _REPORT_ERROR_CLASSES:
        return value
    if value in _TEST_FAILURE_STATUSES:
        return "test_failure"
    return "other"


def _case_runs(c: dict[str, Any]) -> tuple[int, int]:
    """(run_pass_count, runs_graded) for a case, with a legacy fallback.

    runs=N grades a case over N runs (``runs_graded``) and counts the passes
    (``run_pass_count``) — the reliability signal that separates a solid
    1.000 from a "1.000 with a 2/3 flake". Key-presence (not truthiness) picks
    the branch: an all-infra case legitimately carries ``runs_graded == 0``
    and must be preserved as ``(0, 0)``, not conflated with a legacy dict that
    has no key at all (which degrades to a binary 1/1 or 0/1 from
    ``critical_pass``).
    """
    if "runs_graded" in c:
        return int(c.get("run_pass_count", 0)), int(c["runs_graded"])
    return (1 if c.get("critical_pass") else 0), 1


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
    score_components also carries ``malformed_rate`` — the share of cases whose
    output failed strict Finding validation (a routing fact distinct from a
    missed defect; a case may set ``malformed`` to flag it, defaulting False).

    The payload satisfies the Request branch of report_benchmark-v1.schema.json.
    Note: ``score_components`` values must all be numbers (schema constraint).
    ``breakpoint_axis_level`` is therefore surfaced as a top-level string field.
    """
    n = len(case_results)
    crit_pass = sum(1 for c in case_results if c["critical_pass"])
    pass_rate = round(crit_pass / n, 6) if n else 0.0
    # malformed_rate: share of cases whose output was not a valid findings array
    # (unparseable OR failed strict Finding validation). Distinct from a missed
    # defect — a high rate means the agent isn't following the output contract,
    # a real routing fact. Defaults False for legacy case dicts without the key.
    malformed = sum(1 for c in case_results if c.get("malformed", False))
    malformed_rate = round(malformed / n, 6) if n else 0.0
    mean_rubric = (
        round(sum(c["rubric_score"] for c in case_results) / n, 6) if n else 0.0
    )
    # Reliability signal (additive; per_task allows extra props, score_components
    # allows extra numbers): mean per-case pass FRACTION across runs. Unlike the
    # binary critical_pass_rate, this separates a rock-solid 1.000 from a "1.000
    # with a flaky 2/3 case" — the real discriminator on ceiling-effect verticals.
    # An all-infra case (runs_graded == 0) counts as a 0.0 fraction, matching how
    # critical_pass_rate treats it as a failure — excluding it would inflate the
    # metric above the true per-case success rate.
    run_fracs = [
        (rp / rg if rg else 0.0) for rp, rg in (_case_runs(c) for c in case_results)
    ]
    mean_run_pass_rate = round(sum(run_fracs) / len(run_fracs), 6) if run_fracs else 0.0
    per_task = [
        {
            "task_index": i,
            "task_type": "review",
            "score": round(c["rubric_score"] if c["critical_pass"] else 0.0, 6),
            "tokens_used": c["tokens"],
            "duration_seconds": c["duration_seconds"],
            "error_class": normalize_report_error_class(c["error_class"]),
            "run_pass_count": _case_runs(c)[0],
            "runs_graded": _case_runs(c)[1],
            # Per-class usage + provenance (ADR-ECO-003d #1(a); additive —
            # WireTaskResult.additionalProperties=true). Lets the (later,
            # gated) pricing view apply cache-split tariffs and the leaderboard
            # flag estimated usage. None on legacy dicts without the keys.
            "input_tokens": c.get("input_tokens"),
            "output_tokens": c.get("output_tokens"),
            "cache_creation_tokens": c.get("cache_creation_tokens"),
            "cache_read_tokens": c.get("cache_read_tokens"),
            "usage_source": c.get("usage_source"),
        }
        for i, c in enumerate(case_results)
    ]
    bp = _breakpoint(case_results)
    bp_ordinal = _bp_ordinal(bp)
    rank_score = _rank_score(pass_rate, bp_ordinal, malformed_rate, n)
    payload: dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "agent_id": agent_id,
        "ts": ts,
        "score": pass_rate,
        # score_components values must all be numbers (schema constraint).
        # rank_score/bp_ordinal are routing-only signals (NOT a breakdown of
        # score): rank_score is the interim combined tiebreaker the arbiter
        # reader consumes; bp_ordinal is the raw signal for the slice-B repayment.
        "score_components": {
            "critical_pass_rate": pass_rate,
            "mean_rubric": mean_rubric,
            "malformed_rate": malformed_rate,
            "mean_run_pass_rate": mean_run_pass_rate,
            "rank_score": rank_score,
            "bp_ordinal": bp_ordinal,
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
        """Fail fast — this is not a SuiteReport formatter.

        The arbiter payload is built from per-case (critical_pass / rubric /
        axis_level) results via the module-level
        ``build_report_benchmark_payload``; a generic ``SuiteReport`` does not
        carry that shape. Raising here prevents a silent "successful" run that
        produces no output when selected as an output format. (Wiring a real
        SuiteReport→payload mapping is Phase-1b.)
        """
        raise NotImplementedError(
            "BenchmarkReporter does not format a SuiteReport. Call "
            "build_report_benchmark_payload(...) from the run wiring instead."
        )
