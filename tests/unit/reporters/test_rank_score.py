"""rank_score tiebreaker (R-07 track B slice A) — pure math + payload emission."""

import pytest

from atp.reporters.benchmark_reporter import (
    _bp_ordinal,
    _rank_score,
    build_report_benchmark_payload,
)


def test_bp_ordinal_none_is_best_and_levels_map_to_index() -> None:
    assert _bp_ordinal(None) == 5  # never breaks = best
    assert _bp_ordinal("clean") == 0
    assert _bp_ordinal("moderate") == 2
    assert _bp_ordinal("severe") == 3
    assert _bp_ordinal("very_severe") == 4
    assert _bp_ordinal("bogus") == 0  # unknown level = worst


def test_rank_score_equals_cpr_at_max_tiebreaker() -> None:
    # t == 1 (bp None + malformed 0). In real payloads bp None ⇒ cpr == 1.0, so
    # this coincides with the ceiling; here it is exercised directly.
    assert _rank_score(1.0, 5, 0.0, 5) == 1.0


def test_rank_score_penalizes_below_cpr_at_min_tiebreaker() -> None:
    # t == 0 → cpr - 1/(N+1)
    # rank_score is unrounded (full float precision keeps the invariant for all N)
    assert _rank_score(0.8, 0, 1.0, 5) == pytest.approx(0.8 - 1 / 6)


def test_rank_score_monotonic_in_breakpoint() -> None:
    # equal cpr, later breakpoint ⇒ higher rank_score
    assert _rank_score(0.8, 3, 0.0, 5) > _rank_score(0.8, 2, 0.0, 5)


@pytest.mark.parametrize("n", [3, 5, 15, 100])
def test_rank_score_never_crosses_real_gap_adversarial(n: int) -> None:
    # For any N: an agent one critical-pass step higher (k+1 vs k) must keep the
    # higher rank_score even with the WORST tiebreaker (t=0) against the lower
    # agent's BEST (t=1). Sweeps k across the range.
    for k in range(n):
        cpr_low = round(k / n, 6)
        cpr_high = round((k + 1) / n, 6)
        low_best = _rank_score(cpr_low, 5, 0.0, n)  # t = 1
        high_worst = _rank_score(cpr_high, 0, 1.0, n)  # t = 0
        assert high_worst > low_best, (n, k, high_worst, low_best)


def test_rank_score_breaks_ceiling_tie_within_bounds() -> None:
    a = _rank_score(1.0, 5, 0.0, 5)  # malformed 0 → 1.0
    b = _rank_score(1.0, 5, 0.2, 5)  # malformed 0.2 → < 1.0
    assert a != b
    assert a <= 1.0 and b <= 1.0


def test_rank_score_zero_cases() -> None:
    assert _rank_score(0.0, 5, 0.0, 0) == 0.0


def test_payload_emits_rank_score_and_bp_ordinal_without_touching_score() -> None:
    cases = [
        {
            "axis_level": "clean",
            "critical_pass": True,
            "malformed": False,
            "rubric_score": 1.0,
            "tokens": 1,
            "cost_usd": 0.0,
            "duration_seconds": 0.1,
            "error_class": None,
        },
        {
            "axis_level": "moderate",
            "critical_pass": False,
            "malformed": False,
            "rubric_score": 0.0,
            "tokens": 1,
            "cost_usd": 0.0,
            "duration_seconds": 0.1,
            "error_class": None,
        },
    ]
    p = build_report_benchmark_payload(
        run_id="r",
        benchmark_id="code-review",
        agent_id="x@m",
        ts="2026-07-01T00:00:00Z",
        case_results=cases,
    )
    assert p["score"] == 0.5  # critical_pass_rate unchanged
    sc = p["score_components"]
    assert sc["bp_ordinal"] == 2  # breaks at moderate (index 2)
    assert isinstance(sc["rank_score"], float)
    assert sc["rank_score"] <= p["score"]
