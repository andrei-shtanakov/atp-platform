"""rank_score tiebreaker (R-07 track B slice A) — pure math + payload emission."""

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
    # t == 1 (bp None + malformed 0), which only happens at the cpr==1.0 ceiling
    assert _rank_score(1.0, 5, 0.0, 5) == 1.0


def test_rank_score_penalizes_below_cpr_at_min_tiebreaker() -> None:
    # t == 0 → cpr - 1/(N+1)
    assert _rank_score(0.8, 0, 1.0, 5) == round(0.8 - 1 / 6, 6)


def test_rank_score_monotonic_in_breakpoint() -> None:
    # equal cpr, later breakpoint ⇒ higher rank_score
    assert _rank_score(0.8, 3, 0.0, 5) > _rank_score(0.8, 2, 0.0, 5)


def test_rank_score_never_crosses_real_gap_adversarial() -> None:
    # lower cpr + best tiebreaker vs higher cpr + worst tiebreaker: higher wins
    low = _rank_score(0.6, 5, 0.0, 5)  # cpr 3/5, t=1 → 0.6
    high = _rank_score(0.8, 0, 1.0, 5)  # cpr 4/5, t=0 → 0.633333
    assert high > low


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
