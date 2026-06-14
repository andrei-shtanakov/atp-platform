"""SP-3: pure OLS slope + trend classification."""

from atp.dashboard.trend_stats import classify_trend, ols_slope


def test_ols_slope_increasing() -> None:
    assert ols_slope([0.2, 0.4, 0.6]) == 0.2


def test_ols_slope_flat() -> None:
    assert ols_slope([0.5, 0.5, 0.5]) == 0.0


def test_ols_slope_needs_two_points() -> None:
    assert ols_slope([]) is None
    assert ols_slope([0.7]) is None


def test_classify_trend() -> None:
    assert classify_trend(0.05) == "improving"
    assert classify_trend(-0.05) == "degrading"
    assert classify_trend(0.001) == "stable"
    assert classify_trend(None) == "n/a"
