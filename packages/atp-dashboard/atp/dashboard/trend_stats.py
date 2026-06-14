"""Pure trend statistics for the eval dashboard (SP-3). No DB.

Mirrors the OLS approach in atp/analytics/trend.py but operates on an in-memory
series (the dashboard reads points from the DB, not JSON files).
"""

import statistics

# Slope magnitude below this is treated as flat (per-run change in a 0..1 rate).
_STABLE_THRESHOLD = 0.01


def ols_slope(values: list[float]) -> float | None:
    """OLS slope of ``values`` against their 0-based index. None if < 2 points."""
    if len(values) < 2:
        return None
    xs = list(range(len(values)))
    slope, _intercept = statistics.linear_regression(xs, values)
    return round(slope, 6)


def classify_trend(slope: float | None, threshold: float = _STABLE_THRESHOLD) -> str:
    """Label a slope: improving / degrading / stable / n/a (None)."""
    if slope is None:
        return "n/a"
    if slope > threshold:
        return "improving"
    if slope < -threshold:
        return "degrading"
    return "stable"
