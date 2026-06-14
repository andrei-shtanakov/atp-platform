"""Pure mappers from a run's TestDefinition tags + eval results to the SP-1
dimension/outcome columns. No DB — unit-testable in isolation.

Method runs carry `level_*`/`capability_*`/`family_*`/`version_*` tags and a
`critical_check` EvalCheck whose `details` is the CaseVerdict dump. Native runs
have neither, so every field degrades to None (the columns are nullable).
"""

from typing import Any

from atp.core.results import EvalResult
from atp.loader.models import TestDefinition

_AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]


def _tag_value(tags: list[str], prefix: str) -> str | None:
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix) :]
    return None


def _critical_details(results: list[EvalResult]) -> dict[str, Any]:
    for r in results:
        for c in r.checks:
            if c.name == "critical_check" and c.details:
                return c.details
    return {}


def _rubric_score(results: list[EvalResult]) -> float | None:
    for r in results:
        for c in r.checks:
            if c.name == "rubric":
                return c.score
    return None


def case_dimensions(
    test: TestDefinition, eval_results: list[EvalResult]
) -> dict[str, Any]:
    """Map one case's tags + eval results into the test_executions columns."""
    tags = test.tags or []
    v = _critical_details(eval_results)
    version = _tag_value(tags, "version_")
    return {
        "axis_level": _tag_value(tags, "level_"),
        "capability": _tag_value(tags, "capability_"),
        "family": _tag_value(tags, "family_"),
        "case_version": int(version) if version and version.isdigit() else None,
        "critical_pass": v.get("critical_pass"),
        "malformed": v.get("malformed"),
        "recall": v.get("recall"),
        "precision": v.get("precision"),
        "fp_count": v.get("fp_count"),
        "rubric_score": _rubric_score(eval_results),
        "grader_version": v.get("grader_version"),
    }


def aggregate_run(case_dims: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-case dims into the suite_executions run-level columns.

    Only cases with a non-null critical_pass count toward the rates (native
    runs contribute nothing). Returns None metrics when there is no signal.
    """
    graded = [c for c in case_dims if c.get("critical_pass") is not None]
    n = len(graded)
    if not n:
        return {
            "critical_pass_rate": None,
            "malformed_rate": None,
            "mean_rubric": None,
            "breakpoint_axis_level": None,
        }
    passed = sum(1 for c in graded if c["critical_pass"])
    malformed = sum(1 for c in graded if c.get("malformed"))
    rubrics = [c["rubric_score"] for c in graded if c.get("rubric_score") is not None]
    failed_levels = [
        c["axis_level"]
        for c in graded
        if not c["critical_pass"] and c.get("axis_level")
    ]
    breakpoint = (
        min(
            failed_levels,
            key=lambda a: _AXIS_ORDER.index(a) if a in _AXIS_ORDER else 99,
        )
        if failed_levels
        else None
    )
    return {
        "critical_pass_rate": round(passed / n, 6),
        "malformed_rate": round(malformed / n, 6),
        "mean_rubric": round(sum(rubrics) / len(rubrics), 6) if rubrics else None,
        "breakpoint_axis_level": breakpoint,
    }
