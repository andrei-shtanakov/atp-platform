"""Pure mappers from a run's TestDefinition tags + eval results to the SP-1
dimension/outcome columns. No DB — unit-testable in isolation.

Method runs carry `level_*`/`capability_*`/`family_*`/`version_*` tags and a
`critical_check` EvalCheck. Both grading paths populate `critical_pass` (from the
check's `passed`); the checker (findings_match) path additionally carries a
CaseVerdict dump in `details` (`malformed`/`recall`/`precision`/`fp_count`/
`grader_version`), while the LLM-judge path leaves those null. Native runs have
no `critical_check`, so every field degrades to None (the columns are nullable).
"""

from typing import Any

from atp.core.results import EvalCheck, EvalResult
from atp.loader.models import TestDefinition

_AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]


def _tag_value(tags: list[str], prefix: str) -> str | None:
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix) :]
    return None


def _critical_check(results: list[EvalResult]) -> EvalCheck | None:
    """The critical_check EvalCheck, present on both judge and checker paths."""
    for r in results:
        for c in r.checks:
            if c.name == "critical_check":
                return c
    return None


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
    check = _critical_check(eval_results)
    # critical_pass works for BOTH grading paths via the check's `passed`; the
    # richer findings fields live only in the checker path's CaseVerdict `details`.
    v = (check.details or {}) if check else {}
    version = _tag_value(tags, "version_")
    return {
        "axis_level": _tag_value(tags, "level_"),
        "capability": _tag_value(tags, "capability_"),
        "family": _tag_value(tags, "family_"),
        "case_version": int(version) if version and version.isdigit() else None,
        "critical_pass": check.passed if check else None,
        "malformed": v.get("malformed"),
        "recall": v.get("recall"),
        "precision": v.get("precision"),
        "fp_count": v.get("fp_count"),
        "rubric_score": _rubric_score(eval_results),
        "grader_version": v.get("grader_version"),
        "task_type": _tag_value(tags, "task_type_"),
        "language": _tag_value(tags, "language_"),
    }


def aggregate_run(case_dims: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-case dims into the suite_executions run-level columns.

    Only cases with a non-null critical_pass count toward the rates (native
    runs contribute nothing). Returns None metrics when there is no signal.
    """

    def _common(key: str) -> str | None:
        vals = {c.get(key) for c in case_dims if c.get(key)}
        return next(iter(vals)) if len(vals) == 1 else None

    # Computed over ALL cases (not just graded) so an ungraded/native run still
    # records task_type/language when present.
    task_type = _common("task_type")
    language = _common("language")

    graded = [c for c in case_dims if c.get("critical_pass") is not None]
    n = len(graded)
    if not n:
        return {
            "critical_pass_rate": None,
            "malformed_rate": None,
            "mean_rubric": None,
            "breakpoint_axis_level": None,
            "task_type": task_type,
            "language": language,
        }
    passed = sum(1 for c in graded if c["critical_pass"])
    # malformed is findings-only: judge-graded cases report None, which must NOT
    # dilute the rate to a misleading 0.0 — compute over the non-null subset.
    malformable = [c for c in graded if c.get("malformed") is not None]
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
    malformed_rate = (
        round(sum(1 for c in malformable if c["malformed"]) / len(malformable), 6)
        if malformable
        else None
    )
    return {
        "critical_pass_rate": round(passed / n, 6),
        "malformed_rate": malformed_rate,
        "mean_rubric": round(sum(rubrics) / len(rubrics), 6) if rubrics else None,
        "breakpoint_axis_level": breakpoint,
        "task_type": task_type,
        "language": language,
    }
