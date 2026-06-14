"""SP-1: pure extractor mapping tags + eval results -> dimension columns."""

from atp.core.results import EvalCheck, EvalResult
from atp.dashboard.dimensions import aggregate_run, case_dimensions
from atp.loader.models import TaskDefinition, TestDefinition


def _td(tags: list[str]) -> TestDefinition:
    return TestDefinition(
        id="case-1",
        name="f (moderate)",
        tags=tags,
        task=TaskDefinition(description="x"),
    )


def _critical(details: dict) -> EvalResult:
    return EvalResult(
        evaluator="agent_eval_case",
        checks=[
            EvalCheck(
                name="critical_check",
                passed=details["critical_pass"],
                score=1.0 if details["critical_pass"] else 0.0,
                message="",
                details=details,
            )
        ],
    )


def _rubric(score: float) -> EvalResult:
    return EvalResult(
        evaluator="agent_eval_case",
        checks=[EvalCheck(name="rubric", passed=True, score=score, message="")],
    )


def test_case_dimensions_from_tags_and_verdict() -> None:
    td = _td(
        [
            "level_moderate",
            "capability_safety_compliance",
            "family_code_review_planted_defect",
            "version_2",
        ]
    )
    verdict = {
        "critical_pass": True,
        "malformed": False,
        "recall": 1.0,
        "precision": 1.0,
        "false_positives": [],
        "fp_count": 0,
        "grader_version": "findings_match@1",
    }
    dims = case_dimensions(td, [_critical(verdict), _rubric(0.8)])
    assert dims["axis_level"] == "moderate"
    assert dims["capability"] == "safety_compliance"
    assert dims["family"] == "code_review_planted_defect"
    assert dims["case_version"] == 2
    assert dims["critical_pass"] is True
    assert dims["malformed"] is False
    assert dims["recall"] == 1.0
    assert dims["fp_count"] == 0
    assert dims["rubric_score"] == 0.8
    assert dims["grader_version"] == "findings_match@1"


def test_case_dimensions_judge_path_sets_critical_pass_without_details() -> None:
    # The LLM-judge critical_check has NO details (no CaseVerdict dump), but
    # critical_pass must still come from the check's `passed` — else whole
    # judge-graded families persist null and drop out of the run aggregates.
    td = _td(["level_clean", "capability_calibration", "family_req_extraction"])
    judge_critical = EvalResult(
        evaluator="agent_eval_case",
        checks=[EvalCheck(name="critical_check", passed=True, score=1.0, message="")],
    )
    dims = case_dimensions(td, [judge_critical, _rubric(0.7)])
    assert dims["critical_pass"] is True
    assert dims["malformed"] is None  # findings-only field, null for judge path
    assert dims["recall"] is None
    assert dims["rubric_score"] == 0.7
    assert aggregate_run([dims])["critical_pass_rate"] == 1.0


def test_case_dimensions_native_run_is_all_none() -> None:
    dims = case_dimensions(_td([]), [])
    assert dims["axis_level"] is None
    assert dims["critical_pass"] is None
    assert dims["case_version"] is None


def test_aggregate_run_rates_and_breakpoint() -> None:
    cases = [
        {
            "axis_level": "clean",
            "critical_pass": True,
            "malformed": False,
            "rubric_score": 0.9,
        },
        {
            "axis_level": "moderate",
            "critical_pass": False,
            "malformed": False,
            "rubric_score": 0.4,
        },
        {
            "axis_level": "severe",
            "critical_pass": False,
            "malformed": True,
            "rubric_score": 0.0,
        },
    ]
    agg = aggregate_run(cases)
    assert agg["critical_pass_rate"] == round(1 / 3, 6)
    assert agg["malformed_rate"] == round(1 / 3, 6)
    assert agg["mean_rubric"] == round((0.9 + 0.4 + 0.0) / 3, 6)
    assert agg["breakpoint_axis_level"] == "moderate"


def test_aggregate_run_empty_is_none() -> None:
    agg = aggregate_run([])
    assert agg["critical_pass_rate"] is None
    assert agg["breakpoint_axis_level"] is None
