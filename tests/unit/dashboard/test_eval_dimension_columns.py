"""SP-1: SuiteExecution/TestExecution carry the eval dimension columns."""

from atp.dashboard.models import SuiteExecution, TestExecution

CASE_COLS = {
    "axis_level",
    "capability",
    "family",
    "case_version",
    "critical_pass",
    "malformed",
    "recall",
    "precision",
    "fp_count",
    "rubric_score",
    "grader_version",
    "task_type",
    "language",
}
RUN_COLS = {
    "task_type",
    "run_uuid",
    "critical_pass_rate",
    "malformed_rate",
    "mean_rubric",
    "breakpoint_axis_level",
    "language",
}


def test_test_execution_has_case_columns() -> None:
    cols = set(TestExecution.__table__.columns.keys())
    assert CASE_COLS <= cols


def test_suite_execution_has_run_columns() -> None:
    cols = set(SuiteExecution.__table__.columns.keys())
    assert RUN_COLS <= cols
