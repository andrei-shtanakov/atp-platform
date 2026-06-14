"""SP-1: `_save_results_to_db` persists eval dimensions + run aggregates.

A method-style case carries `level_*`/`capability_*`/`family_*`/`version_*`
tags and a `critical_check`/`rubric` EvalResult pair. These must land in the
`test_executions` dimension columns, and their roll-up must land in the
`suite_executions` run-level columns (run_uuid + critical_pass_rate + ...).
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from atp.cli.main import _save_results_to_db
from atp.core.results import EvalCheck, EvalResult, RunResult, SuiteResult, TestResult
from atp.dashboard import get_database
from atp.dashboard.models import SuiteExecution, TestExecution
from atp.loader.models import TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus

pytestmark = pytest.mark.anyio


def _method_test(test_id: str) -> TestResult:
    """A method-style case (COMPLETED run) with SP-1 dimension tags."""
    response = ATPResponse(
        version="1.0", task_id=test_id, status=ResponseStatus.COMPLETED
    )
    run = RunResult(
        test_id=test_id,
        run_number=1,
        response=response,
        start_time=datetime.now(tz=UTC),
        end_time=datetime.now(tz=UTC),
    )
    test = TestDefinition(
        id=test_id,
        name=test_id,
        task=TaskDefinition(description="do the thing"),
        tags=[
            "level_moderate",
            "capability_safety_compliance",
            "family_fam",
            "version_2",
        ],
    )
    return TestResult(
        test=test,
        runs=[run],
        start_time=datetime.now(tz=UTC),
        end_time=datetime.now(tz=UTC),
    )


async def test_persist_records_dimensions(tmp_path, monkeypatch) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'dims.db'}"
    monkeypatch.setenv("ATP_DATABASE_URL", url)

    test_id = "t-method"
    test_result = _method_test(test_id)
    suite = SuiteResult(
        suite_name="s",
        agent_name="a",
        tests=[test_result],
        start_time=datetime.now(tz=UTC),
    )

    # critical_check.details carries the CaseVerdict dump; rubric.score the score.
    critical = EvalResult(
        evaluator="findings_match",
        checks=[
            EvalCheck(
                name="critical_check",
                passed=True,
                score=1.0,
                details={
                    "critical_pass": True,
                    "malformed": False,
                    "recall": 1.0,
                    "precision": 1.0,
                    "fp_count": 0,
                    "grader_version": "findings_match@1",
                },
            )
        ],
    )
    rubric = EvalResult(
        evaluator="rubric",
        checks=[EvalCheck(name="rubric", passed=True, score=0.8)],
    )
    all_eval_results = {test_id: [critical, rubric]}
    scored = {test_id: SimpleNamespace(score=100.0, passed=True)}

    await _save_results_to_db(
        result=suite,
        suite_name="s",
        agent_name="a",
        runs_per_test=1,
        scored_results=scored,
        adapter="cli",
        model="claude",
        eval_results=all_eval_results,
    )

    db = get_database()
    async with db.session() as session:
        te = (await session.execute(select(TestExecution))).scalars().one()
        assert te.axis_level == "moderate"
        assert te.capability == "safety_compliance"
        assert te.family == "fam"
        assert te.case_version == 2
        assert te.critical_pass is True
        assert te.malformed is False
        assert te.rubric_score == 0.8
        assert te.grader_version == "findings_match@1"

        suite_row = (await session.execute(select(SuiteExecution))).scalars().one()
        assert suite_row.run_uuid
        assert suite_row.critical_pass_rate is not None
        assert suite_row.critical_pass_rate == pytest.approx(1.0)
        # One passing case → no breakpoint axis level.
        assert suite_row.breakpoint_axis_level is None
    await db.close()
