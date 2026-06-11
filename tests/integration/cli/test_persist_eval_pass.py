"""The CLI run-history persist records evaluation pass/fail, not mere execution.

Regression: a test that executed cleanly but was hard-gated to score 0 (failed
critical_check) was stored with success=True and counted toward passed_tests,
so the dashboard showed a green "Pass" and "100% success" on score-0 rows.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from atp.cli.main import _save_results_to_db
from atp.core.results import RunResult, SuiteResult, TestResult
from atp.dashboard import init_database
from atp.dashboard.models import SuiteExecution, TestExecution
from atp.loader.models import TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus

pytestmark = pytest.mark.anyio


def _completed_test(test_id: str) -> TestResult:
    """A test whose single run executed to completion (execution success)."""
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
        id=test_id, name=test_id, task=TaskDefinition(description="do the thing")
    )
    return TestResult(
        test=test,
        runs=[run],
        start_time=datetime.now(tz=UTC),
        end_time=datetime.now(tz=UTC),
    )


async def test_persist_records_eval_pass_not_execution(tmp_path, monkeypatch) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'hist.db'}"
    monkeypatch.setenv("ATP_DATABASE_URL", url)

    passed = _completed_test("t-pass")
    gated = _completed_test("t-gated")  # ran fine, but hard-gated below
    suite = SuiteResult(
        suite_name="s",
        agent_name="a",
        tests=[passed, gated],
        start_time=datetime.now(tz=UTC),
    )
    # Both runs completed (execution success True), but the scorer hard-gated
    # the second test to 0 (a failed critical_check → passed=False).
    scored = {
        "t-pass": SimpleNamespace(score=100.0, passed=True),
        "t-gated": SimpleNamespace(score=0.0, passed=False),
    }

    await _save_results_to_db(
        result=suite,
        suite_name="s",
        agent_name="a",
        runs_per_test=1,
        scored_results=scored,
        adapter="http",
        model="qwen2.5:7b",
    )

    db = await init_database(url=url)
    async with db.session() as session:
        suite_row = (await session.execute(select(SuiteExecution))).scalars().one()
        # Headline counts evaluation passes, not executions.
        assert suite_row.passed_tests == 1
        assert suite_row.failed_tests == 1
        assert suite_row.success_rate == 0.5

        tes = {
            te.test_id: te
            for te in (await session.execute(select(TestExecution))).scalars().all()
        }
        assert tes["t-pass"].success is True
        assert tes["t-pass"].score == 100.0
        # Executed but hard-gated → Fail, not Pass; status still "completed".
        assert tes["t-gated"].success is False
        assert tes["t-gated"].score == 0.0
        assert tes["t-gated"].status == "completed"
    await db.close()
