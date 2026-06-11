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
from atp.dashboard import get_database
from atp.dashboard.models import SuiteExecution, TestExecution
from atp.loader.models import TaskDefinition, TestDefinition
from atp.protocol import ATPResponse, ResponseStatus

pytestmark = pytest.mark.anyio


def _test(test_id: str, status: ResponseStatus) -> TestResult:
    """A test whose single run ends in ``status`` (COMPLETED → execution OK)."""
    response = ATPResponse(version="1.0", task_id=test_id, status=status)
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

    passed = _test("t-pass", ResponseStatus.COMPLETED)
    gated = _test("t-gated", ResponseStatus.COMPLETED)  # ran, but hard-gated
    exec_failed = _test("t-execfail", ResponseStatus.FAILED)  # never completed
    suite = SuiteResult(
        suite_name="s",
        agent_name="a",
        tests=[passed, gated, exec_failed],
        start_time=datetime.now(tz=UTC),
    )
    scored = {
        "t-pass": SimpleNamespace(score=100.0, passed=True),
        # Ran fine but the scorer hard-gated it (failed critical_check).
        "t-gated": SimpleNamespace(score=0.0, passed=False),
        # Execution failed, yet the scorer marked it passed — success must
        # still be False (a Pass can't outrank a failed run).
        "t-execfail": SimpleNamespace(score=80.0, passed=True),
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

    # _save_results_to_db already initialised the global DB (against our
    # monkeypatched ATP_DATABASE_URL); reuse it rather than constructing a
    # second engine that would leak the first.
    db = get_database()
    async with db.session() as session:
        suite_row = (await session.execute(select(SuiteExecution))).scalars().one()
        # Headline counts evaluation passes (gated on execution), not executions.
        assert suite_row.passed_tests == 1
        assert suite_row.failed_tests == 2
        assert suite_row.success_rate == pytest.approx(1 / 3)

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
        # Execution failed → Fail regardless of the scorer's passed=True.
        assert tes["t-execfail"].success is False
        assert tes["t-execfail"].status == "failed"
    await db.close()
