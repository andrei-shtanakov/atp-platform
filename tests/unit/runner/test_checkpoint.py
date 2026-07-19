"""Tests for suite checkpoint persistence."""

from datetime import UTC, datetime
from pathlib import Path

from atp.core.results import RunResult, TestResult
from atp.loader.models import (
    Constraints,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)
from atp.protocol import ATPResponse, ResponseStatus
from atp.runner.checkpoint import SuiteCheckpoint


def make_test(test_id: str) -> TestDefinition:
    return TestDefinition(
        id=test_id,
        name=f"Test {test_id}",
        task=TaskDefinition(description="do a thing"),
        constraints=Constraints(timeout_seconds=10),
    )


def make_result(test: TestDefinition) -> TestResult:
    run = RunResult(
        test_id=test.id,
        run_number=1,
        response=ATPResponse(task_id=test.id, status=ResponseStatus.COMPLETED),
        end_time=datetime.now(tz=UTC),
    )
    return TestResult(test=test, runs=[run], end_time=datetime.now(tz=UTC))


def make_suite(*tests: TestDefinition) -> TestSuite:
    return TestSuite(
        test_suite="cp-suite",
        tests=list(tests),
        defaults=TestDefaults(runs_per_test=1),
    )


def test_record_and_reload_round_trip(tmp_path: Path) -> None:
    test = make_test("t-1")
    cp = SuiteCheckpoint(tmp_path / "cp.json")
    cp.record(make_result(test))

    reloaded = SuiteCheckpoint(tmp_path / "cp.json")
    assert reloaded.completed_ids() == {"t-1"}
    results = reloaded.load_results(make_suite(test))
    assert len(results) == 1
    assert results[0].test.id == "t-1"
    assert results[0].success is True
    assert results[0].runs[0].response.status == ResponseStatus.COMPLETED


def test_load_ignores_tests_missing_from_suite(tmp_path: Path) -> None:
    stale = make_test("gone")
    cp = SuiteCheckpoint(tmp_path / "cp.json")
    cp.record(make_result(stale))
    reloaded = SuiteCheckpoint(tmp_path / "cp.json")
    assert reloaded.load_results(make_suite(make_test("t-2"))) == []


def test_delete_removes_file(tmp_path: Path) -> None:
    cp = SuiteCheckpoint(tmp_path / "cp.json")
    cp.record(make_result(make_test("t-1")))
    assert (tmp_path / "cp.json").exists()
    cp.delete()
    assert not (tmp_path / "cp.json").exists()
    cp.delete()  # idempotent


def test_default_path_slugifies(tmp_path: Path) -> None:
    p = SuiteCheckpoint.default_path("My Suite!", "agent/x", base_dir=tmp_path)
    assert p.parent == tmp_path
    assert p.name == "My-Suite--agent-x.json"
