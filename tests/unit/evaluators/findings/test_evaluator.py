"""Tests for the FindingsMatchEvaluator (R-07 Phase-1b #2)."""

import json

import pytest

from atp.evaluators.findings.evaluator import FindingsMatchEvaluator
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol.models import ArtifactFile, ATPResponse, ResponseStatus

pytestmark = pytest.mark.anyio

EXPECTED_FINDINGS = [
    {
        "rule_ids": ["SEC-011", "sql-injection", "cwe-89"],
        "anchor": 'f"SELECT',
        "severity": "critical",
    }
]
MUST_NOT_FLAG = [
    {"anchor": "cursor.execute(query, (user_id,))"},
]


def _make_task() -> TestDefinition:
    return TestDefinition(
        id="test-findings-eval",
        name="Test FindingsMatch Evaluator",
        task=TaskDefinition(description="Review the code for SQL injection"),
    )


def _make_response(content: str) -> ATPResponse:
    artifact = ArtifactFile(path="findings.json", content=content)
    return ATPResponse(
        task_id="test-findings-eval",
        status=ResponseStatus.COMPLETED,
        artifacts=[artifact],
    )


def _make_assertion(**kwargs: object) -> Assertion:
    return Assertion(
        type="findings_match",
        critical=True,
        config={
            "expected_findings": EXPECTED_FINDINGS,
            "must_not_flag": MUST_NOT_FLAG,
            **kwargs,
        },
    )


@pytest.fixture
def evaluator() -> FindingsMatchEvaluator:
    return FindingsMatchEvaluator()


async def test_matching_finding_passes(evaluator: FindingsMatchEvaluator) -> None:
    """A finding with a synonym rule_id and matching anchor should pass."""
    findings_json = json.dumps(
        [
            {
                "rule_id": "cwe-89",
                "file": "app.py",
                "anchor": 'query = f"SELECT * FROM users WHERE id = {user_id}"',
                "severity": "critical",
            }
        ]
    )
    task = _make_task()
    response = _make_response(findings_json)
    assertion = _make_assertion()

    result = await evaluator.evaluate(task, response, [], assertion)

    assert result.passed is True
    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.passed is True
    assert check.details is not None
    assert check.details["recall"] == 1.0


async def test_unparseable_content_fails(evaluator: FindingsMatchEvaluator) -> None:
    """Non-JSON prose content should produce a failed, malformed result."""
    task = _make_task()
    response = _make_response("I found a SQL injection issue on line 42.")
    assertion = _make_assertion()

    result = await evaluator.evaluate(task, response, [], assertion)

    assert result.passed is False
    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.passed is False
    assert check.message is not None
    assert "malformed" in check.message.lower()
    assert check.details is not None
    assert check.details["malformed"] is True


async def test_finding_missing_severity_is_malformed(
    evaluator: FindingsMatchEvaluator,
) -> None:
    """A finding that omits the required severity field is malformed — distinct
    from a missed defect, so the routing signal isn't conflated."""
    findings_json = json.dumps([{"rule_id": "cwe-89", "anchor": 'f"SELECT'}])
    result = await evaluator.evaluate(
        _make_task(), _make_response(findings_json), [], _make_assertion()
    )
    check = result.checks[0]
    assert check.passed is False
    assert check.details is not None
    assert check.details["malformed"] is True
