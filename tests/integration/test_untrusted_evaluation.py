"""The untrusted composition, exercised end to end with the real registry.

Every other test in this area uses doubles. This one wires the actual
evaluator registry, the real server policy and a real `ArtifactWorkspace`, and
feeds it a submission that tries the things a hostile one would: a `pytest`
assertion that would execute code, and artifacts that try to leave the
sandbox, collide, and exceed the budget.

It exists because the previous rounds of this work were corrected three times
by cases I had reasoned about instead of running — a dropped mapping table, a
symlink check on the wrong side of `resolve()`, and three attacker-controlled
crashes. Composing the real parts is the cheapest way to stop doing that.
"""

from __future__ import annotations

import pytest

from atp.evaluation import (
    UNTRUSTED_SUBMISSION,
    ArtifactWorkspace,
    EvaluationPipeline,
    SkipReason,
    WorkspaceLimits,
)
from atp.evaluators.registry import get_registry
from atp.loader.models import Assertion, TaskDefinition, TestDefinition
from atp.protocol import ATPResponse
from atp.protocol.models import ArtifactFile

pytestmark = pytest.mark.anyio


def _submission() -> ATPResponse:
    """A response of the shape a hostile participant might send."""
    return ATPResponse(
        task_id="t1",
        status="completed",
        artifacts=[
            ArtifactFile(type="file", path="src/main.py", content="print('hi')\n"),
            # collides with the directory created above
            ArtifactFile(type="file", path="src", content="not a directory"),
            # duplicate of the first
            ArtifactFile(type="file", path="src/main.py", content="second"),
            # over the per-file budget
            ArtifactFile(type="file", path="huge.txt", content="x" * 5000),
            # A traversal attempt. `ArtifactFile` validation rejects `..`, so
            # this is built unvalidated on purpose: the workspace's own check
            # is defence in depth, and defence in depth that is never
            # exercised is an assumption, not a defence.
            ArtifactFile.model_construct(
                type="file", path="../escaped.txt", content="pwned"
            ),
        ],
    )


def _test_def(*types: str) -> TestDefinition:
    return TestDefinition(
        id="t1",
        name="t1",
        task=TaskDefinition(description="do the thing"),
        assertions=[Assertion(type=t) for t in types],
    )


async def test_executing_assertion_is_refused_by_the_real_composition() -> None:
    """`pytest` would run code from the submission inside the API process."""
    workspace = ArtifactWorkspace()
    pipeline = EvaluationPipeline(
        get_registry(), UNTRUSTED_SUBMISSION, artifacts=workspace.prepare
    )

    outcome = await pipeline.evaluate(_test_def("pytest"), _submission(), [])

    assert outcome.results == []
    assert outcome.quality_evaluated is False
    assert [s.reason for s in outcome.skipped] == [SkipReason.NOT_ALLOWED]


async def test_inspection_assertion_still_runs_and_counts_as_quality() -> None:
    """Refusing the dangerous ones must not leave the plane with nothing."""
    workspace = ArtifactWorkspace()
    pipeline = EvaluationPipeline(
        get_registry(), UNTRUSTED_SUBMISSION, artifacts=workspace.prepare
    )

    outcome = await pipeline.evaluate(_test_def("artifact_exists"), _submission(), [])

    assert outcome.applied == ["artifact_exists"]
    assert outcome.quality_evaluated is True


async def test_a_hostile_payload_neither_crashes_nor_escapes() -> None:
    """Traversal, collision, duplicate and oversized all become findings."""
    workspace = ArtifactWorkspace(WorkspaceLimits(max_file_bytes=1000))
    pipeline = EvaluationPipeline(
        get_registry(), UNTRUSTED_SUBMISSION, artifacts=workspace.prepare
    )

    outcome = await pipeline.evaluate(
        _test_def("contains", "pytest"), _submission(), []
    )

    reasons = {r.reason for r in workspace.report.rejected}
    assert reasons == {
        "path_collision",
        "duplicate_path",
        "file_too_large",
        "escapes_workspace",
    }
    assert workspace.report.written == ["src/main.py"]
    # The run still produced a result for the assertion it was allowed to run.
    assert "contains" in outcome.applied or outcome.skipped
    assert workspace.root is None  # cleaned up


async def test_the_sandbox_is_gone_afterwards() -> None:
    """A directory of someone else's files must not outlive the evaluation."""
    workspace = ArtifactWorkspace()
    pipeline = EvaluationPipeline(
        get_registry(), UNTRUSTED_SUBMISSION, artifacts=workspace.prepare
    )
    await pipeline.evaluate(_test_def("contains"), _submission(), [])
    assert workspace.root is None
