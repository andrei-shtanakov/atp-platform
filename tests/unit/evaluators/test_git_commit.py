"""Tests for GitCommitEvaluator."""

import pytest

from atp.evaluators.git_commit import (
    GitCommitEvaluator,
    _compute_line_similarity,
    _extract_changed_files,
)
from atp.loader.models import Assertion
from atp.loader.models import TestDefinition as _TestDefinition
from atp.protocol import ATPResponse, ResponseStatus
from atp.protocol.models import ArtifactFile

SAMPLE_DIFF = """\
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
     pass
"""


class TestExtractChangedFiles:
    def test_extracts_files(self) -> None:
        files = _extract_changed_files(SAMPLE_DIFF)
        assert "src/main.py" in files

    def test_empty_diff(self) -> None:
        assert _extract_changed_files("") == set()


class TestLineSimilarity:
    def test_identical(self) -> None:
        assert _compute_line_similarity("a\nb\nc", "a\nb\nc") == 1.0

    def test_completely_different(self) -> None:
        score = _compute_line_similarity("a\nb\nc", "x\ny\nz")
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        score = _compute_line_similarity("a\nb\nc", "a\nb\nz")
        assert 0.0 < score < 1.0

    def test_empty_ground_truth(self) -> None:
        assert _compute_line_similarity("", "") == 1.0
        assert _compute_line_similarity("", "something") == 0.0


class TestGitCommitEvaluator:
    def test_name(self) -> None:
        e = GitCommitEvaluator()
        assert e.name == "git_commit"

    @pytest.mark.anyio
    async def test_no_ground_truth(self) -> None:
        e = GitCommitEvaluator()
        task = _TestDefinition(
            id="t1",
            name="test",
            task={"description": "reconstruct commit"},
        )
        response = ATPResponse(
            task_id="t1",
            status=ResponseStatus.COMPLETED,
            artifacts=[],
        )
        assertion = Assertion(type="git_commit")

        result = await e.evaluate(task, response, [], assertion)
        assert not result.passed
        assert "ground truth" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_no_agent_diff(self) -> None:
        e = GitCommitEvaluator()
        task = _TestDefinition(
            id="t1",
            name="test",
            task={"description": "reconstruct commit"},
        )
        response = ATPResponse(
            task_id="t1",
            status=ResponseStatus.COMPLETED,
            artifacts=[],
        )
        assertion = Assertion(
            type="git_commit",
            config={"expected": SAMPLE_DIFF},
        )

        result = await e.evaluate(task, response, [], assertion)
        assert not result.passed
        assert "no diff" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_identical_diff(self) -> None:
        e = GitCommitEvaluator()
        task = _TestDefinition(
            id="t1",
            name="test",
            task={"description": "reconstruct commit"},
        )
        response = ATPResponse(
            task_id="t1",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    type="file",
                    path="output.diff",
                    content=SAMPLE_DIFF,
                ),
            ],
        )
        assertion = Assertion(
            type="git_commit",
            config={"expected": SAMPLE_DIFF},
        )

        result = await e.evaluate(task, response, [], assertion)
        assert result.passed
        assert len(result.checks) == 4
        assert result.score > 0.8

    @pytest.mark.anyio
    async def test_ground_truth_from_config(self) -> None:
        """Ground truth provided via assertion config."""
        e = GitCommitEvaluator()
        task = _TestDefinition(
            id="t1",
            name="test",
            task={"description": "reconstruct commit"},
        )
        response = ATPResponse(
            task_id="t1",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    type="file",
                    path="output.diff",
                    content=SAMPLE_DIFF,
                ),
            ],
        )
        assertion = Assertion(
            type="git_commit",
            config={"expected": SAMPLE_DIFF},
        )

        result = await e.evaluate(task, response, [], assertion)
        assert result.passed
