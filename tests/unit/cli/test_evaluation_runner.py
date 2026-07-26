"""Parity tests for the CLI's composition of the shared pipeline.

The inline block these replace is gone, so the only remaining record of what
it printed and when is here. Message strings are asserted literally on
purpose: an operator greps these, and "Warning: evaluator for 'x' failed" is
part of the CLI's observable behaviour, not an implementation detail.

Taken from the pre-extraction `atp/cli/main.py`:

* a guardrail skip printed **once per test**, not once per assertion;
* an evaluator that raised printed a warning naming the assertion type, and
  the remaining assertions still ran;
* an unknown assertion type reached the same warning, because resolving it
  raised inside the same try;
* a scoring failure printed its own warning and left the test unscored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atp.cli.evaluation_runner import (
    TRUSTED_LOCAL,
    materialize_artifacts,
    report_skips,
    score_outcome,
)
from atp.evaluation import EvaluationOutcome, SkippedEvaluation, SkipReason
from atp.protocol import ATPResponse
from atp.protocol.models import ArtifactFile


def _collector() -> tuple[list[str], Any]:
    """An echo double that records what the CLI would print."""
    printed: list[str] = []
    return printed, printed.append


class TestSkipReporting:
    """Operator-facing messages must not drift from the inline original."""

    def test_guardrail_is_reported_once_per_test(self) -> None:
        """The pipeline records it per assertion; the operator sees one line."""
        outcome = EvaluationOutcome(
            skipped=[
                SkippedEvaluation("contains", SkipReason.GUARDRAIL, "budget exceeded"),
                SkippedEvaluation("schema", SkipReason.GUARDRAIL, "budget exceeded"),
            ]
        )
        printed, echo = _collector()
        report_skips("t1", outcome, echo)
        assert printed == ["  Skipping evaluation for 't1': budget exceeded"]

    def test_evaluator_error_names_the_assertion_type(self) -> None:
        outcome = EvaluationOutcome(
            skipped=[SkippedEvaluation("schema", SkipReason.EVALUATOR_ERROR, "boom")]
        )
        printed, echo = _collector()
        report_skips("t1", outcome, echo)
        assert printed == ["  Warning: evaluator for 'schema' failed: boom"]

    def test_unknown_type_uses_the_same_warning(self) -> None:
        """Pre-extraction, resolution failure fell into the same except."""
        outcome = EvaluationOutcome(
            skipped=[SkippedEvaluation("nope", SkipReason.UNSUPPORTED, "no evaluator")]
        )
        printed, echo = _collector()
        report_skips("t1", outcome, echo)
        assert printed == ["  Warning: evaluator for 'nope' failed: no evaluator"]

    def test_guardrail_suppresses_per_assertion_noise(self) -> None:
        """One clear reason beats a list of consequences of that reason."""
        outcome = EvaluationOutcome(
            skipped=[
                SkippedEvaluation("contains", SkipReason.GUARDRAIL, "timeout"),
                SkippedEvaluation("schema", SkipReason.EVALUATOR_ERROR, "boom"),
            ]
        )
        printed, echo = _collector()
        report_skips("t1", outcome, echo)
        assert printed == ["  Skipping evaluation for 't1': timeout"]

    def test_nothing_skipped_prints_nothing(self) -> None:
        printed, echo = _collector()
        report_skips("t1", EvaluationOutcome(applied=["contains"]), echo)
        assert printed == []


class TestArtifactMaterialization:
    """code_exec evaluators import files from disk; the CLI puts them there."""

    def _response(self, *artifacts: ArtifactFile) -> ATPResponse:
        return ATPResponse(task_id="t1", status="completed", artifacts=list(artifacts))

    def test_artifact_is_written_and_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        response = self._response(
            ArtifactFile(type="file", path="out/solution.py", content="print(1)\n")
        )
        with materialize_artifacts(response):
            assert (tmp_path / "out/solution.py").read_text() == "print(1)\n"
        assert not (tmp_path / "out/solution.py").exists()

    def test_cleanup_happens_when_the_body_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing evaluator must not leave files behind in the workspace."""
        monkeypatch.chdir(tmp_path)
        response = self._response(
            ArtifactFile(type="file", path="leftover.txt", content="x")
        )
        with pytest.raises(RuntimeError):
            with materialize_artifacts(response):
                raise RuntimeError("evaluator exploded")
        assert not (tmp_path / "leftover.txt").exists()

    def test_artifact_without_content_or_path_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        response = self._response(ArtifactFile(type="file", path="empty.txt"))
        with materialize_artifacts(response):
            assert not (tmp_path / "empty.txt").exists()

    def test_no_artifacts_is_a_no_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with materialize_artifacts(self._response()):
            pass
        assert list(tmp_path.iterdir()) == []


class TestScoring:
    """Scoring failures warn and leave the test unscored, as before."""

    def test_no_results_means_no_score_and_no_noise(self) -> None:
        printed, echo = _collector()
        scored = score_outcome("t1", object(), object(), EvaluationOutcome(), echo)
        assert scored is None and printed == []

    def test_scoring_failure_warns_and_returns_none(self) -> None:
        from atp.core.results import EvalCheck, EvalResult

        outcome = EvaluationOutcome(
            results=[
                EvalResult(
                    evaluator="e", checks=[EvalCheck(name="c", passed=True, score=1.0)]
                )
            ]
        )
        printed, echo = _collector()
        # A test_def missing `scoring`/`constraints` makes the aggregator raise,
        # which is the shape of a real scoring failure.
        scored = score_outcome("t1", object(), object(), outcome, echo)
        assert scored is None
        assert printed and printed[0].startswith("  Warning: scoring for 't1' failed:")


def test_cli_policy_withholds_nothing() -> None:
    """The CLI runs the operator's own suite; restricting it would be wrong."""
    assert TRUSTED_LOCAL.allowed_assertion_types is None
    assert TRUSTED_LOCAL.permits("code_exec") is True
