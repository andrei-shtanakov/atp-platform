"""CLI-side composition of the shared evaluation pipeline.

The CLI is the trusted, local execution context: the suite is the operator's
own, so every registered evaluator is permitted and artifacts are materialized
into the working directory, which is what `code_exec` evaluators need in order
to import the files under test.

This module is the composition root for that context — it is the only place
that knows both `atp.evaluators` (implementations) and `atp.evaluation`
(orchestration). The server plane composes the same pipeline with a
restrictive policy and a sandboxed workspace instead; neither context reaches
into the other's choices.

Behaviour here is deliberately identical to the inline block this replaced in
`atp/cli/main.py`, warning text included, so a run before and after the
extraction prints the same thing.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from atp.evaluation import (
    EvaluationOutcome,
    EvaluationPipeline,
    EvaluationPolicy,
    SkipReason,
)
from atp.loader.models import TestDefinition
from atp.protocol import ATPResponse

#: The CLI trusts its input, so no assertion type is withheld.
TRUSTED_LOCAL = EvaluationPolicy(name="trusted_local")


@contextlib.contextmanager
def materialize_artifacts(response: ATPResponse) -> Iterator[ATPResponse]:
    """Write inline artifacts to the working directory for the block's duration.

    `code_exec` evaluators (pytest, npm, lint) import the files under test from
    disk, so an artifact that only exists in the response is invisible to them.

    Local-only by design: the path comes from the agent's response, so this is
    safe exactly to the degree that the agent is trusted. The server plane must
    not reuse it — that is what the injected artifact context exists for.
    """
    written: list[Path] = []
    try:
        for artifact in response.artifacts:
            content = getattr(artifact, "content", None)
            artifact_path = getattr(artifact, "path", None)
            if content and artifact_path:
                file_path = Path(artifact_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
                written.append(file_path)
        # The CLI evaluates the response as-is: paths are the operator's own.
        yield response
    finally:
        for file_path in written:
            file_path.unlink(missing_ok=True)


def report_skips(
    test_id: str, outcome: EvaluationOutcome, echo: Callable[[str], None]
) -> None:
    """Print the same warnings the inline implementation printed.

    A guardrail skip is reported once per test rather than once per assertion:
    the pipeline records it against every assertion (it is per-assertion data),
    but the operator-facing fact is a single "this test was not evaluated".
    """
    guardrail_skips = [s for s in outcome.skipped if s.reason == SkipReason.GUARDRAIL]
    if guardrail_skips:
        echo(f"  Skipping evaluation for '{test_id}': {guardrail_skips[0].detail}")
        return

    for skipped in outcome.skipped:
        if skipped.reason in (SkipReason.EVALUATOR_ERROR, SkipReason.UNSUPPORTED):
            echo(
                f"  Warning: evaluator for '{skipped.assertion_type}' failed: "
                f"{skipped.detail}"
            )


def build_cli_pipeline() -> EvaluationPipeline:
    """The pipeline as the CLI composes it: full registry, local artifacts."""
    # Imported here rather than at module scope so that importing the CLI does
    # not construct every evaluator; the registry is built once per invocation.
    from atp.evaluators.guardrails import run_guardrails, should_skip_evaluation
    from atp.evaluators.registry import get_registry

    registry = get_registry()

    def guardrail(task: TestDefinition, response: ATPResponse) -> str | None:
        """Pre-evaluation gate: returns a reason to skip, or None."""
        return should_skip_evaluation(run_guardrails(task, response))

    return EvaluationPipeline(
        registry,
        TRUSTED_LOCAL,
        guardrail=guardrail,
        artifacts=materialize_artifacts,
    )


def score_outcome(
    test_id: str,
    test_def: TestDefinition,
    response: ATPResponse,
    outcome: EvaluationOutcome,
    echo: Callable[[str], None],
) -> Any | None:
    """Aggregate an outcome into a scored result, or None if nothing scored."""
    if not outcome.results:
        return None

    from atp.scoring.aggregator import ScoreAggregator

    try:
        aggregator = ScoreAggregator(weights=test_def.scoring)
        return aggregator.score_test_result(
            test_id=test_id,
            eval_results=outcome.results,
            response=response,
            max_steps=test_def.constraints.max_steps,
            max_tokens=test_def.constraints.max_tokens,
            max_cost_usd=test_def.constraints.budget_usd,
        )
    except Exception as exc:
        echo(f"  Warning: scoring for '{test_id}' failed: {exc}")
        return None
