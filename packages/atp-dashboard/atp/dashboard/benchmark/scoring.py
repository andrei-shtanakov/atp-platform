"""Scoring for the benchmark plane: what ran, what it produced, what it means.

Until now this plane scored 100 for a completed response and 0 otherwise, and
said so on the wire (`score_contract`). This module is where a submission can
actually be evaluated — under the server policy, in a bounded workspace, using
the shared pipeline the CLI runs — and where the resulting number is labelled
honestly enough that a consumer can tell the two apart.

Three rules decide the label, and they are all about evidence rather than
capability:

* **A wired resolver is not a score.** An app composed with evaluators still
  produces a completion score for a suite with no assertions, for a submission
  that did not complete, and for one whose every assertion the policy withheld.
  Only an evaluator that *ran* moves the needle.
* **Nothing that failed to run becomes a zero.** A skipped, refused, unknown
  or crashed evaluator is reported as coverage, never as a component worth
  0.0 — those read identically to a consumer and mean opposite things.
* **A mixed run says so.** When some tasks were evaluated and others were
  scored by completion, the mean of the two is still published, because that
  is what `total_score` has always been — but the composition travels with it,
  so the number is never an unlabelled blend.

Per-task detail is persisted in the existing, so-far-unused
`TaskResult.eval_results` column. No new column and no migration: the run-level
`score_components` map stays derived at read time, so a change in how it is
computed does not need a backfill, and the storage-unification EPIC still owns
the question of where evaluation results eventually live.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from atp.core.results import EvalResult
from atp.evaluation import (
    ArtifactWorkspace,
    EvaluationPipeline,
    FilteredResolver,
    SkipReason,
)
from atp.loader.models import TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from atp.dashboard.benchmark.score_contract import (
    AGGREGATED_EVALUATION,
    COMPLETION_RATE,
    run_score_semantics,
)

logger = logging.getLogger(__name__)

#: Version of the per-task record shape stored in `TaskResult.eval_results`.
#: Bumped when the shape changes; readers branch on it rather than guessing.
#:
#: Stamped on every record rather than once per task: the column is a
#: `list[dict]`, so there is no envelope to put it in, and a row written today
#: will be read by code written later — that is the whole reason the number
#: exists. A constant nobody writes down is a version scheme in name only.
EVALUATION_RECORD_VERSION = 1

#: Score for a task whose response completed, when no evaluator ran on it.
COMPLETION_SCORE = 100.0
#: Score for a task whose response did not complete. Evaluation is not even
#: attempted: an errored submission has nothing to assess, and running
#: evaluators over it would report failed assertions as if the agent had tried.
INCOMPLETE_SCORE = 0.0


class RecordStatus:
    """Status of one assertion in a stored task record. Consumers match these."""

    APPLIED = "applied"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class TaskScore:
    """The score for one submitted task, and the evidence behind it."""

    score: float
    quality_evaluated: bool
    #: Written to `TaskResult.eval_results`; None when nothing was attempted,
    #: which keeps a completion-only plane's rows exactly as they are today.
    records: list[dict[str, Any]] | None = None


def build_benchmark_pipeline(resolver: FilteredResolver) -> EvaluationPipeline:
    """Compose the pipeline as the benchmark plane runs it.

    Two differences from the CLI's composition, both because the input is a
    stranger's: the resolver arrives already restricted by the server policy,
    and artifacts are materialized into a throwaway bounded directory instead
    of the working directory.

    No guardrail is injected. The CLI's guardrail lives in `atp.evaluators`,
    which this package may not import, and the one precondition that matters
    here — a response that did not complete is not evaluated — is enforced by
    the caller, where it also decides the score.
    """
    workspace = ArtifactWorkspace()
    return EvaluationPipeline(
        resolver,
        resolver.policy,
        artifacts=workspace.prepare,
    )


async def score_submission(
    resolver: FilteredResolver | None,
    test_def: TestDefinition | None,
    response: ATPResponse,
    events: list[ATPEvent],
) -> TaskScore:
    """Score one submitted task, evaluating it when there is anything to run.

    `resolver` is None on a completion-only deployment; `test_def` is None when
    the suite has no test at this index, which is a malformed submission rather
    than a scoring question.
    """
    if response.status != "completed":
        return TaskScore(score=INCOMPLETE_SCORE, quality_evaluated=False)

    completion = TaskScore(score=COMPLETION_SCORE, quality_evaluated=False)
    if resolver is None or test_def is None or not test_def.assertions:
        return completion

    pipeline = build_benchmark_pipeline(resolver)
    outcome = await pipeline.evaluate(test_def, response, events)

    records = [
        _applied_record(assertion_type, result)
        for assertion_type, result in outcome.applied_results
    ]
    records.extend(
        {
            "record_version": EVALUATION_RECORD_VERSION,
            "assertion_type": skipped.assertion_type,
            "status": RecordStatus.SKIPPED,
            "reason": skipped.reason,
            "detail": skipped.detail,
        }
        for skipped in outcome.skipped
    )

    if not outcome.quality_evaluated:
        # Nothing ran. The score is the honest completion score, and the
        # records exist so the run can say *why* nothing ran rather than
        # looking like a suite that never asked for anything.
        return TaskScore(
            score=COMPLETION_SCORE, quality_evaluated=False, records=records or None
        )

    score = _aggregate_score(test_def, response, outcome.results)
    return TaskScore(score=score, quality_evaluated=True, records=records)


def _applied_record(assertion_type: str, result: EvalResult) -> dict[str, Any]:
    """One evaluator's outcome, flattened for storage."""
    return {
        "record_version": EVALUATION_RECORD_VERSION,
        "assertion_type": assertion_type,
        "status": RecordStatus.APPLIED,
        "evaluator": result.evaluator,
        "score": round(result.score, 4),
        "passed": result.passed,
        "critical": result.critical,
        "checks": [
            {"name": check.name, "passed": check.passed, "score": check.score}
            for check in result.checks
        ],
    }


def _aggregate_score(
    test_def: TestDefinition, response: ATPResponse, results: list[EvalResult]
) -> float:
    """Weighted 0-100 score, or the completion score if aggregation fails.

    Falling back rather than raising is deliberate: a submission that was
    evaluated must not 500 because the weighting step tripped. The caller has
    already established that at least one evaluator ran, so the label stays
    `aggregated_evaluation` and the components still report what was measured.
    """
    from atp.scoring.aggregator import ScoreAggregator

    try:
        aggregator = ScoreAggregator(weights=test_def.scoring)
        scored = aggregator.score_test_result(
            test_id=test_def.id,
            eval_results=results,
            response=response,
            max_steps=test_def.constraints.max_steps,
            max_tokens=test_def.constraints.max_tokens,
            max_cost_usd=test_def.constraints.budget_usd,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("scoring for '%s' failed: %s", test_def.id, exc, exc_info=True)
        return COMPLETION_SCORE
    return scored.score


# ----------------------------------------------------------------------
# Run-level view, derived from the stored per-task records
# ----------------------------------------------------------------------


@dataclass
class _Coverage:
    """Tally of what happened across a run's submitted tasks."""

    tasks_submitted: int = 0
    tasks_evaluated: int = 0
    #: Records stored in a shape this code does not know how to read.
    unreadable: int = 0
    applied: Counter[str] = field(default_factory=Counter)
    skipped: Counter[tuple[str, str]] = field(default_factory=Counter)
    scores: defaultdict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


def derive_run_score_view(
    task_records: list[list[dict[str, Any]] | None],
    tasks_total: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build `(score_semantics, score_components)` from stored task records.

    Derived rather than stored, so the meaning of a published number is
    recomputed from evidence on every read instead of being frozen at the
    moment a run finished.
    """
    coverage = _tally(task_records)
    components = _components(coverage)
    semantics = run_score_semantics(
        kind=AGGREGATED_EVALUATION if components else COMPLETION_RATE,
        quality_signal=bool(components),
        coverage=_coverage_view(coverage, tasks_total),
        mixed=bool(components) and coverage.tasks_evaluated < coverage.tasks_submitted,
    )
    return semantics, components


def _tally(task_records: list[list[dict[str, Any]] | None]) -> _Coverage:
    """Fold the per-task records into run-level counts.

    A record whose `record_version` this code does not recognise is counted
    and otherwise ignored. Reading unknown fields with today's expectations
    would turn a shape change into a quietly wrong measurement, and a wrong
    measurement is the one outcome the whole contract is built to avoid.
    """
    coverage = _Coverage(tasks_submitted=len(task_records))
    for records in task_records:
        evaluated = False
        for record in records or []:
            if record.get("record_version") != EVALUATION_RECORD_VERSION:
                coverage.unreadable += 1
                continue
            assertion_type = str(record.get("assertion_type", "unknown"))
            if record.get("status") == RecordStatus.APPLIED:
                evaluated = True
                coverage.applied[assertion_type] += 1
                coverage.scores[assertion_type].append(float(record.get("score", 0.0)))
            else:
                reason = str(record.get("reason", SkipReason.UNSUPPORTED))
                coverage.skipped[(assertion_type, reason)] += 1
        if evaluated:
            coverage.tasks_evaluated += 1
    return coverage


def _components(coverage: _Coverage) -> dict[str, float]:
    """Mean score per assertion type, over the tasks where it actually ran.

    Keys are assertion types rather than evaluator names: that is what the
    suite author wrote, and several assertion types share one evaluator, so
    evaluator keys would silently merge distinct measurements.

    Only successfully applied types appear. A withheld or crashed evaluator is
    absent, never present with 0.0 — `dict[str, float]` is what the consumer
    reads (`maestro/benchmark/models.py`), and in that shape a zero is a
    measurement.
    """
    return {
        assertion_type: round(sum(scores) / len(scores) * 100, 2)
        for assertion_type, scores in sorted(coverage.scores.items())
        if scores
    }


def _coverage_view(coverage: _Coverage, tasks_total: int) -> dict[str, Any]:
    """Coverage as a consumer reads it: counts, then per-type detail."""
    return {
        "tasks_total": tasks_total,
        "tasks_submitted": coverage.tasks_submitted,
        "tasks_evaluated": coverage.tasks_evaluated,
        "tasks_completion_only": coverage.tasks_submitted - coverage.tasks_evaluated,
        # Stored under a record shape this code cannot read. Reported rather
        # than dropped: silently missing evidence reads as evidence of absence.
        "records_unreadable": coverage.unreadable,
        "assertions_applied": dict(sorted(coverage.applied.items())),
        "assertions_skipped": [
            {"assertion_type": assertion_type, "reason": reason, "count": count}
            for (assertion_type, reason), count in sorted(coverage.skipped.items())
        ],
    }
