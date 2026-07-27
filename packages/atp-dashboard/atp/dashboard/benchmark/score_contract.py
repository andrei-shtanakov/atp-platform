"""Versioned meaning of the benchmark-plane score, for external consumers.

Maestro's `finalize()` reads `total_score` off `GET /api/v1/runs/{id}/status`
(`maestro/benchmark/atp_client.py`). A bare number invites the reader to assume
it means quality, and on this plane that assumption used to be wrong in every
case: a task scored 100 when the agent returned a *completed* response,
whatever the response said.

It is now wrong only in *some* cases, which is worse unless the wire says
which. So `score_semantics` reports what the number actually is:

* `completion_rate` — no evaluator ran. The number counts completions, exactly
  as before, and `quality_signal` is false.
* `aggregated_evaluation` — at least one evaluator ran and produced a result.
  `score_components` carries the per-assertion-type breakdown, and only the
  parts that were successfully measured appear in it.

`quality_signal` is the single field to branch on. It is true only when an
evaluator was *applied* — not when one was permitted, resolved, or attempted.
A server wired with evaluators still publishes `completion_rate` for a suite
that asserts nothing, because capability is not evidence.

`coverage` says what was requested, applied and skipped, each skip with a
reason. That is what separates "assessed and scored low" from "never
assessed"; without it a withheld evaluator is indistinguishable from a failed
one, and both look like zero quality.

Consumers must ignore unknown keys under `score_components` and unknown keys
inside `score_semantics` — that is what makes adding components additive.
A payload with no `score_semantics` at all is a legacy producer of unknown
semantics; it must never be read as a quality score.
"""

from __future__ import annotations

import copy
from typing import Any

SCORE_CONTRACT_VERSION = 1

#: The number counts completed responses. No evaluator contributed to it.
COMPLETION_RATE = "completion_rate"
#: At least one evaluator ran; the number is a weighted aggregate.
AGGREGATED_EVALUATION = "aggregated_evaluation"

#: Always true of `total_score`, in either kind.
_BASE_CAVEATS = (
    "null_until_finalized: total_score is null until the run completes",
    "zero_is_ambiguous: a run that scored no tasks finalizes to 0.0, "
    "which is indistinguishable from every task failing",
)

#: Added when a run's tasks were not all scored the same way. The mean is
#: still published — it is what total_score has always been — but a blend of
#: two different quantities must not travel unlabelled.
_MIXED_CAVEAT = (
    "mixed_task_scores: some tasks were scored by evaluation and others by "
    "completion; see coverage.tasks_evaluated and coverage.tasks_completion_only"
)

_NOTES: dict[str, str] = {
    COMPLETION_RATE: (
        "Completion, not quality: a task scores 100 when the agent returned a "
        "completed response, regardless of what the response contained. No "
        "evaluator ran on this run."
    ),
    AGGREGATED_EVALUATION: (
        "At least one evaluator ran. score_components lists the assertion "
        "types that were successfully measured; anything absent from it was "
        "not measured, and is not a zero."
    ),
}

_TASK_SCORE: dict[str, Any] = {
    "kind": "completion_boolean",
    "level": "task",
    "unit": "percent",
    "values": [0.0, 100.0],
}


def run_score_semantics(
    *,
    kind: str = COMPLETION_RATE,
    quality_signal: bool = False,
    coverage: dict[str, Any] | None = None,
    mixed: bool = False,
) -> dict[str, Any]:
    """What `total_score` means for one run, versioned.

    Defaults describe a run on which nothing was evaluated, which is both the
    honest answer for a run with no results yet and the only answer a
    completion-only deployment can give.
    """
    caveats = list(_BASE_CAVEATS)
    if mixed:
        caveats.append(_MIXED_CAVEAT)

    semantics: dict[str, Any] = {
        "schema_version": SCORE_CONTRACT_VERSION,
        "kind": kind,
        "level": "run",
        "unit": "percent",
        "range": {"min": 0.0, "max": 100.0},
        # The single field a consumer should branch on before showing this to
        # a human or feeding it to a router.
        "quality_signal": quality_signal,
        "aggregation": {"function": "mean", "over": "task_score"},
        "task_score": copy.deepcopy(_TASK_SCORE),
        "caveats": caveats,
        "note": _NOTES[kind],
    }
    if coverage is not None:
        semantics["coverage"] = coverage
    return semantics


def empty_score_components() -> dict[str, Any]:
    """The score breakdown for a run on which no evaluator ran.

    Not `None`: the key is always present so a consumer can iterate it
    unconditionally, and its emptiness is a fact about the run rather than a
    missing field.
    """
    return {}
