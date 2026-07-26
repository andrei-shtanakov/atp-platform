"""Versioned meaning of the benchmark-plane score, for external consumers.

Maestro's `finalize()` reads `total_score` off `GET /api/v1/runs/{id}/status`
(`maestro/benchmark/atp_client.py`). Until now that number arrived bare, and a
bare number invites the reader to assume it means quality. It does not: on
this plane a task scores 100 when the agent returned a *completed* response,
whatever the response said — evaluators are not wired in here yet (see the
open "Benchmark API scoring" item in TODO.md).

So the wire carries the number *and* a versioned statement of what it means.
Two levels are described separately, because they are different quantities:

* per task — `TaskResultResponse.score`, a completion boolean expressed as
  0.0 or 100.0 (`BenchmarkService._evaluate_sync`);
* per run — `total_score`, the mean of those, i.e. a percentage
  (`BenchmarkService._finalize_run`).

`score_components` is the agreed wire shape for a future breakdown. It is an
empty map today and deliberately has **no database column**: storing `{}` on
every row would mint a second persistence representation next to the existing
`ScoreComponent` table (on the CLI/`TestExecution` plane) before the
storage-unification EPIC has chosen one. When the first real component is
computed, that EPIC picks the persistence model; this map does not change
shape, so consumers are unaffected.

Consumers must ignore unknown keys under `score_components` and unknown keys
inside `score_semantics` — that is what makes adding components additive.
A payload with no `score_semantics` at all is a legacy producer of unknown
semantics; it must never be read as a quality score.
"""

from __future__ import annotations

import copy
from typing import Any

SCORE_CONTRACT_VERSION = 1

_RUN_SCORE_SEMANTICS: dict[str, Any] = {
    "schema_version": SCORE_CONTRACT_VERSION,
    "kind": "completion_rate",
    "level": "run",
    "unit": "percent",
    "range": {"min": 0.0, "max": 100.0},
    # The single field a consumer should branch on before showing this to a
    # human or feeding it to a router.
    "quality_signal": False,
    "aggregation": {"function": "mean", "over": "task_score"},
    "task_score": {
        "kind": "completion_boolean",
        "level": "task",
        "unit": "percent",
        "values": [0.0, 100.0],
    },
    "caveats": [
        # Both are real behaviours of the current implementation, stated here
        # rather than left for a consumer to discover in production.
        "null_until_finalized: total_score is null until the run completes",
        "zero_is_ambiguous: a run that scored no tasks finalizes to 0.0, "
        "which is indistinguishable from every task failing",
    ],
    "note": (
        "Completion, not quality: a task scores 100 when the agent returned a "
        "completed response, regardless of what the response contained. "
        "Evaluators are not wired into the benchmark plane yet."
    ),
}


def run_score_semantics() -> dict[str, Any]:
    """What `total_score` means on the benchmark plane, versioned.

    Returns a deep copy: the constant is shared module state and a consumer
    mutating a response must not corrupt the next one.
    """
    return copy.deepcopy(_RUN_SCORE_SEMANTICS)


def empty_score_components() -> dict[str, Any]:
    """The score breakdown: an empty map until evaluators land on this plane.

    Not `None`: the key is always present so a consumer can iterate it
    unconditionally, and its emptiness is a fact about today's scoring rather
    than a missing field.
    """
    return {}
