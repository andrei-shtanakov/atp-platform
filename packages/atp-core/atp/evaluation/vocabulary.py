"""Assertion-type vocabulary: which assertion is served by which evaluator.

This is *data*, not implementation, so it lives at the bottom of the
dependency graph. Two consumers need it and only one of them may import
evaluator classes:

* `atp.evaluators.EvaluatorRegistry` (atp-platform) builds its assertion
  mappings from this table, so the vocabulary has one owner;
* the dashboard's suite-upload validation needs the set of known assertion
  types to warn on typos — a question about *names*, which must not drag
  evaluator implementations into the dashboard process.

The classification sets below describe evaluators by what they *do* rather
than by what they are called, so a server-side policy can refuse a whole class
instead of maintaining its own denylist that silently misses the next
evaluator someone adds.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

#: assertion type -> evaluator type.
ASSERTION_TO_EVALUATOR: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
        "artifact_exists": "artifact",
        "contains": "artifact",
        "schema": "artifact",
        "sections": "artifact",
        "behavior": "behavior",
        "must_use_tools": "behavior",
        "max_tool_calls": "behavior",
        "min_tool_calls": "behavior",
        "no_errors": "behavior",
        "forbidden_tools": "behavior",
        "llm_eval": "llm_judge",
        "code_exec": "code_exec",
        "pytest": "code_exec",
        "npm": "code_exec",
        "custom_command": "code_exec",
        "lint": "code_exec",
        "security": "security",
        "factuality": "factuality",
        "performance": "performance",
        "style": "style",
        "style_rules": "style",
        "tone": "style",
        "readability": "style",
        "passive_voice": "style",
        "sentence_length": "style",
        "file_exists": "filesystem",
        "file_not_exists": "filesystem",
        "file_contains": "filesystem",
        "file_count": "filesystem",
        "dir_exists": "filesystem",
        "composite": "composite",
        "findings_match": "findings_match",
    }
)

#: Evaluators that run code or commands supplied by, or derived from, the
#: agent's output. Unsafe outside an isolated worker.
EXECUTES_UNTRUSTED_INPUT: Final[frozenset[str]] = frozenset({"code_exec"})

#: Evaluators that reach a network service, which costs money per call and
#: makes evaluation non-deterministic.
CALLS_EXTERNAL_SERVICE: Final[frozenset[str]] = frozenset({"llm_judge", "factuality"})

#: Evaluators safe to run in-process against an untrusted submission: they
#: only inspect the response and the trace.
#:
#: Two members are here because a *class* stopped being true of them, not
#: because the class was relaxed.
#:
#: `composite` builds sub-evaluators through the resolver it is handed rather
#: than a global registry (ADR-008 track A). Its leaves are therefore filtered
#: by whatever policy governs it, and a leaf it may not evaluate is reported
#: as unevaluated rather than as a failure.
#:
#: `filesystem` used to be the sole member of a `READS_HOST_FILESYSTEM` class:
#: it took its root from `assertion.config`, so server-side it inspected some
#: directory of the server's own rather than the submission, and reported a
#: confident answer about it. It now receives the root from its composition
#: (ADR-008 track B) — the per-evaluation artifact sandbox on the server — and
#: cannot address anything outside it, so the class had no members left and
#: was removed rather than kept as an empty set nobody would notice.
#:
#: Derived by subtraction, so adding an evaluator without classifying it makes
#: it *permitted* — which is why `test_every_evaluator_is_classified` exists.
DETERMINISTIC_EVALUATORS: Final[frozenset[str]] = frozenset(
    set(ASSERTION_TO_EVALUATOR.values())
    - EXECUTES_UNTRUSTED_INPUT
    - CALLS_EXTERNAL_SERVICE
)


def known_assertion_types() -> frozenset[str]:
    """Every assertion type the platform understands."""
    return frozenset(ASSERTION_TO_EVALUATOR)


def deterministic_assertion_types() -> frozenset[str]:
    """Assertion types safe to evaluate in-process against untrusted input."""
    return frozenset(
        assertion
        for assertion, evaluator in ASSERTION_TO_EVALUATOR.items()
        if evaluator in DETERMINISTIC_EVALUATORS
    )
