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

#: Evaluators that address the *host* filesystem using a path taken from the
#: suite rather than from a sandbox handed to them.
#:
#: `filesystem` reads `workspace_path` straight out of `assertion.config`.
#: Server-side that is wrong twice over. It cannot measure the submission at
#: all — the submitted artifacts are materialized into a per-evaluation
#: sandbox the evaluator is never told about, so it inspects some unrelated
#: directory and reports a confident pass or fail about it. And the pass/fail
#: it reports is an existence answer about the server's own disk, handed to
#: whoever ran the benchmark. Creating a benchmark is admin-only today, so
#: that is a misconfiguration leak rather than a self-service oracle; the
#: first reason stands on its own regardless.
#:
#: Nothing in the pipeline can redirect that config — it is the evaluator's
#: own input — so withholding the evaluator is the only control available
#: until it accepts a sandbox instead of naming a directory.
READS_HOST_FILESYSTEM: Final[frozenset[str]] = frozenset({"filesystem"})

#: Evaluators that build their own sub-evaluators instead of receiving them.
#:
#: `CompositeEvaluator` resolves each leaf through `get_registry()` directly,
#: so an assertion the policy would refuse becomes reachable by nesting it:
#: `composite` → leaf `pytest` executes code that `EXECUTES_UNTRUSTED_INPUT`
#: exists to keep out. The pipeline's policy check happens before *it* resolves
#: an evaluator and cannot see inside one. Until a delegating evaluator is
#: handed the restricted resolver instead of reaching for the global registry,
#: allowing this class would make every other exclusion advisory.
DELEGATES_TO_REGISTRY: Final[frozenset[str]] = frozenset({"composite"})

#: Evaluators safe to run in-process against an untrusted submission: they
#: only inspect the response and the trace.
#:
#: Derived by subtraction, so adding an evaluator without classifying it makes
#: it *permitted* — which is why `test_every_evaluator_is_classified` exists.
DETERMINISTIC_EVALUATORS: Final[frozenset[str]] = frozenset(
    set(ASSERTION_TO_EVALUATOR.values())
    - EXECUTES_UNTRUSTED_INPUT
    - CALLS_EXTERNAL_SERVICE
    - READS_HOST_FILESYSTEM
    - DELEGATES_TO_REGISTRY
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
