"""Dependency-neutral evaluation pipeline.

The assertion→evaluator→score sequence used to live inline in
`atp/cli/main.py`, which meant the benchmark plane could only get real scoring
by growing a second copy. This module owns the orchestration; it deliberately
owns no evaluator implementations, so it can sit in atp-core and be used by
both the CLI (trusted, local, broad evaluator set) and the server (untrusted
submission, deterministic allowlist) under different policies.

Everything implementation-specific arrives by injection:

* `EvaluatorResolver` — builds an evaluator for an assertion type. The real
  registry lives in atp-platform and is passed in from there.
* `guardrail` — the pre-evaluation gate, likewise platform-side.
* `artifacts` — a context manager that makes response artifacts reachable on
  disk. The CLI materializes into the working directory; the server will use a
  bounded sandbox. Filesystem policy is not core's business.

The artifact context also names the directory it materialized into, and that
root is handed to any evaluator that addresses the filesystem. An evaluator
must never derive a root from the suite: on the server the suite and the
submission come from different people, and a root taken from the suite points
at the server's own disk rather than at the submission (ADR-008 track B).

Two behaviours matter more than they look:

* **A disallowed evaluator is never resolved**, only reported. Checking the
  policy after construction would still import and instantiate the thing the
  policy exists to keep out of the process.
* **An unknown assertion type is an explicit `unsupported` outcome**, never a
  silent fallback to some default evaluator. A submission scored by an
  evaluator nobody chose is worse than one openly left unscored.

The outcome lists what ran and what did not, with reasons, so a consumer can
tell "assessed and scored low" from "never assessed" — otherwise an unscored
submission reads as zero quality.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from atp.core.results import EvalResult
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

logger = logging.getLogger(__name__)


class SkipReason:
    """Why an assertion was not evaluated. Stable strings; consumers match."""

    NOT_ALLOWED = "not_allowed_by_policy"
    UNSUPPORTED = "unsupported_assertion_type"
    GUARDRAIL = "guardrail_precondition_failed"
    EVALUATOR_ERROR = "evaluator_error"
    #: The evaluator ran and declined to reach a verdict — distinct from
    #: crashing, and very distinct from deciding "fail".
    INDETERMINATE = "indeterminate"


class AssertionUnevaluated(Exception):
    """Raised by an evaluator that ran but could not reach a verdict.

    The alternative is for such an evaluator to invent one, and the invented
    verdict is always "fail": a score of zero, indistinguishable from a real
    measurement of zero. `CompositeEvaluator` raises this when its operator
    cannot be decided from the leaves it was allowed to evaluate.
    """


@runtime_checkable
class EvaluatorLike(Protocol):
    """Structural type of an evaluator, so core need not import the base class."""

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate one assertion against an agent response."""
        ...


class EvaluatorResolver(Protocol):
    """Builds evaluators. Implemented by the platform registry."""

    def create_for_assertion(self, assertion_type: str) -> EvaluatorLike:
        """Return an evaluator for this assertion type, or raise if unknown."""
        ...


@runtime_checkable
class ResolverAware(Protocol):
    """An evaluator that builds sub-evaluators and must be told which resolver.

    Only `CompositeEvaluator` implements this today. It exists because an
    evaluator that reaches for a global registry evaluates its children under
    *no* policy — which is how a `composite` wrapping `pytest` used to walk
    past the server allowlist entirely.
    """

    def bind_resolver(self, resolver: EvaluatorResolver) -> None:
        """Receive the resolver this evaluator must build children through."""
        ...


def bind_resolver(evaluator: object, resolver: EvaluatorResolver) -> None:
    """Hand a freshly resolved evaluator the resolver that produced it.

    One rule, applied at every site that resolves an evaluator: whoever
    resolved it passes on the resolver *it* was governed by. A composite
    nested three levels deep is then restricted exactly as its root was,
    without any level having to remember to re-apply the policy.
    """
    if isinstance(evaluator, ResolverAware):
        evaluator.bind_resolver(resolver)


@runtime_checkable
class WorkspaceAware(Protocol):
    """An evaluator that addresses the filesystem and must be told where.

    The root is a grant, not a hint: an evaluator holding one may look inside
    it and nowhere else. `trusted` says whether the surrounding plane is the
    operator's own machine, which is the only place a suite may still name a
    root of its own — see `FilesystemEvaluator` for what that legacy form
    costs and how long it lives.
    """

    def bind_workspace_root(self, root: Path, *, trusted: bool = False) -> None:
        """Receive the directory this evaluator may address, and nothing else."""
        ...


def bind_workspace_root(
    evaluator: object, root: Path | None, *, trusted: bool = False
) -> None:
    """Hand a freshly resolved evaluator the root it is allowed to address.

    Mirrors :func:`bind_resolver`, and for the same reason: applied at every
    site that resolves an evaluator, so a leaf nested inside a composite is
    confined exactly as its root was.

    A `None` root binds nothing. That is not an oversight to paper over — an
    evaluator left unrooted refuses to answer, which is the correct outcome
    when no composition offered it a directory.
    """
    if root is not None and isinstance(evaluator, WorkspaceAware):
        evaluator.bind_workspace_root(root, trusted=trusted)


#: Returns a reason to skip evaluation entirely, or None to proceed.
GuardrailFn = Callable[[TestDefinition, ATPResponse], str | None]


@dataclass(frozen=True)
class PreparedResponse:
    """The response evaluators should see, and where its files live.

    Carrying the root here rather than letting evaluators find it themselves
    is the whole point: it exists only for the duration of one evaluation, and
    the only object that knows it is the context manager that made it.
    """

    response: ATPResponse
    #: Directory the artifacts were materialized into, or None when nothing
    #: was written to disk at all.
    root: Path | None = None


#: Makes response artifacts reachable on disk for the duration of the block,
#: and yields the response evaluators should see plus the root it lives in. A
#: sandboxing implementation returns a rewritten response whose paths are
#: workspace-relative, so an agent-chosen path can never reach an evaluator.
ArtifactContext = Callable[
    [ATPResponse], contextlib.AbstractContextManager[PreparedResponse]
]


@contextlib.contextmanager
def no_artifacts(response: ATPResponse) -> Iterator[PreparedResponse]:
    """Default artifact context: touch nothing on disk, change nothing.

    Yields no root, so a filesystem evaluator composed this way declines to
    answer rather than reaching for the working directory.
    """
    yield PreparedResponse(response)


@dataclass(frozen=True)
class EvaluationPolicy:
    """What an execution context permits.

    `allowed_assertion_types=None` means "no restriction" and is only
    appropriate where the input is trusted.

    `trusted` is declared separately rather than inferred from an empty
    allowlist. They answer different questions — *which* evaluators may run
    versus *whose* machine this is — and a plane that one day restricts the
    CLI's evaluator set must not silently stop being the operator's own
    machine as a side effect.
    """

    name: str
    allowed_assertion_types: frozenset[str] | None = None
    #: True where the suite and the machine belong to the same person.
    trusted: bool = False

    def permits(self, assertion_type: str) -> bool:
        """True if this context may evaluate the assertion type."""
        return (
            self.allowed_assertion_types is None
            or assertion_type in self.allowed_assertion_types
        )


@dataclass(frozen=True)
class SkippedEvaluation:
    """One assertion that was not evaluated, and why."""

    assertion_type: str
    reason: str
    detail: str | None = None


@dataclass
class EvaluationOutcome:
    """What the pipeline actually did, not merely what it produced."""

    results: list[EvalResult] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    skipped: list[SkippedEvaluation] = field(default_factory=list)

    @property
    def quality_evaluated(self) -> bool:
        """True if at least one evaluator really ran.

        The caller uses this to decide whether a score carries a quality
        signal at all, rather than inferring quality from a number that only
        ever measured completion.
        """
        return bool(self.applied)

    @property
    def applied_results(self) -> list[tuple[str, EvalResult]]:
        """Each applied assertion type paired with the result it produced.

        The two lists are appended together and are therefore parallel, but a
        caller zipping them relies on an invariant it cannot see. Exposing the
        pairing keeps that knowledge here, where the invariant is maintained.
        """
        if len(self.applied) != len(self.results):
            raise RuntimeError(
                "outcome is inconsistent: "
                f"{len(self.applied)} applied vs {len(self.results)} results"
            )
        return list(zip(self.applied, self.results, strict=True))


class EvaluationPipeline:
    """Runs a test's assertions under a policy, collecting results and skips."""

    def __init__(
        self,
        resolver: EvaluatorResolver,
        policy: EvaluationPolicy,
        *,
        guardrail: GuardrailFn | None = None,
        artifacts: ArtifactContext = no_artifacts,
    ) -> None:
        self._resolver = resolver
        self._policy = policy
        self._guardrail = guardrail
        self._artifacts = artifacts

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
    ) -> EvaluationOutcome:
        """Evaluate every assertion on the test that the policy permits."""
        outcome = EvaluationOutcome()
        assertions = task.assertions or []
        if not assertions:
            return outcome

        if self._guardrail is not None:
            skip_reason = self._guardrail(task, response)
            if skip_reason:
                outcome.skipped.extend(
                    SkippedEvaluation(a.type, SkipReason.GUARDRAIL, skip_reason)
                    for a in assertions
                )
                return outcome

        with self._artifacts(response) as prepared:
            for assertion in assertions:
                await self._evaluate_one(task, prepared, trace, assertion, outcome)
        return outcome

    async def _evaluate_one(
        self,
        task: TestDefinition,
        prepared: PreparedResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
        outcome: EvaluationOutcome,
    ) -> None:
        """Evaluate a single assertion, recording either a result or a skip."""
        # Policy first: resolving would construct the very evaluator the
        # policy is meant to keep out of this process.
        if not self._policy.permits(assertion.type):
            outcome.skipped.append(
                SkippedEvaluation(
                    assertion.type,
                    SkipReason.NOT_ALLOWED,
                    f"policy '{self._policy.name}' does not permit this evaluator",
                )
            )
            return

        try:
            evaluator = self._resolver.create_for_assertion(assertion.type)
        except Exception as exc:
            outcome.skipped.append(
                SkippedEvaluation(assertion.type, SkipReason.UNSUPPORTED, str(exc))
            )
            return

        # A delegating evaluator builds its children through the same resolver
        # this pipeline uses, so the policy reaches all the way down.
        bind_resolver(evaluator, self._resolver)
        # Likewise for the filesystem: the root comes from the composition
        # that prepared this response, never from the assertion's own config.
        bind_workspace_root(evaluator, prepared.root, trusted=self._policy.trusted)

        try:
            result = await evaluator.evaluate(task, prepared.response, trace, assertion)
        except AssertionUnevaluated as exc:
            # Ran, declined to guess. Recorded as coverage rather than as a
            # failed measurement — the distinction the whole contract rests on.
            outcome.skipped.append(
                SkippedEvaluation(assertion.type, SkipReason.INDETERMINATE, str(exc))
            )
            return
        except Exception as exc:
            logger.warning(
                "evaluator for '%s' failed: %s", assertion.type, exc, exc_info=True
            )
            outcome.skipped.append(
                SkippedEvaluation(assertion.type, SkipReason.EVALUATOR_ERROR, str(exc))
            )
            return

        # Propagate the hard gate so scoring can fail the test outright.
        result.critical = assertion.critical
        outcome.results.append(result)
        outcome.applied.append(assertion.type)
