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


#: Returns a reason to skip evaluation entirely, or None to proceed.
GuardrailFn = Callable[[TestDefinition, ATPResponse], str | None]

#: Makes response artifacts reachable on disk for the duration of the block,
#: and yields the response evaluators should see. A sandboxing implementation
#: returns a rewritten response whose paths are workspace-relative, so an
#: agent-chosen path can never reach an evaluator.
ArtifactContext = Callable[
    [ATPResponse], contextlib.AbstractContextManager[ATPResponse]
]


@contextlib.contextmanager
def no_artifacts(response: ATPResponse) -> Iterator[ATPResponse]:
    """Default artifact context: touch nothing on disk, change nothing."""
    yield response


@dataclass(frozen=True)
class EvaluationPolicy:
    """What an execution context permits.

    `allowed_assertion_types=None` means "no restriction" and is only
    appropriate where the input is trusted.
    """

    name: str
    allowed_assertion_types: frozenset[str] | None = None

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
        response: ATPResponse,
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

        try:
            result = await evaluator.evaluate(task, response, trace, assertion)
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
