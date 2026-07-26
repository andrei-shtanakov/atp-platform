"""How a server composes evaluation, and what it is allowed to hand over.

The dashboard may not import evaluator implementations — they live in
atp-platform, it declares only atp-core, and an architecture test enforces it.
So the implementations arrive by constructor injection from a composition root
that can see both.

Two rules shape what crosses that boundary.

**The resolver arrives already restricted.** `FilteredResolver` is the only
thing the server side receives, and it cannot produce an evaluator the policy
forbids — not because callers remember to ask politely, but because the
unrestricted registry never reaches them. Handing over the full registry with
a policy alongside would make the restriction a convention, and conventions
are what get refactored away.

**The mode is declared, never inferred.** A server states which of the two it
is running; `completion_only` with no resolver is a legitimate deployment,
`deterministic_allowlist` with no resolver is a broken one. Inferring the mode
from whether a resolver happened to be passed would make the difference
between "scoring is off by choice" and "someone launched the wrong entrypoint"
invisible in exactly the place it matters.
"""

from __future__ import annotations

from typing import Literal

from atp.evaluation.pipeline import EvaluationPolicy, EvaluatorLike, EvaluatorResolver

#: What a deployment declares about its evaluation capability.
#:
#: ``completion_only`` — no evaluators; the score measures completion and says
#: so. ``deterministic_allowlist`` — inspection-only evaluators run under the
#: server policy.
EvaluationMode = Literal["completion_only", "deterministic_allowlist"]

COMPLETION_ONLY: EvaluationMode = "completion_only"
DETERMINISTIC_ALLOWLIST: EvaluationMode = "deterministic_allowlist"


class EvaluatorNotPermitted(LookupError):
    """Raised when a caller asks for an evaluator the policy withholds."""


class FilteredResolver:
    """A resolver that can only produce what its policy permits.

    Wrapping rather than checking at the call site is deliberate: the registry
    is not part of this object's public surface, so ordinary downstream code
    cannot widen what it yields.

    This is an engineering boundary, not a sandbox. Python has no private
    attributes, so `_inner` remains reachable by anyone who goes looking. It
    stops the accident and the shortcut, which is what package boundaries are
    for; it does not stop code that has already decided to break out.
    """

    def __init__(self, inner: EvaluatorResolver, policy: EvaluationPolicy) -> None:
        self._inner = inner
        self._policy = policy

    @property
    def policy(self) -> EvaluationPolicy:
        """The policy this resolver enforces."""
        return self._policy

    def permitted_assertion_types(self) -> frozenset[str] | None:
        """What this resolver will produce; None means unrestricted."""
        return self._policy.allowed_assertion_types

    def create_for_assertion(self, assertion_type: str) -> EvaluatorLike:
        """Build an evaluator, or refuse if the policy withholds it."""
        if not self._policy.permits(assertion_type):
            raise EvaluatorNotPermitted(
                f"policy '{self._policy.name}' does not permit '{assertion_type}'"
            )
        return self._inner.create_for_assertion(assertion_type)


class IncompleteComposition(RuntimeError):
    """The declared mode and the injected parts do not agree.

    Deliberately fatal at startup. A server that meant to score quality and
    silently fell back to counting completions would publish numbers that look
    like the real thing, and nobody would notice until the numbers mattered.
    """


def validate_composition(
    mode: EvaluationMode, resolver: FilteredResolver | None
) -> None:
    """Check that the declared mode matches what was actually injected.

    Both mismatches are errors, including the harmless-looking one: passing a
    resolver in ``completion_only`` means someone believed evaluation was on.
    Ignoring it quietly would leave that belief uncorrected.
    """
    if mode == DETERMINISTIC_ALLOWLIST and resolver is None:
        raise IncompleteComposition(
            "evaluation_mode='deterministic_allowlist' requires an evaluator "
            "resolver; launch through the canonical entrypoint (`atp dashboard`) "
            "or declare evaluation_mode='completion_only' explicitly"
        )
    if mode == COMPLETION_ONLY and resolver is not None:
        raise IncompleteComposition(
            "evaluation_mode='completion_only' was declared but an evaluator "
            "resolver was supplied; the resolver would be ignored, so the "
            "declaration and the wiring disagree"
        )
