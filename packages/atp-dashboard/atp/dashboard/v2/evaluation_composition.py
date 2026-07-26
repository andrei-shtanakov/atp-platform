"""Where the dashboard receives its evaluation capability, and refuses to guess.

The dashboard cannot build evaluators: they live in atp-platform, it declares
only atp-core, and `tests/architecture/test_package_boundaries.py` enforces
that. What it can do is state precisely what it was given and fail loudly when
that does not match what it was told to be.

There is deliberately no module-level setter. A `set_evaluator_registry()`
would make behaviour depend on import order, leak between tests, and become
ambiguous the moment a process holds two apps. The capability is constructor
state, stored on `app.state`, and read from there by the routes that need it.
"""

from __future__ import annotations

from dataclasses import dataclass

from atp.evaluation import (
    COMPLETION_ONLY,
    EvaluationMode,
    FilteredResolver,
    validate_composition,
)


@dataclass(frozen=True)
class EvaluationCapability:
    """What this application instance can actually evaluate.

    Frozen: a running app's evaluation capability is decided at composition
    time. Anything that could change it later would reintroduce the ordering
    problem the constructor injection exists to remove.
    """

    mode: EvaluationMode
    resolver: FilteredResolver | None

    @classmethod
    def build(
        cls, mode: EvaluationMode, resolver: FilteredResolver | None
    ) -> EvaluationCapability:
        """Validate the pairing, then freeze it.

        Raises `IncompleteComposition` when the declaration and the wiring
        disagree — at startup, where a misconfigured deployment is cheap to
        notice, rather than at scoring time, where it is not.
        """
        validate_composition(mode, resolver)
        return cls(mode=mode, resolver=resolver)

    @property
    def evaluates_quality(self) -> bool:
        """True when evaluators are wired and may produce a quality signal."""
        return self.resolver is not None

    def describe(self) -> dict[str, object]:
        """The capability as an operator needs to see it.

        Lists the permitted assertion types rather than a count: "25 allowed"
        answers nothing when the question is whether a particular suite will
        actually be scored.
        """
        permitted = (
            sorted(self.resolver.permitted_assertion_types() or [])
            if self.resolver is not None
            else []
        )
        return {
            "evaluation_mode": self.mode,
            "resolver_connected": self.resolver is not None,
            "policy": (
                self.resolver.policy.name if self.resolver is not None else None
            ),
            "allowed_assertion_types": permitted,
        }


#: A server that deliberately runs without evaluators.
COMPLETION_ONLY_CAPABILITY = EvaluationCapability(mode=COMPLETION_ONLY, resolver=None)
