"""Map an agent-eval-case into an ATP ``TestDefinition``.

The methodology format is adapter-neutral: this loader only translates structure.
The two assertion types it emits are the contract consumed by the
methodology-aware evaluator (added in a later slice):

- ``method_critical_check`` — the binary trap, marked ``critical=True`` so a
  failure hard-gates the test (score 0) via the core ScoreAggregator.
- ``method_rubric`` — the weighted graded criteria (only when a rubric exists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition

from atp_method.schema import AgentEvalCase

METHOD_CRITICAL_CHECK = "method_critical_check"
METHOD_RUBRIC = "method_rubric"


def _allowed_tools(case: AgentEvalCase) -> list[str] | None:
    """Map the case tool surface to ATP ``allowed_tools``.

    ``["none"]`` means text-in/text-out only → an empty allow-list (no tools).
    """
    if case.environment.tools == ["none"]:
        return []
    return list(case.environment.tools)


def _tags(case: AgentEvalCase) -> list[str]:
    """Case tags plus namespaced governance/sweep tags for filtering."""
    derived = [
        f"family_{case.family.replace('-', '_')}",
        f"capability_{case.capability}",
        f"axis_{case.construction_axis}",
        f"level_{case.axis_level}",
        f"suite_{case.suite_type}",
    ]
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for tag in [*case.tags, *derived]:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _assertions(case: AgentEvalCase) -> list[Assertion]:
    """Build the critical-check (hard gate) and optional rubric assertions."""
    assertions = [
        Assertion(
            type=METHOD_CRITICAL_CHECK,
            critical=True,
            config={
                "check": case.grader.critical_check,
                "expected_failure_mode": case.expected_failure_mode,
                "grader_type": case.grader.type,
                "gold": case.grader.gold,
            },
        )
    ]
    if case.grader.rubric:
        assertions.append(
            Assertion(
                type=METHOD_RUBRIC,
                config={
                    "rubric": [item.model_dump() for item in case.grader.rubric],
                    "grader_type": case.grader.type,
                    "scoring": case.grader.scoring,
                    "gold": case.grader.gold,
                },
            )
        )
    return assertions


def case_to_test_definition(case: AgentEvalCase) -> TestDefinition:
    """Translate a validated agent-eval-case into an ATP ``TestDefinition``."""
    input_data: dict[str, Any] = {
        "artifacts": [a.model_dump(exclude_none=True) for a in case.artifacts],
        "constraints": case.constraints,
    }
    if case.distractor is not None:
        input_data["distractor"] = case.distractor
    if case.turns:
        input_data["turns"] = [t.model_dump() for t in case.turns]

    return TestDefinition(
        id=case.id,
        name=f"{case.family} ({case.axis_level})",
        description=case.expected_failure_mode,
        tags=_tags(case),
        task=TaskDefinition(
            description=case.instruction,
            input_data=input_data,
        ),
        constraints=Constraints(allowed_tools=_allowed_tools(case)),
        assertions=_assertions(case),
    )


def load_case(path: str | Path) -> TestDefinition:
    """Load and validate a single agent-eval-case YAML into a ``TestDefinition``."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    case = AgentEvalCase.model_validate(doc)
    return case_to_test_definition(case)
