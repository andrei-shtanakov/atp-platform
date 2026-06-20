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
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
    TestSuite,
)

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
        f"version_{case.version}",
    ]
    if case.task_type:
        derived.append(f"task_type_{case.task_type}")
    if case.language:
        derived.append(f"language_{case.language}")
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
    checker_config = dict(case.grader.config or {})
    assertions = [
        Assertion(
            type=METHOD_CRITICAL_CHECK,
            critical=True,
            config={
                "check": case.grader.critical_check,
                "expected_failure_mode": case.expected_failure_mode,
                "grader_type": case.grader.type,
                "checker": case.grader.checker,
                "gold": case.grader.gold,
                "expected_findings": [
                    f.model_dump() for f in (case.grader.expected_findings or [])
                ],
                "must_not_flag": [
                    m.model_dump() for m in (case.grader.must_not_flag or [])
                ],
                "schema": (
                    case.output_contract.json_schema if case.output_contract else None
                ),
                "assertions": checker_config.get("assertions", []),
                **checker_config,
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
                    "checker": case.grader.checker,
                    "scoring": case.grader.scoring,
                    "gold": case.grader.gold,
                },
            )
        )
    return assertions


def case_to_test_definition(
    case: AgentEvalCase, case_path: str | Path | None = None
) -> TestDefinition:
    """Translate a validated agent-eval-case into an ATP ``TestDefinition``."""
    input_data: dict[str, Any] = {
        "artifacts": [a.model_dump(exclude_none=True) for a in case.artifacts],
        "constraints": case.constraints,
        "run_mode": case.run_mode,
    }
    if case_path is not None:
        input_data["case_path"] = str(case_path)
    if case.artifact_corpus is not None:
        input_data["artifact_corpus"] = case.artifact_corpus.model_dump(
            exclude_none=True
        )
        input_data["request_preparer"] = "corpus"
    if case.distractor is not None:
        input_data["distractor"] = case.distractor
    if case.turns:
        input_data["turns"] = [t.model_dump() for t in case.turns]
    if case.output_contract is not None:
        input_data["output_contract"] = case.output_contract.model_dump(
            by_alias=True, exclude_none=True
        )

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
    return case_to_test_definition(case, Path(path))


def _case_files(path: Path) -> list[Path]:
    """Resolve a path to the case file(s): a directory expands to its *.yaml."""
    return sorted(path.glob("*.yaml")) if path.is_dir() else [path]


def is_agent_eval_case(path: str | Path) -> bool:
    """Detect an agent-eval-case source (a single file or a directory of cases).

    Recognized by the distinctive ``construction_axis`` field plus a ``grader``
    carrying a ``critical_check`` — present in every case, absent from native ATP
    suites and game suites.
    """
    files = _case_files(Path(path))
    if not files:
        return False
    try:
        doc = yaml.safe_load(files[0].read_text())
    except Exception:
        return False
    grader = doc.get("grader") if isinstance(doc, dict) else None
    return (
        isinstance(doc, dict)
        and "construction_axis" in doc
        and isinstance(grader, dict)
        and "critical_check" in grader
    )


def load_suite(path: str | Path, name: str | None = None) -> TestSuite:
    """Load one case file or a directory of cases (a sweep) into a TestSuite."""
    p = Path(path)
    tests = [load_case(f) for f in _case_files(p)]
    suite_name = name or (p.name if p.is_dir() else p.stem)
    return TestSuite(test_suite=suite_name, tests=tests)
