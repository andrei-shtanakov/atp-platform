"""Plugin registration hook (``atp.plugins`` entry point).

``register()`` wires atp-method into the core registries so that
``atp test method/cases/<case>.yaml`` works:

- the methodology evaluator handles the ``method_critical_check`` and
  ``method_rubric`` assertion types the loader emits;
- the agent-eval-case source format parses case files (or a directory sweep)
  into a ``TestSuite`` that then runs through the normal adapter / orchestrator /
  evaluator / reporter path.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atp.loader.models import TestDefinition
    from atp.protocol import ATPRequest
    from atp.runner.preparation import PreparedRequest


class _LazyCorpusRunPreparer:
    """Load corpus runtime dependencies only when corpus preparation is used."""

    async def prepare(
        self, test: TestDefinition, request: ATPRequest
    ) -> PreparedRequest:
        """Prepare a corpus request with the runtime preparer."""
        try:
            runtime = importlib.import_module("atp_method.runtime")
        except ModuleNotFoundError as exc:
            if exc.name not in {"fastapi", "uvicorn"}:
                raise
            raise RuntimeError(
                "corpus-backed method cases require mock tool server dependencies "
                "fastapi and uvicorn"
            ) from exc

        return await runtime.CorpusRunPreparer().prepare(test, request)


def register() -> None:
    """Register the evaluator and the agent-eval-case source format."""
    from atp.evaluators.registry import get_registry
    from atp.loader import get_suite_source_registry
    from atp.runner.preparation import register_request_preparer

    from atp_method.evaluators import AgentEvalCaseEvaluator
    from atp_method.loader import (
        METHOD_CRITICAL_CHECK,
        METHOD_RUBRIC,
        is_agent_eval_case,
        load_suite,
    )

    registry = get_registry()
    registry.register("agent_eval_case", AgentEvalCaseEvaluator)
    registry.register_assertion_mapping(METHOD_CRITICAL_CHECK, "agent_eval_case")
    registry.register_assertion_mapping(METHOD_RUBRIC, "agent_eval_case")

    get_suite_source_registry().register(
        "agent_eval_case", is_agent_eval_case, load_suite
    )
    register_request_preparer("corpus", _LazyCorpusRunPreparer())
