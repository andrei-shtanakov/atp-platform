# Spec: `atp-method` plugin — run agent-eval-case methodology cases via ATP

**Status:** in progress (slice 1)
**Created:** 2026-06-09
**Decision:** the `method/` methodology is a self-contained domain that evolves on
its own cadence → ship it as a **plugin**, not by merging its concepts into the
core schema (which would couple every methodology change to a core release).

## Problem

`method/` (PR #140) is a docs-only package: a JSON Schema (`agent-eval-case`),
methodology, glossary, an LLM case-generator instruction, and 3 example cases. No
code reads it; the ATP runner only loads its native `TestSuite`
(`loader/models.py`). To actually run these cases through the platform (and get
adapters, sandbox, statistics, dashboard, reporters) a thin integration is needed.

## Approach (chosen over alternatives)

Considered: runtime loader, AOT compiler, custom evaluator only, external SDK
harness, and merging into the native schema. Chosen: **plugin + one small core
primitive**, because the methodology evolves independently (plugin decouples it
from core) while one concept — a hard-gate check — is universally useful and
belongs in core.

- **Core (small, stable, universal):**
  1. a **hard-gate / critical assertion** primitive — `critical: true` on an
     assertion fails the whole test regardless of rubric score (the native home
     for `agent-eval-case.grader.critical_check`);
  2. a tiny **format-dispatch registry** so non-native suite formats register a
     detector + handler instead of growing the hardcoded `_is_game_suite` branch
     in `atp test`.
- **Plugin `packages/atp-method/` (evolves on its own):** the `agent-eval-case`
  schema model, the case→`TestDefinition` loader, and a methodology-aware
  evaluator that implements `critical_check → then rubric`.

## Integration seams (verified in code)

- Plugins register via entry-point group `atp.plugins` → a `register()` function
  (e.g. `atp-games`: `game = atp_games.plugin:register`).
- Evaluators register through the core registry via
  `_register_assertion_mapping(assertion_type, evaluator_key)`.
- There is **no** loader plugin group (`ALL_GROUPS` = adapters/evaluators/
  reporters only); today `atp test` dispatches a non-native format with a
  hardcoded `if _is_game_suite(file)` branch (`cli/main.py:400`). We replace that
  pattern with a small registry.

## Mapping rules (loader)

| agent-eval-case | → ATP |
|---|---|
| `instruction`, `artifacts` | `task.description` / `task.input_data` |
| `environment.tools` | `constraints.allowed_tools` |
| `grader.critical_check` | a critical assertion (`critical: true`) |
| `grader.rubric` (type=rubric) | `llm_eval` / `composite` |
| `grader` (type=programmatic) | `code_exec` / `artifact` |
| `family` / `axis_level` / `capability` / `suite_type` | `tags` (sweep analysis + governance) |

## Build sequence (layered, TDD; each slice its own PR)

1. **Core hard-gate** — `Assertion.critical` (loader), `EvalResult.critical`
   (core/results), `ScoreAggregator.score_test_result` hard-fail when a critical
   result fails, CLI eval loop propagates `assertion.critical`. *(this PR)*
2. **Core format-dispatch registry** — replace the hardcoded `_is_game_suite`
   branch with a registry `{detector → handler}`; migrate the game branch onto it.
3. **Plugin: schema + loader** — `agent-eval-case` pydantic model + case→
   `TestDefinition` mapping; tested on the 3 `req-extraction` example cases.
4. **Plugin: evaluator** — `AgentEvalCaseEvaluator` (`critical_check` then rubric).
5. **Plugin: register() + dispatch** — wire entry point so
   `atp test method/cases/...yaml` runs; E2E sweep against the demo agent.

## Package layout

```
packages/atp-method/
  pyproject.toml          # entry-point atp.plugins: method = atp_method.plugin:register
  atp_method/
    plugin.py             # register(): evaluator + format detector/handler
    schema.py             # pydantic model mirroring agent-eval-case.schema.json
    loader.py             # agent-eval-case.yaml → TestDefinition
    evaluators/case_evaluator.py
  tests/
```

## Out of scope / authoring-linter (not in this plugin)

The README notes two checks that are not JSON-Schema-expressible and stay in an
authoring linter/review: rubric weights sum to 1.0, and `critical_check`
semantically matches `expected_failure_mode`.

## Verification

Per slice: `ruff` + `pyrefly` + `pytest`. Final E2E: run the `req-extraction`
sweep through `atp test` against the compose demo agent and observe the
"curve of collapse" via `--runs`.
