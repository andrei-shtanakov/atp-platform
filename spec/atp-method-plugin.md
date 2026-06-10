# Spec: `atp-method` plugin — run agent-eval-case methodology cases via ATP

**Status:** done (2026-06-10) — all 5 slices merged (PRs #142–#146).
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

## Build sequence (layered, TDD; each slice its own PR) — DONE

1. ✅ **Core hard-gate** (#142) — `Assertion.critical`, `EvalResult.critical`,
   `ScoreAggregator` hard-fail when a critical result fails, CLI propagates
   `assertion.critical`.
2. ✅ **Core format-dispatch registry** (#143) — replaced the hardcoded
   `_is_game_suite` branch with `{detector → handler}`; migrated the game branch.
3. ✅ **Plugin: schema + loader** (#144) — `agent-eval-case` pydantic model + case→
   `TestDefinition` mapping; tested on the 3 `req-extraction` example cases.
4. ✅ **Plugin: evaluator** (#145) — `AgentEvalCaseEvaluator` (`critical_check` then
   weighted rubric), delegating model calls to the platform LLM judge.
5. ✅ **Plugin: register() + dispatch + E2E** (#146).

### Refinements discovered during implementation

- **Two registries, not one.** Slice 2's `format_dispatch` is *run-level* (a format
  with its own execution path, e.g. game suites). agent-eval-case is a *source*
  format — it parses into a normal `TestSuite` and reuses the native adapter /
  orchestrator / evaluator / reporter path — so slice 5 added a complementary
  *loader-level* seam, `atp.loader.suite_source.SuiteSourceRegistry`. The plugin
  registers there; `_run_suite` (and thus `--adapter`) is reused unchanged.
- **Entry-point loader.** The `atp.plugins` group was declared (atp-games) but
  never invoked. Slice 5 added `atp.plugins.entrypoints.load_entrypoint_plugins()`,
  called once at CLI startup, which runs every plugin's `register()` hook.
- **Public registry API.** Added `EvaluatorRegistry.register_assertion_mapping()`
  so plugins don't reach into the private `_register_assertion_mapping`.

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
