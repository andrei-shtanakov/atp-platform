# atp-method

ATP plugin that runs [`method/`](../../method/) **agent-eval-case** methodology
cases through the platform: a schema model + a loader that maps each case to an
ATP `TestDefinition`, plus (in later slices) a methodology-aware evaluator and
the format-dispatch registration so `atp test method/cases/*.yaml` just works.

See the design in [`spec/atp-method-plugin.md`](../../spec/atp-method-plugin.md).

## Status

- [x] schema model + loader (case → `TestDefinition`)
- [ ] `AgentEvalCaseEvaluator` (`critical_check` then rubric)
- [ ] `register()` + format dispatch + E2E
