# atp-method

ATP plugin that runs [`method/`](../../method/) **agent-eval-case** methodology
cases through the platform: a schema model + a loader that maps each case to an
ATP `TestDefinition`, a methodology-aware evaluator, and corpus preparation for
file-grounded `read_only_corpus` cases.

See the design in [`spec/atp-method-plugin.md`](../../spec/atp-method-plugin.md).

## Status

- [x] schema model + loader (case → `TestDefinition`)
- [x] `AgentEvalCaseEvaluator` (`critical_check` then rubric)
- [x] `register()` + source dispatch + E2E (`atp test method/cases/<case>.yaml`)
- [x] `read_only_corpus` text/markdown corpus preparation + `file_read`
- [x] `citation_grounding` deterministic checker

## Usage

Installed as a plugin, the platform runs methodology cases directly — a single
case or a whole sweep (directory):

```bash
atp test method/cases/req-extraction --adapter=http \
  --adapter-config endpoint=http://agent:8000/execute,allow_internal=true --runs=10
```
