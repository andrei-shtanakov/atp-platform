# Run State — 20260716-082734-0vx3bm

**Program:** examples/16-parallel-reviews.prose
**State backend:** filesystem
**Status:** completed

## Execution

```prose
agent reviewer:                                    # s001 (definition)
  model: sonnet
  prompt: "You are an expert code reviewer"

parallel:                                          # s002 ✓ joined (all, 3/3)
  security = session: reviewer                     # s003 ✓ -> bindings/security.md
    prompt: "Review for security vulnerabilities"
  perf = session: reviewer                         # s004 ✓ -> bindings/perf.md
    prompt: "Review for performance issues"
  style = session: reviewer                        # s005 ✓ -> bindings/style.md
    prompt: "Review for code style and readability"

session "Create unified code review report"        # s006 ✓ -> bindings/anon_006.md
  context: { security, perf, style }
```

- All three branches spawned concurrently as real subagents (sonnet); the
  join strategy is the default `"all"` (3/3 completed, 0 failed).
- The program gives the reviewers no explicit target; the VM supplied the
  execution context (this repository) and the reviewers examined the
  `tools/` Python package.
- `s006` consumed the three branch bindings by reference —
  `input_fingerprints` in its receipt records exactly which versions were
  wired in.

## Notes

- Token usage basis: **estimated** — the substrate reports per-subagent
  totals without an input/output split; output tokens estimated from
  binding size (bytes/4), input = total − output.
- Receipt ledger: `receipts.jsonl` (7 receipts: run_start, 3 branches,
  parallel join, synthesis, run_end), anchored in `run.json`. Contract:
  `contracts/receipt.md`.
