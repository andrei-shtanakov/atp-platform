# Run State — 20260716-094019-fsylki

**Program:** examples/16-parallel-reviews.prose
**State backend:** filesystem
**Mode:** `--resume 20260716-082734-0vx3bm` (skip semantics, Phase 4)
**Status:** completed

## Execution

Preconditions verified before any reuse:

- Program byte-identical to the source run's `program.prose`
  (`sha256:cdce5073…` on both sides).
- Fresh compile IR (`examples/dist/16-parallel-reviews.ir.json`,
  `ir-check: OK`) — statement IDs shared with the source run.

```prose
parallel:                                          # s002 join (3/3 skipped)
  security = session: reviewer                     # s003 SKIPPED <- source s003
  perf = session: reviewer                         # s004 SKIPPED <- source s004
  style = session: reviewer                        # s005 SKIPPED <- source s005

session "Create unified code review report"        # s006 SKIPPED <- source s006
  context: { security, perf, style }
```

Every session's memo identity — (program hash, statement_id, material
input fingerprints) — was unchanged, so no subagent was spawned. Bindings
were **copied** from the source run (copy-with-provenance; each copy's
sha256 verified against the reused `output_fingerprint`), and each
receipt carries `reused_from` coordinates.

## Measured savings (evaluation addendum material)

| Run | Tokens (in+out) | Sessions spawned |
| --- | ---: | ---: |
| Source `…-0vx3bm` (rendered) | 218,285 | 4 |
| This run (all skipped) | **0** | **0** |

`openprose-tools cost <this-run> --compare <source-run>`:
`delta: -218285 tokens (saved 100%)`.

## Notes

- Skipped receipts carry zero usage with `basis: "exact"` — zero spend is
  exactly known.
- Ledger: 7 receipts (run_start with `resume` marker, 3 skipped branches,
  join, skipped synthesis, run_end); `run.json` carries
  `reuse_source_run`.
