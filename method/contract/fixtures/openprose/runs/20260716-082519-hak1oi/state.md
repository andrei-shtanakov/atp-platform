# Run State — 20260716-082519-hak1oi

**Program:** examples/01-hello-world.prose
**State backend:** filesystem
**Status:** completed

## Execution

```prose
session "Say hello and briefly introduce yourself"   # ✓ completed (s001)
```

- `s001` — anonymous session, spawned as a real subagent (haiku). Output
  written by the subagent to `bindings/anon_001.md`; the VM received a
  confirmation pointer, not the content.

## Notes

- Token usage basis: **estimated**. The substrate reported a real total
  (26829 tokens) without an input/output split; output tokens estimated
  from binding size (457 bytes / 4 ≈ 114), input = total − output.
- Receipt ledger: `receipts.jsonl` (3 receipts — run_start, s001, run_end),
  anchored by `ledger_head` in `run.json`. Contract:
  `contracts/receipt.md` (`openprose.receipt.v1`).
