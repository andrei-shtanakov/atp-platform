# work

kind: parallel branch binding
agent: executor

source:
```prose
work = session: executor
  prompt: "Execute the plan"
  context: plan
```

---

The Receipt Ledger is OpenProse's mandatory audit trail for AI orchestration runs: after every completed statement instance, the VM appends a single machine-readable receipt to an append-only `receipts.jsonl` file, so the ledger grows only by addition and is never edited or reordered. Each receipt is hash-chained — it carries a `content_hash` (a sha256 fingerprint over its own canonical bytes) plus a `prev` pointer to the preceding receipt's hash, and the run's `ledger_head` always tracks the tip of that chain. Because every entry commits to the one before it, any tampering, deletion, or reordering breaks the chain and is immediately detectable, giving each run a verifiable, tamper-evident record of what executed, in what order, and over which inputs and outputs. Receipts also fingerprint each session's inputs and outputs and record its disposition, which turns the ledger into the primitive for deterministic replay: a later run can substitute recorded outcomes instead of re-invoking agents. For contributors and adopters, the primary value is trustworthy provenance and reproducibility — the ability to prove exactly how a multi-agent run unfolded and to replay it faithfully, without relying on the human-readable narration alone.
