# synthesis

kind: let (reassigned in if-branch)
agent: captain

source:
```prose
synthesis = session: captain
  prompt: "Integrate the work while addressing critic's concerns"
  context: { work, review }
```

---

The Receipt Ledger is OpenProse's mandatory audit trail for AI orchestration runs: after every completed statement instance, the VM appends a single machine-readable receipt to an append-only `receipts.jsonl` file, so the ledger grows only by addition and is never edited or reordered. Each receipt is hash-chained — it carries a `content_hash` (a sha256 fingerprint over its own canonical bytes) plus a `prev` pointer to the preceding receipt's hash, and the run's `ledger_head` always tracks the tip of that chain. Because every entry commits to the one before it, any tampering, deletion, or reordering breaks the chain, so that a verifier holding a trusted `ledger_head` anchor can detect the break by recomputing hashes; without such an anchor the chain proves internal consistency and ordering rather than tamper-proofing, since an attacker with write access could in principle rewrite the entire chain and its head. Receipts also fingerprint each session's inputs and outputs and record its disposition, which turns the ledger into the primitive for deterministic replay: a later run can substitute recorded outcomes instead of re-invoking agents. For contributors and adopters, the primary value is trustworthy provenance and reproducibility — the ability to demonstrate how a multi-agent run unfolded and to replay it faithfully, without relying on the human-readable narration alone.

**Notes on critic concerns:**

- Addressed — the executor's "immediately detectable" tamper claim: reframed as chain consistency verifiable against a *trusted `ledger_head` anchor*, not standalone tamper-proofing, since an attacker with write access could rewrite the whole chain plus its head. Detection is contingent on an out-of-band trust root.
- Addressed — the "underspecified inputs / possibly-fabricated mechanics" concern: softened absolute verifiability language ("prove exactly" → "demonstrate") so the description does not overclaim guarantees that the recorded hash-chain alone cannot deliver.
- Set aside as plan-level only — the process critiques (over-engineered fan-out, serial-dependency-mislabeled-as-parallel, missing integration step) concern how the workflow was orchestrated, not the correctness of the deliverable paragraph; they are noted for the plan author but do not change the integrated description.
