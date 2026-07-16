# plan

kind: let
agent: captain

source:
```prose
let plan = session: captain
  prompt: "Break down this task into work items: {task}"
```

---

## Work Items

**WI-1: Draft the core description paragraph**
Write one paragraph (4-6 sentences) describing the receipt ledger feature.
Acceptance: names the feature, states it is append-only and hash-chained, explains
it produces a tamper-evident audit trail for AI orchestration runs, and identifies
the primary user value (verifiable provenance/replayability). Single paragraph, no
lists.

**WI-2: Verify domain accuracy**
Cross-check terminology against OpenProse concepts (runs, bindings, sessions, state
backends) so the description fits the project's model.
Acceptance: no invented mechanics; wording consistent with prose.md/compiler.md
vocabulary; hash-chain claim is technically coherent (each entry references prior
entry's hash).

**WI-3: Polish for tone and length**
Edit for clarity, neutral technical register, and audience (contributors/adopters).
Acceptance: reads as a project description (not a spec), free of jargon overload,
grammatically clean, fits as an intro blurb.

Dispatch: WI-1 to a writer specialist, WI-2 to a domain reviewer, WI-3 to an editor.
Captain validates each result before accepting.
