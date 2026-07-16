# OpenProse Receipt Contract — `openprose.receipt.v1`

Machine-readable audit contract for OpenProse runs. Every statement the VM
completes appends one **receipt** — a JSON object on its own line — to the
run's ledger. The ledger is the deterministic, replayable record of what
happened; the human-readable `state.md` narrative is written *alongside* it,
never instead of it.

This contract is adapted from the upstream `openprose/prose` Reactor receipt
envelope (see `contracts/vendored/prose-receipt-schema/`), reshaped from
node/responsibility scope to OpenProse v1 statement scope.

## Files

| File | Purpose |
| ---- | ------- |
| `.prose/runs/{run_id}/receipts.jsonl` | Append-only ledger, one receipt per line |
| `.prose/runs/{run_id}/run.json` | Run manifest; anchors `ledger_head` |

Both are written by the VM via the `emit_receipt` host primitive
(`primitives/session.md`). Subagents never write them.

## Receipt schema

A concrete `rendered` example:

```json
{
  "v": "openprose.receipt.v1",
  "run_id": "20260716-093000-a1b2c3",
  "statement_id": "s003",
  "kind": "session",
  "agent": "researcher",
  "input_fingerprints": {"topic": "sha256:9f2c…"},
  "output_fingerprint": "sha256:4b7e…",
  "status": "rendered",
  "surprise_cause": null,
  "usage": {
    "basis": "estimated",
    "input_tokens": 46210,
    "output_tokens": 1840,
    "model": "haiku"
  },
  "error": null,
  "detail": null,
  "reused_from": null,
  "prev": "sha256:c81d…",
  "hash_algorithm": "sha256",
  "content_hash": "sha256:07aa…"
}
```

### Fields

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `v` | string | Schema tag. Always `"openprose.receipt.v1"` for this version |
| `run_id` | string | The run this receipt belongs to (`{YYYYMMDD}-{HHMMSS}-{random6}`) |
| `statement_id` | string | Stable statement instance ID (see statement-id contract below) |
| `kind` | enum | `session` \| `parallel_branch` \| `block_call` \| `discretion` \| `control` |
| `agent` | string \| null | Agent name for `session`/`parallel_branch`; null otherwise |
| `input_fingerprints` | object | Map of wired context/input names → fingerprints of the values consumed. `{}` when the statement consumed nothing |
| `output_fingerprint` | string \| null | Fingerprint of the value the statement produced; null if it produced none |
| `status` | enum | `rendered` \| `skipped` \| `failed` |
| `surprise_cause` | enum \| null | Why a `rendered` statement spent: `input` (a material fingerprint moved), `self` (first-ever render), `external` (user-forced re-render). Null on `skipped`, `failed`, and non-spending kinds |
| `usage` | object | Token/cost attribution — see below |
| `error` | object \| null | On `failed`: `{"type": string, "message": string, "retry_count": number}`; null otherwise |
| `detail` | object \| null | Kind-specific payload. For `discretion`: `{"condition": string, "outcome": string, "branch": string \| null, "reason": string \| null}` — the replay primitive. `outcome` MUST be a stable literal a replayer can consume without parsing (`"true"`/`"false"` for boolean conditions, the chosen option name for `choice`); free-text justification goes in `reason`. For `control`: loop/join bookkeeping. Null for plain sessions |
| `reused_from` | object \| null | Skip provenance: `{"run_id", "statement_id", "output_fingerprint"}` of the receipt whose output this statement reused. Non-null exactly when `status: skipped` with a reused binding (see Skipped receipts below) |
| `prev` | string \| null | `content_hash` of the previous receipt in this ledger; null for the first line |
| `hash_algorithm` | string | Always `"sha256"` |
| `content_hash` | string | This receipt's chain identity — see Hashing |

### Enum values

- `kind`: `session` (a spawned subagent), `parallel_branch` (one branch of a
  `parallel` block; the joining `parallel` itself emits a `control` receipt),
  `block_call` (a `do name(...)` invocation completing, carrying the block's
  return fingerprint), `discretion` (an evaluated `**...**` condition),
  `control` (loop iteration bookkeeping, parallel join, try/catch transitions,
  program start/end).
- `status`: `rendered` (work happened), `skipped` (memoized/skip semantics —
  Phase 4; also used when a branch was never started), `failed`.
- `usage.basis`: `exact` | `estimated` | `unavailable`.

### `usage` — honest attribution

```json
{"basis": "exact", "input_tokens": 46210, "output_tokens": 1840, "model": "haiku"}
```

`basis` is mandatory:

- `exact` — the substrate reported real token counts for this statement.
- `estimated` — the VM approximated (state the method in `state.md`; e.g.
  chars/4 over the prompt and the returned confirmation). Token fields carry
  the estimate.
- `unavailable` — no data; token fields MUST be `0` and consumers MUST NOT
  treat them as measurements.

Every consumer (`inspect`, `cost`, future `budget:` enforcement) MUST surface
the basis — a cost rollup built on estimates says so.

All numeric fields in a receipt are **integers**. Floats are forbidden by the
canonical form (see below), which keeps hashing portable across languages.

### Skipped receipts (skip semantics — `prose.md`)

A `status: skipped` receipt records a memoized reuse, not absence of work:

- `input_fingerprints` — the current (matching) material fingerprints.
- `output_fingerprint` — copied forward from the reused receipt; the
  binding file copied into this run MUST hash to it.
- `reused_from` — the source coordinates (required).
- `usage` — zero tokens with `basis: "exact"` (zero spend is exactly
  known), `model: "none"`.
- `surprise_cause` — null.

The run manifest of a resuming run carries `reuse_source_run: "<run_id>"`
(optional field of `openprose.run.v1`; absent when the run reused
nothing).

## Fingerprints and facets

A **fingerprint** is `"sha256:" + hex(sha256(bytes))` over the exact bytes
of the value being referenced — for filesystem bindings, the binding
file's content at receipt time.

A **facet** is an independently fingerprintable named part of a binding,
addressed as `binding.facet` (used by `material:` — compiler.md). For v1
markdown bindings: facet `f` of binding `b` is the exact byte span of the
top-level section titled `f` in `bindings/b.md` — from the line `## f`
(exclusive) to the next top-level `##` heading or end of file
(exclusive), trailing newline included. A `material:` entry naming a
facet means only that span's fingerprint participates in the memo
identity. Facet fingerprints appear in `input_fingerprints` under the
dotted key (`"review.summary": "sha256:…"`).

## Statement IDs — `openprose.statement-id.v1`

Receipts need statement identities that are stable across runs of the same
program and across tooling generations. This contract is **frozen**: the
Phase 3 compile IR adopts it verbatim and must not redefine it.

### Static base ID

Number every statement in the program source `s001`, `s002`, … in **source
order** (top to bottom), counting:

- every statement at root scope,
- every statement inside `block` definitions (at its source position — block
  *definitions* are numbered where they appear, not where they are invoked),
- branch statements inside `if`/`elif`/`else`, `choice`, `try`/`catch`/
  `finally`, and loop bodies.

Comments and blank lines are not statements. Multi-line statements (a
`session` with properties) are one statement. A `parallel` block is one
statement; each of its branches is additionally one statement.

### Dynamic instance suffixes

At runtime a static statement may execute more than once. The receipt's
`statement_id` is the base ID plus dynamic suffixes, **outermost context
first**, in nesting order:

| Suffix | Meaning |
| ------ | ------- |
| `.x{execution_id}` | Executed inside block invocation frame `{execution_id}` |
| `.i{n}` | Iteration `n` (1-based) of the nearest enclosing loop |
| `.b{n}` | Parallel branch `n` (1-based, source order) |
| `.d{n}` | Discretion condition `n` (1-based) within the statement |

Examples:

- `s014` — plain root statement, first (only) execution.
- `s014.i2` — same statement, second iteration of the loop around it.
- `s009.x3` — statement 9 executing inside block frame `execution_id: 3`.
- `s009.x3.i2.b1` — inside frame 3, loop iteration 2, parallel branch 1.
- `s021.d1` — the first discretion condition evaluated within statement 21.

`execution_id` is the VM's monotonic block-frame counter (`prose.md`, Call
Stack Management) — unique within a run, so recursive invocations of the same
block get distinct IDs.

## Hashing and chain — **chain consistency**, not tamper-proofing

### Canonical form

The canonical serialization of a receipt (and of any JSON value inside one):

1. Objects: keys sorted lexicographically (byte order), rendered as
   `{"k1":v1,"k2":v2}` — no whitespace.
2. Arrays: element order preserved, rendered `[v1,v2]` — no whitespace.
3. Strings: JSON-escaped; non-ASCII characters are NOT escaped (UTF-8 bytes
   are hashed as-is).
4. Numbers: integers only. Floats, NaN, and Infinity are invalid in a receipt.
5. Booleans/null: `true`, `false`, `null`.

### Content hash

`content_hash = "sha256:" + hex(sha256(canonical(receipt sans content_hash)))`

The hash covers every field except `content_hash` itself — including `prev`,
so each receipt commits to the entire chain behind it.

### Chain rule

- Line 1: `prev: null`.
- Line N: `prev` = line N−1's `content_hash`.
- The ledger is **append-only**. Rewriting, reordering, or deleting lines is
  a contract violation and is detectable given a trusted head.

### `ledger_head` anchor

`run.json` (the run manifest) records the head:

```json
{
  "v": "openprose.run.v1",
  "run_id": "20260716-093000-a1b2c3",
  "program": "examples/01-hello-world.prose",
  "state_backend": "filesystem",
  "status": "completed",
  "receipt_count": 7,
  "ledger_head": "sha256:07aa…"
}
```

The VM updates `receipt_count` and `ledger_head` on every append (via
`emit_receipt`). A truncated-but-internally-consistent ledger no longer
matches the anchored head.

### What this guarantees — and what it does not

**Guaranteed (chain consistency):** given a trusted `run.json`, any
corruption, reordering, insertion, deletion, or truncation of
`receipts.jsonl` is detectable by recomputing hashes (`openprose-tools
verify`).

**Not guaranteed:** tamper-proofing. Whoever can rewrite the ledger can
rewrite `run.json` too. v1 carries no signatures — this mirrors the upstream
v1 posture (null signer; "signed" means chain consistency at the meaning
layer). A signature seam is deliberately left for a future version and MUST
be added as a new optional field, never by reinterpreting existing ones.

## Writer rules (VM obligations)

1. Append exactly one receipt per completed statement instance — after the
   statement's effects are durable (binding written, state.md updated).
2. Discretion evaluations always emit a `discretion` receipt with
   `detail.condition` (the marker text), `detail.outcome` (a **stable
   literal** — `"true"`/`"false"` or the chosen option name, never mixed
   with prose), `detail.branch` (which branch was taken, if branching), and
   optionally `detail.reason` (free-text justification) — this is what
   makes deterministic replay possible without parsing.
3. A `failed` statement still gets its receipt (with `error` populated) —
   failures are part of the record, including each retry
   (`error.retry_count`).
4. Program start and end emit `control` receipts (`detail: {"event":
   "run_start"}` / `{"event": "run_end", "outcome": …}`), so an interrupted
   run is distinguishable from a completed one even without `run.json`.
   Run-level control receipts are not statements: they carry the reserved
   `statement_id: "run"` (never a `sNNN` ID).
5. Receipts are never edited. A correction is a new receipt.

## Versioning

`openprose.receipt.v1` is append-frozen: fields may be **added** in future
versions (bumping `v`), existing fields are never renamed, retyped, or
reinterpreted. Consumers MUST ignore unknown fields and MUST refuse ledgers
whose `v` they do not know.
