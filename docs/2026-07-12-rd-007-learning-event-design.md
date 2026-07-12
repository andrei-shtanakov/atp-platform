# Design: LearningEvent v1 — learning through governance, no silent self-modification (RD-007)

**Date:** 2026-07-12 · **Status:** Draft v1 (design phase; code after approval)
**Basis:** contracts-roadmap RD-007 (`prograph-vault/authored/roadmaps/contracts-v1.yaml`,
phase 6, deps RD-004 ✅ verified); consolidated roadmap phase 6: "LearningEvent — через
steward-governance (PR/CODEOWNERS), не silent-write; **не повторять OpenOPC `save_skill`**".
**Owners:** atp-platform (schema + graduation targets) + ecosystem-kb; consumers
robin-runtime, prograph-vault, dispatcher.

## 0. The invariant (normative)

> **LearningEvent is observational. It never mutates governed knowledge by itself.
> Graduation from event to durable artifact happens only through human-reviewed PR.
> Runtime producers may append to their own var/event stores; governed paths are
> modified only by repository PR workflow.**

RD-007 is not "another learning process" — it is the **contractual protection against
silent self-modification**. Recon confirmed the anti-pattern does not exist in the
ecosystem today (no runtime writes derived knowledge into catalogs/prompts/KBs); what
is missing is the *enforcement layer*: no CODEOWNERS anywhere, catalog governance is
convention only. RD-007 formalizes the existing discipline instead of inventing a loop.

## 1. Scope

**v1 (this design):**
- `method/contract/learning-event-v1.schema.json` — the event shape (next to
  `report_benchmark-v1.schema.json`, this repo's contract convention) + fixtures +
  contract test (`tests/contract/` convention).
- **Governed-paths declaration** (§4) + **CODEOWNERS as v1 acceptance** — without it,
  "flows through PR/CODEOWNERS governance" stays methodology, not evidence.
- **First end-to-end slice** (§5): the Robin regression loop —
  gap → event → graduation PR with an eval case → regression gate.
- Graduation protocol (§6).

**Deferred to M2 (deliberately):**
- Conformance-CI: byte-check of vendored copies (today hidden in un-shipped
  `_cowork_output/devtools/`), "no runtime writes to governed paths" scanner.
- Aggregation read-model over federated event stores (dispatcher) — read-plane only,
  the WorkCorrelation pattern; never a write-plane.
- New producers beyond the first slice (experiment.py recommendations, catalog
  discover-proposals) — they adopt the schema, not new machinery.

## 2. Roles: producer vs graduation-target owner

Provenance must survive: **ATP did not "see" the gap — Robin did.** ATP accepts it as
material for a regression case.

| Role | Who (first slice) | Owns |
|---|---|---|
| **Producer** | robin-runtime (`gaps.py` → `selfreview.py`) | The observation: appends events to its own store (`var/`), never touches governed paths |
| **Graduation target owner** | atp-platform (`robin_shim.py`, `examples/test_suites/robin_regression.yaml`) | The schema, the eval gate, and acceptance of graduation PRs |
| **Governance substrate** | steward (risk-model + gates, RD-004) + CODEOWNERS | Governed paths classify policy/high → mandatory owner approval; RD-007 adds no new teeth, it declares the paths |

## 3. Event shape (LearningEvent v1)

```json
{
  "schema_version": "1",
  "event_id": "01KX9ZD3ULIDLIKE0000000000",
  "producer": "robin-runtime",
  "kind": "gap",
  "ts": "2026-07-12T10:00:00Z",
  "source": {"store": "var/gaps.jsonl", "id": "gap-2026-07-12-0042"},
  "proposed_target": "eval-case",
  "evidence_refs": [
    {"kind": "log", "pipeline_id": "01KX8V7Z9DHBKYWGSN2KTWM8AB"}
  ],
  "payload": {"query": "...", "failure": "zero-retrieval"}
}
```

- `kind` — closed v1 vocabulary of observation classes:
  `gap | eval-signal | recommendation`.
- `proposed_target` — closed v1 vocabulary of graduation targets (optional; filled by
  clustering/review, not necessarily at observation time):
  `eval-case | catalog | prompt | kb | policy`.
- `source` — required producer-local pointer (store + id): every event is traceable to
  its append-only origin even when no cross-repo evidence exists.
- `evidence_refs[]` — optional but encouraged; the inline definition is a **vendored
  pinned copy** of Maestro `contracts/observability/evidence-ref.schema.json`
  (cross-repo contracts are vendored in, never referenced out; vendored-from SHA in
  the schema header; byte-conformance moves to M2 CI).
- `payload` — producer-specific object; never machine-graduated (a human reads it in
  the PR, the same way `note` is never machine-parsed in EvidenceRef).

## 4. Federated stores + governed paths

**Stores are federated, producer-owned, append-only** — `var/gaps.jsonl`,
`benchmark_runs`, future `var/learning_events.jsonl`. No central queue: it would add
infrastructure and mint a contested SSOT. Aggregation, if ever needed, is a
**read-model** (WorkCorrelation pattern), not a new write-plane.

**Governed paths v1** (modified only via PR; CODEOWNERS in each owning repo):

| Repo | Paths |
|---|---|
| atp-platform | `method/contract/**`, `examples/test_suites/robin_regression.yaml`, `method/spawners/robin_shim.py`, `method/agents-catalog.toml` |
| robin-runtime | `src/robin/gaps.py`, `src/robin/selfreview.py` |
| prograph-vault | `authored/**` |

(The catalog was already hand-maintained by convention — `"ЭТО ИСТОЧНИК, НЕ
СГЕНЕРЁННОЕ"` — CODEOWNERS turns the convention into evidence.)

## 5. First slice: the Robin regression loop

Already an embryo of the correct shape (recon): append-only `gaps.jsonl` →
`selfreview.py` clusters into a deterministic work-order → maintainer DM → human PR.
v1 formalizes it end to end:

1. **Observe** (robin-runtime): gap logged to `var/gaps.jsonl`; selfreview emits
   LearningEvent-conformant records (adapter over the existing gap entries — the store
   stays, the shape becomes contractual).
2. **Propose** (robin-runtime → human): the weekly work-order references `event_id`s.
3. **Graduate** (atp-platform PR): a human (or an agent authoring a PR under review)
   adds a case to `examples/test_suites/robin_regression.yaml`; the PR body cites the
   `event_id`s and their `evidence_refs`. CODEOWNERS forces owner review.
4. **Gate**: the regression suite gates changes to Robin's prompt/tools/retrieval
   (`robin_shim.py`, stage 4 of the self-improvement loop) — the learned artifact
   immediately protects against regressions of the very failure it came from.

Merged PR = learned. There is no other path.

## 6. Graduation protocol (normative)

- A graduation PR MUST cite the `event_id`(s) it graduates and their `source` stores.
- A graduation PR MUST touch only governed paths owned by the target repo.
- Producers MUST NOT open-and-self-merge graduation PRs (human review per CODEOWNERS;
  the ecosystem git-workflow rule "мерж делает человек" already says this).
- Events are never deleted or rewritten by graduation — the store is append-only;
  "graduated" is derivable (a PR references the event), not a mutation of the event.
- Rejected proposals need no event mutation either: absence of a referencing PR is
  the state.

## 7. Milestones

- **M0 (this doc):** design review; decisions on OQ below.
- **M1a (atp-platform):** `learning-event-v1.schema.json` + fixtures + contract test;
  `CODEOWNERS` covering the governed paths (v1 acceptance); this doc's §6 as the
  protocol reference.
- **M1b (robin-runtime):** selfreview emits conformant events (adapter over
  `gaps.jsonl`); repo CODEOWNERS. Separate PR in that repo.
- **M1c (prograph-vault):** CODEOWNERS for `authored/**`; RD-007 evidence_rules in
  `contracts-v1.yaml` once paths are fixed (implementation = schema file;
  verification = atp CODEOWNERS file).
- **M2 (deferred):** conformance-CI (vendored byte-checks, no-runtime-writes scanner);
  additional producers (experiment recommendations, catalog proposals) adopt the
  schema; dispatcher read-model.

## 8. Open questions

- **OQ-1 — `event_id` semantics:** ULID minted at append time (proposal), and is
  producer-scoped uniqueness enough (dedup = `(producer, event_id)`)?
- **OQ-2 — gaps.jsonl migration:** adapter emitting conformant events alongside the
  existing format (proposal; zero breakage) vs migrating the store format itself.
- **OQ-3 — CODEOWNERS reviewer identity:** single-owner ecosystem today (Andrei) —
  CODEOWNERS names the user directly in v1; role-based teams when they exist.
