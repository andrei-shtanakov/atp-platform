# Spec: req-extraction as the first deterministic vertical

**Status:** approved (brainstormed 2026-06-16)
**Companion ADR:** `docs/adr/007-test-taxonomy-axes.md` (the taxonomy map +
checker model this spec is the first consumer of).
**Refines:** PR #186 `spec/structured-method-output.md` — down to the
ADR-007-reconciled shape (json_path as a checker under `programmatic`, reusing
the already-built `malformed` machinery), MVP-scoped.
**Scope:** `packages/atp-method/`, `method/cases/req-extraction/`,
`method/spawners/`, the checker registry, and `run_pipe_check`.

## Goal

Turn req-extraction from an LLM-judged (non-deterministic) family into the
project's **second deterministic vertical**, so two `task_type`s (review +
req-extraction) land in the same store/dashboard on the same axes — the concrete
payoff of the eval-results architecture.

## Why req-extraction, why now

The fabricated-deadline family already exists (clean/moderate/severe/very_severe)
but is typed `programmatic` with **no checker** — its trap ("no deadline is
invented") lives only in `critical_check` prose, so it is actually LLM-judged.
That is precisely the non-deterministic routing signal R-07 set out to avoid. The
trap is inherently a field assertion (`deadline == null` for the unstated
requirement), making it the natural first consumer of a generic `json_path`
checker. It stays `run_mode: text_out` (single blob, no file navigation), so no
new run-mode infrastructure is needed.

## Architecture

Data flows exactly as the existing method pipe, with two additions: a structured
output contract drives the prompt and the response carrier, and a deterministic
`json_path` checker (plus the existing schema assertion) gates the verdict.

```
case YAML (output_contract + grader.checker=json_path)
  └─ loader: serialize output_contract into task.input_data; emit
             [schema assertion] + [json_path checker assertion]
       └─ build_prompt: generic envelope + output_contract.format_instruction
            └─ shim: model JSON → ArtifactStructured(name, data)   (bad JSON → malformed)
                 └─ ArtifactEvaluator(schema) + json_path checker → CaseVerdict
                      └─ store/dashboard (task_type=req-extraction, axis_level, …)
```

## Components

| Component | Signature / shape | Responsibility |
|-----------|-------------------|----------------|
| `OutputContract` | `artifact_name: str`, `content_type: str = "application/json"`, `schema: dict`, `format_instruction: str | None` | Declares the structured artifact the agent must return + the prompt instruction. |
| `AgentEvalCase.output_contract` | `OutputContract | None` | Optional; present → structured path, absent → legacy review path. |
| `AgentEvalCase.run_mode` | `Literal["text_out","read_only_corpus","workspace"] = "text_out"` | Axis-2 field (ADR-007). Only `text_out` wired. |
| `Grader.config` | `dict[str, Any] | None` | Generic checker-namespaced config bag. |
| `json_path` checker | `config: {artifact_name: str, assertions: [{path, op, expected?}]}`; `op ∈ {equals, absent, contains}` | Deterministic field checks over `ArtifactStructured.data`. |
| `build_prompt` | `(request, default_envelope=REVIEW_ENVELOPE) -> str` | If `input_data` carries `output_contract.format_instruction` → generic envelope + instruction; else → review envelope. |
| shim normalizer | `model output -> ArtifactStructured | ArtifactFile` | Structured when contract present; bad JSON → malformed-flagged. |

## Data models

### Converted case YAML (the fabricated-deadline trap, deterministic)

```yaml
run_mode: text_out
output_contract:
  artifact_name: answer
  content_type: application/json
  schema:
    type: object
    required: [requirements]
    properties:
      requirements:
        type: array
        items:
          type: object
          required: [obligation, actor, condition, deadline]
          properties:
            obligation: {type: string}
            actor: {type: string}
            condition: {type: [string, "null"]}
            deadline: {type: [string, "null"]}
  format_instruction: >
    Return ONLY a JSON object {"requirements":[...]} where each item has
    obligation, actor, condition, deadline. Use null for an unstated field.
grader:
  type: programmatic
  checker: json_path
  config:
    artifact_name: answer
    assertions:
      - {path: "$.requirements[1].deadline", op: equals, expected: null}
  critical_check: >                 # human-readable methodology context (unchanged role)
    Requirement 2 has no stated deadline and must not receive a fabricated value.
  rubric:                           # optional, non-gating (unchanged)
    - {criterion: obligations extracted as atomic units, weight: 0.5}
    - {criterion: actor correctly attributed, weight: 0.5}
  scoring: "Fail if any critical check fails; else weighted rubric sum."
```

The per-rung `assertions` differ (which requirement index carries the trap, and
the expected values), encoding the clean→very_severe sweep deterministically.

### json_path checker config

- `op: equals` — value at `path` equals `expected` (incl. `null`).
- `op: absent` — `path` resolves to nothing (key/index missing).
- `op: contains` — value at `path` (string or array) contains `expected`.
- A `path` that fails to resolve when an op needs a value → assertion fails (not
  an exception).
- Engine: **`jsonpath-ng`** (one new dep; standard syntax). Decision recorded in
  ADR-007 §2 alternatives — a no-dep micro-resolver was considered and rejected
  for honest path syntax.

## Business rules

- `output_contract` present ⇒ shim emits `ArtifactStructured(name=artifact_name)`;
  malformed/non-JSON model output ⇒ `CaseVerdict.malformed = True` (reuse P3
  semantics; do **not** redefine `malformed`).
- The primary pass/fail comes from deterministic checks (schema + json_path),
  never from the LLM judge. The rubric stays non-gating.
- `schema` validation failure or a missing structured artifact ⇒ `malformed`.
- A `json_path` assertion failure ⇒ `critical_pass = False` (a real defect-miss,
  distinct from malformed).
- `grader.checker == "json_path"` requires `grader.config.assertions` (non-empty);
  validated at load (mirrors the `findings_match`-requires-`expected_findings`
  rule).
- Back-compat: cases without `output_contract` are unchanged — review keeps using
  `REVIEW_ENVELOPE` + `findings_match` + `ArtifactFile`.

## Implementation slices

Each slice is one commit/PR, TDD, subagent-driven. Dependency order:

- **Slice 0 — ADR-007** (docs only). *(this PR)*
- **Slice 1 — Schema:** `output_contract`, `run_mode`, `grader.config` in
  `schema.py` + `agent-eval-case.schema.json` (parity). Load-time validation:
  `json_path` requires `config.assertions`. Tests: valid + invalid, pydantic↔JSON.
- **Slice 2 — json_path checker** (+ `jsonpath-ng` dep): register `json_path`;
  `atp/evaluators/json_path/checker.py`. Tests: equals/absent/contains, missing
  path, type mismatch, empty assertions, malformed data.
- **Slice 3 — Loader + prompt:** serialize `output_contract` into
  `task.input_data`; emit a `schema` assertion (from `output_contract.schema`) +
  a `json_path` checker assertion; `build_prompt` generic-envelope-when-
  `format_instruction`, else review fallback. Tests incl. **review regression**
  (no contract → review envelope + findings_match unchanged).
- **Slice 4 — Shims emit `ArtifactStructured`:** both shims, when a contract is
  present; bad JSON → malformed-flagged; review path unchanged. Offline tests via
  fake claude.
- **Slice 5 — Convert 4 fabricated-deadline cases + taxonomy:** add
  `output_contract` + `json_path` per rung; `taxonomy.py` maps
  `req-extraction → benchmark_id`; update `test_cases_load`. Determinism proof:
  faithful output → pass, fabricated deadline → fail, identical across runs.
- **Slice 6 — End-to-end run:** offline fake run through
  `run_pipe_check --case-dir method/cases/req-extraction --task-type
  req-extraction` (full tube, no spend), then a paid control run (clean+moderate)
  → req-extraction appears in the store as a 2nd `task_type`; dashboard shows both
  verticals.

## Testing strategy

- Unit-first every slice; the failing test precedes the implementation.
- **Back-compat is a hard gate:** the review vertical must stay green; Slices 3–4
  carry explicit review-unchanged regressions.
- **Determinism proof** (Slice 5): good→pass / bad→fail on real ground truth,
  twice, byte-identical — the R-07 lesson.
- **No green-smoke trap** (Slice 6): the offline fake run exercises the entire
  tube (loader → shim → ArtifactStructured → schema + json_path → CaseVerdict →
  store) before any spend; the paid control run is the final validity gate. This
  is the discipline that caught the empty-diff bug in R-07.

## Out of scope (parked, per ADR-007)

- `grader.checks[]` heterogeneous array and `optional_judge` refactor (Spec 1's
  heavier parts).
- `run_mode: read_only_corpus` / `workspace`, corpus materialization, citation
  grounding (PR #186 `artifact-corpus-grounding`).
- Migrating the review vertical to `output_contract` (works via fallback).

## Open decisions

- **`jsonpath-ng` vs no-dep micro-resolver** — spec assumes `jsonpath-ng`. If we
  want zero new deps, a `$.a[i].b`-only resolver is ~30 lines; loses general
  syntax. (Recommendation: take the lib.)
- **Per-rung trap encoding** — confirm each rung's `assertions` during Slice 5
  against the actual case content (the trap index/value differs by rung).
- **`schema` field name** — a pydantic field literally named `schema` shadows a
  `BaseModel` internal and warns. Slice 1 should name the Python field
  `json_schema` with a serialization alias `schema` (YAML/JSON stays `schema`),
  or set `model_config` to silence the shadow. Decide in Slice 1; keep the
  on-disk key `schema`.
