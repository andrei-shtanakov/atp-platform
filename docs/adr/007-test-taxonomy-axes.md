# ADR-007: Test Taxonomy Axes + Deterministic-Checker Model

**Status**: Accepted (2026-06-16)
**Date**: 2026-06-16
**Companion**: `docs/superpowers/specs/2026-06-16-req-extraction-deterministic-vertical-design.md`
— the first vertical built under this taxonomy (req-extraction, deterministic).
**Builds on**: ADR-006 (case/grader/harness spine + `task_type` store dimension).
This ADR adds the **map** that keeps test-types orthogonal as more land, and
pins down how deterministic checks are modeled.

## Context

R-07 validated the first capability eval (`code-review`, text-out, on a diff).
More verticals are planned — req-extraction, architecture-design, repo-analysis,
coding — and a colleague's draft specs (PR #186: `structured-method-output`,
`artifact-corpus-grounding`) push toward agentic, file-reading tests.

The recurring worry (Andrei's "зообарк"): every new test feels like a new
*kind* of thing — its own grader, its own way of reaching the model, its own
difficulty notion — and the set becomes an unnavigable zoo.

The root cause is **mixing orthogonal concerns in one conversation**: "CLI+shim
vs HTTP/Container" (how the agent touches the world), "findings_match vs rubric"
(how we decide pass/fail), "clean…very_severe" (how hard), and "review vs
req-extraction" (what task) are four *independent* axes. Discussed tangled, they
read as a proliferation of types. Discussed as axes, any test is just a point.

## Decision

### 1. A test is a coordinate over five orthogonal axes

Every test (and every result row) is addressed by the same five axes. Each is a
case field and a store column. There is no such thing as a "test type" beyond a
combination of axis values.

| Axis | Field(s) | Values | Status |
|------|----------|--------|--------|
| **Task** (what task) | `task_type` (+ `language`) | review, req-extraction, architecture-design, repo-analysis, coding, … | exists (ADR-006, SP-4) |
| **Run mode** (how the agent interacts) | `run_mode` | `text_out` → `read_only_corpus` → `workspace` | **formalized here** (field added; only `text_out` wired) |
| **Grading** (how we decide pass/fail) | `grader.type` (+ `grader.checker`) | `programmatic`(+checker) · `rubric` (non-gating) · `human` | exists (ADR-006) |
| **Difficulty** (how hidden the defect) | `axis_level` | clean → mild → moderate → severe → very_severe | exists |
| **Lifecycle** (why the test exists) | `suite_type` | probe → regression → held_out | exists |

Four of five already exist as schema fields and store columns (SP-1/SP-4). Only
`run_mode` was implicit ("text_out everywhere"); this ADR makes it explicit.
`suite_type` values are the canonical contract enum (`regression | probe |
held_out`, `agent-eval-case.schema.json`): `probe` hunts new failures, `regression`
is frozen and re-run per agent version, `held_out` is hidden to prevent
overfitting. "Routing benchmark" is how those results are *consumed*, not a
`suite_type` value.

**Naming guard — do not call axis 1 "capability".** `capability` is an *already
taken* closed enum of **quality** dimensions (`correctness | calibration |
efficiency | safety_compliance | recoverability | adaptation`,
`atp_method/schema.py`), not task identity. Axis 1 is `task_type`. `capability`
(quality stressed), `construction_axis` (what is varied), and `family` (case
grouping) are finer facets *within* a `task_type` — distinct dimensions, not
synonyms for it. Reusing "capability" for the task axis would recreate exactly the
`benchmark_id`-vs-`task_type` confusion ADR-006 already corrected.

### 2. Deterministic checks are named checkers under `type: programmatic`

The closed `GraderType` vocab (`exact | regex | programmatic | rubric |
model_graded | human`) stays closed. Deterministic, machine-readable checks
(`citation_grounding`, `findings_match`, `json_path`, …) are **named checkers from
a registry, selected under `type: programmatic`** via `grader.checker`, each reading
its own `grader.config`:

```yaml
grader:
  type: programmatic
  checker: json_path
  config: { assertions: [ { path: "$.x", op: equals, expected: null } ] }
```

This is the existing `findings_match` precedent (ADR-006 §A-1, PR #176)
generalized. Adding a deterministic check = registering a checker, **not** a new
`GraderType` value and **not** a `checks[]` array.

Per-checker config is carried in `grader.config`. The safety property that
matters is that config is **validated at load, before any run** — not at runtime.
As shipped (req-extraction vertical), that validation lives in the pydantic
`Grader` validator as a per-checker rule (e.g. `checker == "json_path"` requires a
non-empty `config.assertions`), mirroring how `findings_match` is already
validated. This keeps the `Grader` model thin (new checkers do not each bolt typed
fields onto it — that bloat is itself a form of the zoo). The richer form — each
checker **declaring its own pydantic config model in the registry**, with the
loader validating `grader.config` against it — is the intended end state but is
**deferred to a follow-up** that migrates both `findings_match` and `json_path`
together (it would add a cross-package dependency from the case schema to the
checker registry, so it earns its own change). (`findings_match` keeps its existing
typed fields for back-compat.)

**Deferred:** a heterogeneous `grader.checks[]` array (multiple *different*
checkers on one case, e.g. `schema` + `findings_match` + `json_path` together).
Not needed yet — a case needing shape-validation + field-checks gets shape from
the `output_contract` schema assertion and fields from one `json_path` checker.
When a case genuinely needs two distinct checkers, `checks[]` returns as its own
ADR.

### 3. Run mode: declare now, wire lazily, pick the lightest tier

`run_mode` is a case field now, defaulting to `text_out`. The `Literal` type
admits all three values, but the loader **rejects any tier that is not yet wired**
(the wired set is `{text_out, read_only_corpus}` — `WIRED_RUN_MODES` in
`atp_method/schema.py`) with an explicit load-time error. A case must not be able
to *declare* a fidelity tier the harness cannot actually deliver — that silent gap
is the very validity problem this section exists to prevent. Tier support:

- **`text_out`** — agent gets a text blob, returns text/JSON. CLI-adapter + shim
  (stdin/stdout). Runs on every adapter.
- **`read_only_corpus`** — agent discovers and reads files via tools (`file_read`),
  ignores distractors, cites sources. Loader-wired; runs on a tool-capable adapter
  (the `anthropic_api` shim does the `file_read` loop). The CLI-adapter pipe-check
  (`run_pipe_check.py`) has no corpus/tool wiring, so it **skips** `read_only_corpus`
  cases with a logged notice rather than failing them uniformly.
- **`workspace`** — agent writes files / executes code. Container + exec. *Unwired
  (eval-results architecture SP-6).*

Rule: **choose the lightest tier that closes the capability's fidelity gap.**
For text-out review the gap is nil, so cheap == valid. For a capability whose
essence is navigating a document set or editing files, text-out has a real gap
(you would measure extraction-from-given-text, not from discovered files), so the
richer tier is required for *validity*, not gold-plating.

### 4. Adapter selection follows run_mode, per capability

CLI+shim is the optimum for `text_out`, not a universal default. The
HTTP/Container/MCP adapters become correct when `run_mode` advances
(`read_only_corpus`/`workspace`) — that is the documented revisit trigger from
the R-07 locked decision, not a contradiction of it.

## How this frames PR #186

- **`structured-method-output`** — right direction (deterministic-first, judge
  non-gating). Refined by the companion spec: json_path becomes a **checker under
  `programmatic`** (decision 2), not the spec's `checks[]` array; and it reuses
  the already-built `malformed`/`malformed_rate` (P3) rather than re-defining it.
- **`artifact-corpus-grounding`** — valid and well-designed (SHA-256 manifest,
  path-safety). It is the first `run_mode: read_only_corpus` vertical (decision 3)
  and depends on `structured-method-output` (citations are structured output). It
  is therefore **parked behind** that spec and the read-only-corpus tier — not a
  violation of the CLI+shim decision but the planned evolution of it.

PR #186 stays a draft; this ADR + the companion spec are the reconciled version.

## Consequences

- New tests are placed by choosing one value per axis — no new "type" to invent.
- The store/dashboard already key on these axes, so results from any vertical are
  comparable in one table (the exit from the zoo).
- One small schema addition now (`run_mode` with wired-set validation,
  `output_contract`, `grader.config` validated per-checker).
- A future heterogeneous-check need will require the deferred `checks[]` ADR.
- Two richer run-mode tiers remain to be implemented when a capability needs them.

## Alternatives considered

- **Extend `GraderType`** with `json_path`, `schema`, … — re-opens the closed
  vocab and blurs "strategy vs checker"; rejected for decision 2.
- **Adopt Spec 1's `checks[]` array up front** — most general, but introduces a
  third grading taxonomy before any case needs heterogeneous checks; deferred.
- **Loosely typed `grader.config: dict`** — keeps `Grader` thin but loses
  validation exactly where it matters (assertion shape), deferring errors to
  runtime. Rejected in favour of per-checker registry-declared config models
  validated at load (decision 2).
- **Leave `run_mode` implicit** — keeps the latent "text_out everywhere"
  assumption that made the corpus spec look like a contradiction; rejected.
- **Full JSONPath (wildcards / recursive descent / filters) for `json_path`** —
  multi-node matches make `op: equals` ambiguous, a determinism hole in a
  deterministic grader. Rejected: a `json_path` assertion must resolve to exactly
  one node or it fails (decision recorded in the companion spec); this makes a
  constrained/no-dep resolver at least as strong as a general library here.
