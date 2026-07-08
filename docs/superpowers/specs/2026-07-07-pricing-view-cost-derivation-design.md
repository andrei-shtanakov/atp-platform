# Design: pricing-view — derived cloud-$ over stored per-class usage

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** ADR-ECO-003d (cost-pricing mechanics) — `_cowork_output/decisions/2026-07-02-adr-eco-003d-cost-pricing-mechanics.md` in the dev-only sibling workspace (not committed to this repo; pointer, not a link — see CLAUDE.md on `../_cowork_output/`)
(cost-pricing mechanics), amends 003c/003a/003b
**Scope:** ATP-only. Surface **A** (costing module + view over sweep reports).
No arbiter coordination, no dashboard UI, no `benchmark_runs` migration.

---

## Problem

The broad model matrix (003c) exposed dishonest token comparison: `claude_code=579k`
vs `anthropic_api=9k` on the **same** sonnet model — ×64 from the agentic harness's
prompt-caching, not from the model. A fair cost axis must (a) price cache classes at
their reduced tariff (`cache_read` ~10% of input at Anthropic), and (b) not conflate
harness architecture with model cost.

The data plumbing is already in place: after #235 (ADR-003d #1(a)), each `report_benchmark`
per-task payload carries per-class usage summed over runs —
`input_tokens` / `output_tokens` / `cache_creation_tokens` / `cache_read_tokens` /
`usage_source`. What is missing is the **derivation**: turning that stored per-class usage
into a reproducible cloud-`$` value at report time.

## Non-goals

- **Dashboard leaderboard `$` column** — later, after 003b + arbiter migration. The core
  pricer built here is reusable by that surface when it lands.
- **`benchmark_runs` migration (arbiter #1(b))** — separate repo, separate ADR action.
- **Baking `$` into the payload** — violates ADR D4 (derived-not-stored). Payload keeps
  raw usage; `$` is computed at report time.
- **token_counter estimated-fallback** — deferred (see §Scope boundary).
- **Local `$`** — out of scope per 003c D4 (local = throughput/VRAM, not `$`).
- **Touching `atp.cost.PricingConfig`** — it stays as the heuristic contour for pre-run
  `estimate`, `llm_judge`, `factuality`, and adapter cost tracking. This work adds a
  separate cache-aware cloud pricer; it does not unify the two (YAGNI — 003d scopes only
  the report axis).

## Principles (from ADR-003d)

- **Store raw, derive cost.** Raw usage is stored; `$` computed at report time; a price
  change → re-derive the view **without a re-sweep**.
- **Measured over estimated.** Provider `usage` (with cache classes) is ground truth;
  LiteLLM is a **pricer over measured tokens**, never a re-counter.
- **Library, not gateway.** LiteLLM used as a library (`cost_per_token`); no Proxy —
  routable harnesses are CLI agents that can't sit behind it.
- **One price contour.** Open-tail prices come from the same catalog/discovery contour
  (003a/003b) via `register_model`; no 4th source of truth.
- **Provenance, not silent averaging.** `usage_source ∈ {measured, estimated}` split into
  separate columns; silent-zero prices flagged, never averaged as "free".

## Token-class invariant (the correctness crux)

`input_tokens` does **not** mean the same thing across shims today — pricing on the raw
fields would be dishonest, exactly the failure this feature exists to fix:

| Shim | Provider convention | `input_tokens` | Cached input |
|---|---|---|---|
| `claude_code` (`method/spawners/claude_code_shim.py:60`) | Anthropic | uncached **delta** only | separate additive classes `cache_creation_input_tokens` / `cache_read_input_tokens` |
| `codex_cli` (`method/spawners/codex_cli_shim.py:65`) | OpenAI | **full** prompt (cached included) | `cached_input_tokens` = **subset** of input, under its own key, **not** read by `_grade_case` |

Consequence unmitigated: claude_code prices correctly, but codex is priced with its cached
input billed at the full rate (cache invisible) → systematic **overcount** for a caching
harness. Naive summing (`input + cache_read + cache_creation`) would additionally
**double-count** any shim that leaves cached inside `input_tokens`.

**Normalized pricer contract (pinned here):**
- `input_tokens` = **billable uncached** prompt tokens.
- `cache_read_tokens`, `cache_creation_tokens` = additive, **mutually exclusive** from
  `input_tokens`, priced at their reduced tariffs.
- `output_tokens` = full completion (reasoning already included per provider convention).

**Where normalization happens — at the shim edge** (clean core; each shim knows its
provider's convention):
- `claude_code` already conforms — no change.
- `codex_cli` must split: emit `input_tokens := input − cached_input`,
  `cache_read_tokens := cached_input` (OpenAI cached input ≈ Anthropic `cache_read`, billed
  at a discount — confirm litellm applies the OpenAI cache tariff to `cache_read_input_tokens`
  during the plan via context7).
- **Audit every routable shim** against the invariant as an explicit plan step; any shim
  that can't be normalized has its non-conforming classes zeroed + `usage_source` left such
  that the pricer marks `cost_unknown` rather than guessing.

**Data lineage — named, placed contract stamp.** The normalized contract has a concrete
name and value: **`usage_contract = "cloud_pricing_usage_v1"`** (`input_tokens` = uncached
billable; `cache_*` additive). It is emitted **top-level in the `report_benchmark` payload**
by the payload builder (once shims conform) and **echoed in the sidecar** `cost_view.json`.
The version bumps whenever token semantics change. Existing sweep reports predate this and
carry **no** `usage_contract` → the view treats a missing/older stamp as **not priceable
with cache discount** and flags it, never silently mixing it with `v1` rows. So the plan
**re-sweeps the routable set** under `cloud_pricing_usage_v1` before the view is
authoritative.

## Architecture

```
report_benchmark_*.json (per_task per-class usage, usage_source)
        │  read at report time (derived-not-stored)
        ▼
method/price_reports.py  ──loads──►  method/price_overrides.toml (open tail)
        │                                     │
        │  derive_cost_view(reports)          │ register_model(...)
        ▼                                     ▼
atp/cost/cloud_pricer.py  ── litellm.cost_per_token(cache-split) ──► CasePrice
        │  measured → priced with cache-split (cache_read at reduced tariff)
        │  usage_source != measured → cost_unknown (estimation deferred)
        │  model absent from map → price_unknown (silent-zero detect)
        ▼
CostView: measured-$ / estimated-$ (empty) columns + reliability flag
          + price_map_version stamp
```

### Component 1 — core pricer (`packages/atp-core/atp/cost/cloud_pricer.py`)

Pure, path-agnostic, lazy-imports litellm. Reusable by future dashboard surface.

- `PerClassUsage` — dataclass mirroring payload fields: `input_tokens`, `output_tokens`,
  `cache_creation_tokens`, `cache_read_tokens`, `usage_source: str | None`.
- `CasePrice` — result: `usd: Decimal | None`, `usage_source: str | None`,
  `price_unknown: bool`, `cache_price_unknown: bool`, `cost_unknown: bool`,
  `pricing_scope ∈ {cloud, local_excluded}`, `price_map_version: str`.
- `PriceOverrides` — parsed open-tail entries; registered once at pricer construction via
  `litellm.register_model(...)`. Core does **not** hardcode a file path — the caller loads
  and passes overrides (clean core, details at the edge).
- `CloudPricer.price_case(usage, model) -> CasePrice`:
  - **local model** (`pricing_scope="local_excluded"`, detected by the view — see
    Component 3): `usd=None`, `cost_unknown=True`, but tagged as *deliberately excluded*,
    not "missing price". Local `$` is out of scope (003c D4).
  - **measured** (`usage_source == "measured"`): `litellm.cost_per_token(model=model,
    prompt_tokens=input, completion_tokens=output, cache_read_input_tokens=cache_read,
    cache_creation_input_tokens=cache_creation)` → sum of (prompt, completion) cost.
    Cache-split priced at the reduced tariff. **litellm API assumption — MUST be verified
    before implementation** (first plan task): confirm via context7 that the real signature
    accepts these cache kwargs and applies the OpenAI cache tariff to `cache_read_input_tokens`.
  - **cache price unknown** (override entry has `cache_pricing="unknown"`, or the map has no
    cache tariff): input/output priced, cache classes fall back to the full input tariff →
    `cache_price_unknown=True`. The `$` is **not** presented as exact; the case counts into
    `cache_pricing_unknown_cases`. This is distinct from `price_unknown` (no price at all).
  - **silent-zero detect**: model unknown to the map (even after `register_model`) and
    `$ == 0` with nonzero tokens → `price_unknown=True`, `usd=None`.
  - **not measured** (`usage_source` is None/other): `cost_unknown=True`, `usd=None` —
    estimation not built (§Scope boundary).
  - **litellm missing**: raise `PricingDependencyError` (hint wording in §Files).
- `price_map_version = f"litellm-{litellm.__version__}+overrides-{sha8}"` where `sha8` is
  the sha256 prefix of the overrides file bytes (or a constant sentinel when no overrides).

**Model resolution.** `price_case` needs a `model`, but a report carries `agent_id`, not a
model. `agent_id = f"{harness}@{model}"` (`method/run_pipe_check.py:135`) → the view derives
`model = agent_id.split("@", 1)[1]` (fall back to the catalog once 003b lands). A bare/legacy
`agent_id` without `@` → `price_unknown` (never guess from the filename). Resolution lives in
the **view** (Component 3); the core pricer stays given an explicit model string.

### Component 2 — open-tail prices (`method/price_overrides.toml`)

Hand-curated interim source for models absent from the LiteLLM community map (mimo,
glm-5.1, qwen3.6), in the shape `litellm.register_model` expects. Co-located with the
pipe-check consumer because it is the open tail our sweeps exercise.

**Every entry carries provenance** (so "hand-curated interim" does not silently become a 4th
source of truth): `source` (URL/citation), `effective_date`, `currency`, `unit`
(e.g. per-1M-tokens), `cache_pricing ∈ {known, unknown}`, and free-text `notes`. An entry
with `cache_pricing = unknown` prices input/output only; cache classes fall back to the full
input tariff and the case is flagged (its cache discount is not trustworthy).

**Forward-compat to 003b:** when the catalog loader lands, these overrides — provenance and
all — fold into the catalog/discovery contour (discovery refreshes price per 003d D5); this
file is the interim, ATP-only stand-in that keeps pricing-view unblocked now.

### Component 3 — view surface (`method/price_reports.py`)

- `derive_cost_view(reports: Iterable[dict], overrides_path: Path) -> CostView`:
  reads saved `report_benchmark_*.json`, resolves `model` from `agent_id`, pulls `per_task`
  per-class usage, aggregates per agent, prices each via `CloudPricer`, and produces:
  - **measured-`$`** column (agents with measured usage),
  - **estimated-`$`** column (empty for now — schema present, values deferred),
  - a **formal, numeric** reliability block per agent — not a vague "dominance" — with the
    raw counts so the consumer sets its own bar:
    `total_cases`, `measured_cases`, `cost_unknown_cases`, `price_unknown_cases`,
    `cache_pricing_unknown_cases`, `estimated_cases`, `local_cases`, and a derived
    `reliability_status ∈ {ok, degraded, unreliable}`. Status is computed over
    **cloud** cases only (`local_cases` excluded from the denominator, since local `$` is
    intentionally out of scope): `ok` = all cloud cases measured & fully priced;
    `unreliable` = any of cost_unknown/price_unknown/estimated is a strict majority of cloud
    cases; `degraded` = otherwise nonzero (incl. any `cache_pricing_unknown`). Thresholds
    defined here, not left to reader intuition.
  - **local detection.** An agent is `pricing_scope="local_excluded"` when its model is a
    local class per 003c (interim: a `local` harness/model set in `price_overrides.toml`
    until the 003b catalog carries the class). Local agents are reported with their tokens
    and `local_excluded`, never as a missing/zero price.
  - **`price_map_version`** and **`token_contract`** stamps on the view.
- **`total_cost_usd` is legacy/ignored.** The payload still sums `total_cost_usd` from old
  `metrics.cost_usd` (`atp/reporters/benchmark_reporter.py:168`) — populated only for
  `claude_code`, `null` elsewhere. The view **does not read it**; it surfaces `derived_usd`
  under a distinct name so no one compares the old field against the new derivation.
- Thin CLI entry (`python method/price_reports.py <reports-dir>`) so `$` re-derives from
  stored reports **without a re-sweep** — the D4 property. Prints a table and/or writes a
  sidecar (`cost_view.json`) next to the reports.

## Scope boundary — estimated-fallback deferred

The token_counter estimated path (003d D2/D3) is **not** built here. Estimating tokens
requires the raw prompt/completion text, which is present only at grade-time in the shim —
it is **not** in the stored payload. So the estimated path architecturally belongs to an
upstream grade-time change, not this report-time view.

What this PR does include: the measured/estimated **columns** and the **reliability flag**
in the schema, and correct handling of non-measured rows — `usage_source != "measured"`
→ `cost_unknown`, counted into the reliability flag. The estimation computation itself is
a separate follow-up.

## Testing

- **Unit — pricer** (`test_cloud_pricer.py`), with litellm monkeypatched:
  - measured with cache-split → cache_read priced at reduced tariff (assert the split
    reaches `cost_per_token` and cheapens vs naive input pricing);
  - silent-zero → `price_unknown=True`, `usd=None`;
  - `register_model` applied from overrides (a tail model prices non-zero);
  - `usage_source=None` → `cost_unknown=True`;
  - litellm absent → `PricingDependencyError` with the install hint.
- **Unit — view** (`test_price_reports.py`): aggregate over fixture reports → correct
  measured-`$` per agent, model resolution from `agent_id` (incl. bare-id → `price_unknown`),
  a `local_excluded` agent reported with tokens but no `$` (not "missing"), a
  `cache_pricing="unknown"` override → `cache_price_unknown` (distinct from `price_unknown`),
  `total_cost_usd` ignored, missing `usage_contract` flagged, and the numeric reliability
  block with the cloud-only denominator.
- **Contract test vs real litellm** (`test_cloud_pricer_contract.py`, marked `slow`/opt-in):
  monkeypatch guards our code but not our *assumptions* about the third-party API. One test
  runs against the installed `litellm.cost_per_token` for a known model, asserting the real
  signature accepts the cache-split kwargs, the return shape is `(prompt, completion)`, and an
  unknown model behaves as the silent-zero path expects. This is the guard against a wrong API
  assumption; skipped when the `[pricing]` extra is absent.
- **Fixtures**: a `report_benchmark` payload with per-class measured usage (conforming to the
  normalized token contract) + one with `usage_source=None`.

## Files

| Path | Change |
|---|---|
| `packages/atp-core/pyproject.toml` | new `[pricing]` optional extra → `litellm` (pricer lives here) |
| `pyproject.toml` (root) | proxy extra → `atp-core[pricing]` |
| `packages/atp-core/atp/cost/cloud_pricer.py` | new — pure pricer, lazy litellm |
| `method/spawners/codex_cli_shim.py` | normalize usage to the token invariant (split cached out of input) |
| `atp/reporters/benchmark_reporter.py` | stamp top-level `usage_contract = "cloud_pricing_usage_v1"` in payload |
| `method/price_overrides.toml` | new — open-tail prices + provenance + `local` set |
| `method/price_reports.py` | new — view + model resolution + re-derive CLI |
| `tests/unit/cost/test_cloud_pricer.py` | new — monkeypatched pricer |
| `tests/unit/cost/test_cloud_pricer_contract.py` | new — opt-in vs real litellm |
| `tests/unit/method/test_price_reports.py` | new — view + resolution + reliability |
| `tests/fixtures/` | per-class usage report fixtures |

**Error-hint wording:** the `PricingDependencyError` hint targets external users
(`pip install atp-platform[pricing]`); in-repo the dev path is `uv sync --extra pricing` —
the message names both so neither audience is misled.

Re-sweep of the routable set under the normalized token contract, and Docs/CLAUDE.md pointer
updates, are explicit steps in the implementation plan.
