# Design: pricing-view — derived cloud-$ over stored per-class usage

**Date:** 2026-07-07
**Status:** Approved (brainstorm) — ready for implementation plan
**ADR:** [ADR-ECO-003d](../../../../_cowork_output/decisions/2026-07-02-adr-eco-003d-cost-pricing-mechanics.md)
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
  `price_unknown: bool`, `cost_unknown: bool`, `price_map_version: str`.
- `PriceOverrides` — parsed open-tail entries; registered once at pricer construction via
  `litellm.register_model(...)`. Core does **not** hardcode a file path — the caller loads
  and passes overrides (clean core, details at the edge).
- `CloudPricer.price_case(usage, model) -> CasePrice`:
  - **measured** (`usage_source == "measured"`): `litellm.cost_per_token(model=model,
    prompt_tokens=input, completion_tokens=output, cache_read_input_tokens=cache_read,
    cache_creation_input_tokens=cache_creation)` → sum of (prompt, completion) cost.
    Cache-split priced at the reduced tariff.
  - **silent-zero detect**: model unknown to the map (even after `register_model`) and
    `$ == 0` with nonzero tokens → `price_unknown=True`, `usd=None`.
  - **not measured** (`usage_source` is None/other): `cost_unknown=True`, `usd=None` —
    estimation not built (§Scope boundary).
  - **litellm missing**: raise `PricingDependencyError` with hint
    `pip install atp-platform[pricing]`.
- `price_map_version = f"litellm-{litellm.__version__}+overrides-{sha8}"` where `sha8` is
  the sha256 prefix of the overrides file bytes (or a constant sentinel when no overrides).

### Component 2 — open-tail prices (`method/price_overrides.toml`)

Hand-curated interim source for models absent from the LiteLLM community map (mimo,
glm-5.1, qwen3.6). One TOML table per model with input/output (and cache, where known)
per-token prices, in the shape `litellm.register_model` expects. Co-located with the
pipe-check consumer because it is the open tail our sweeps exercise.

**Forward-compat to 003b:** when the catalog loader lands, these overrides fold into the
catalog/discovery contour (discovery refreshes price per 003d D5); this file is the
interim, ATP-only stand-in that keeps pricing-view unblocked now.

### Component 3 — view surface (`method/price_reports.py`)

- `derive_cost_view(reports: Iterable[dict], overrides_path: Path) -> CostView`:
  reads saved `report_benchmark_*.json`, pulls `per_task` per-class usage, aggregates per
  agent, prices each via `CloudPricer`, and produces:
  - **measured-`$`** column (agents with measured usage),
  - **estimated-`$`** column (empty for now — schema present, values deferred),
  - **reliability flag** per tier: share of `estimated` and `price_unknown` cases, surfaced
    like the runs=1 variance flag (dominance → "cost unreliable").
  - **`price_map_version`** stamp on the view.
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
  measured-`$` per agent + reliability shares (estimated / price_unknown).
- **Fixtures**: a `report_benchmark` payload with per-class measured usage + one with
  `usage_source=None`.

## Files

| Path | Change |
|---|---|
| `pyproject.toml` | new `[pricing]` optional extra → `litellm` |
| `packages/atp-core/atp/cost/cloud_pricer.py` | new — pure pricer, lazy litellm |
| `method/price_overrides.toml` | new — open-tail prices |
| `method/price_reports.py` | new — view + re-derive CLI |
| `tests/unit/cost/test_cloud_pricer.py` | new |
| `tests/unit/method/test_price_reports.py` | new |
| `tests/fixtures/` | per-class usage report fixtures |

Docs/CLAUDE.md pointer updates deferred to the plan's final step.
