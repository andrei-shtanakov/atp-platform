# Pricing-view Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive a reproducible, honest cloud-`$` value from the per-class token usage
already stored in `report_benchmark` payloads, priced with cache-split tariffs.

**Architecture:** A pure, cache-aware LiteLLM pricer lives in `atp-core`
(`atp/cost/cloud_pricer.py`); a thin view (`method/price_reports.py`) reads saved
`report_benchmark_*.json`, resolves each agent's model, prices per-case, and emits a table +
sidecar with a formal numeric reliability block. Token semantics are normalized at the shim
edge (only the shim knows its provider's convention). Derived-not-stored: `$` is computed at
report time; a price change re-derives without a re-sweep.

**Tech Stack:** Python 3.12+, pydantic, uv workspaces, LiteLLM (optional `[pricing]` extra),
pytest + anyio.

## Global Constraints

- Package management: **uv only**, never pip. Add deps with `uv add`, run tools with `uv run`.
- Line length **88**; type hints on all code; `uv run ruff format .` + `uv run ruff check .`
  + `uv run pyrefly check` clean after every task.
- Async tests use **anyio**, not asyncio.
- **Derived-not-stored** (ADR-003d D4): never bake `$` into the payload or `benchmark_runs`.
- **One price contour** (D5): open-tail prices via `litellm.register_model` from
  `method/price_overrides.toml`; no 4th source of truth.
- **Normalized token contract** `cloud_pricing_usage_v1`: `input_tokens` = billable uncached
  prompt; `cache_read_tokens`/`cache_creation_tokens` additive & mutually exclusive from
  `input_tokens`; `output_tokens` = full completion.
- **Scope: ATP-only.** No `benchmark_runs` migration (arbiter), no dashboard UI, no
  token_counter estimated-fallback (deferred — needs grade-time text absent from payload).
- LiteLLM floor: `litellm>=1.81` (cache-aware `generic_cost_per_token`); the exact behavior
  is pinned by the Task 1 contract test, not assumed from docs.

Spec: `docs/superpowers/specs/2026-07-07-pricing-view-cost-derivation-design.md`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `packages/atp-core/pyproject.toml` | declare `[pricing]` optional extra → `litellm` |
| `pyproject.toml` (root) | proxy extra `pricing = ["atp-core[pricing]"]` |
| `packages/atp-core/atp/cost/cloud_pricer.py` | pure pricer: dataclasses, `CloudPricer`, overrides loader, `PricingDependencyError` |
| `method/spawners/codex_cli_shim.py` | normalize usage to `cloud_pricing_usage_v1` |
| `atp/reporters/benchmark_reporter.py` | stamp top-level `usage_contract` in payload |
| `method/price_overrides.toml` | open-tail prices + provenance + `local` set |
| `method/price_reports.py` | view: load → resolve model → price → reliability → CLI + sidecar |
| `tests/unit/cost/test_cloud_pricer_contract.py` | opt-in contract test vs real litellm |
| `tests/unit/cost/test_cloud_pricer.py` | monkeypatched pricer units |
| `tests/unit/method/test_price_reports.py` | view units over fixtures |
| `tests/fixtures/pricing/` | report fixtures (measured / no-usage / local) |

---

## Task 1: `[pricing]` extra + LiteLLM API contract test (the verify-first gate)

Pins the third-party API before any pricer logic depends on it. The contract test is the
source of truth for the `cost_per_token` signature and the inclusive/exclusive semantics of
`prompt_tokens` vs the cache classes.

**Files:**
- Modify: `packages/atp-core/pyproject.toml` (add optional-dependency group)
- Modify: `pyproject.toml` (root, add proxy extra)
- Create: `tests/unit/cost/test_cloud_pricer_contract.py`

**Interfaces:**
- Produces: the installed `litellm` with `cost_per_token(...)` accepting
  `prompt_tokens`, `completion_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens` and returning `(prompt_cost: float, completion_cost: float)`.
  The test records, in an assertion + comment, whether `prompt_tokens` must be passed
  **inclusive** of cache tokens (the assumption Task 3 encodes).

- [ ] **Step 1: Add the extra to atp-core**

In `packages/atp-core/pyproject.toml`, under `[project.optional-dependencies]` (create the
table if absent):

```toml
[project.optional-dependencies]
pricing = ["litellm>=1.81"]
```

- [ ] **Step 2: Proxy the extra from the root package**

In root `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
pricing = ["atp-core[pricing]"]
```

- [ ] **Step 3: Install and confirm import**

Run: `uv sync --extra pricing`
Then: `uv run python -c "import litellm; print(litellm.__version__)"`
Expected: prints a version `>= 1.81`.

- [ ] **Step 4: Write the contract test**

```python
"""Contract test: pins the real litellm.cost_per_token cache API.

Monkeypatched unit tests guard our code but not our ASSUMPTIONS about litellm.
This test runs against the installed library and fails loudly if the signature,
return shape, or cache semantics drift. Marked slow/opt-in: skipped when the
[pricing] extra is not installed.
"""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")


@pytest.mark.slow
def test_cost_per_token_accepts_cache_kwargs_and_returns_pair() -> None:
    # A known Anthropic model with published cache tariffs.
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="claude-3-5-sonnet-20240620",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
    )
    assert isinstance(prompt_cost, float)
    assert isinstance(completion_cost, float)
    # completion is billed at the full output rate.
    assert completion_cost > 0


@pytest.mark.slow
def test_cache_read_is_cheaper_than_full_input() -> None:
    """Pins the semantic we depend on: cache_read is discounted, and prompt_tokens
    is passed INCLUSIVE of the cache classes (litellm subtracts internally).
    If this fails, Task 3's argument wiring must flip to exclusive."""
    full, _ = litellm.cost_per_token(
        model="claude-3-5-sonnet-20240620",
        prompt_tokens=1000,
        completion_tokens=0,
        cache_read_input_tokens=0,
    )
    with_cache, _ = litellm.cost_per_token(
        model="claude-3-5-sonnet-20240620",
        prompt_tokens=1000,          # inclusive: 900 uncached + 100 cache_read
        completion_tokens=0,
        cache_read_input_tokens=900,
    )
    # Same total prompt tokens, but 900 read from cache => strictly cheaper.
    assert with_cache < full
```

- [ ] **Step 5: Run the contract test**

Run: `uv run pytest tests/unit/cost/test_cloud_pricer_contract.py -v -m slow`
Expected: PASS. If `test_cache_read_is_cheaper_than_full_input` FAILS, litellm expects
`prompt_tokens` **exclusive** of cache — record that and adjust Task 3 Step 3's
`_total_prompt_tokens` accordingly (do not proceed on a wrong assumption).

- [ ] **Step 6: Commit**

```bash
git add packages/atp-core/pyproject.toml pyproject.toml tests/unit/cost/test_cloud_pricer_contract.py
git commit -m "feat(pricing): [pricing] extra + litellm cost_per_token contract test"
```

---

## Task 2: Core pricer data model + measured pricing (monkeypatched)

**Files:**
- Create: `packages/atp-core/atp/cost/cloud_pricer.py`
- Create: `tests/unit/cost/test_cloud_pricer.py`

**Interfaces:**
- Produces:
  - `PerClassUsage(input_tokens: int, output_tokens: int, cache_creation_tokens: int,
    cache_read_tokens: int, usage_source: str | None)`
  - `CasePrice(usd: Decimal | None, usage_source: str | None, price_unknown: bool,
    cache_price_unknown: bool, cost_unknown: bool, pricing_scope: str, price_map_version: str)`
  - `PricingDependencyError(RuntimeError)`
  - `CloudPricer(overrides: PriceOverrides | None = None)` with
    `price_case(usage: PerClassUsage, model: str, *, is_local: bool = False) -> CasePrice`
  - `PRICING_INSTALL_HINT: str`, `USAGE_CONTRACT = "cloud_pricing_usage_v1"`
- Consumes: `PriceOverrides` (full definition lands in Task 3; Task 2 uses `overrides=None`).

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from atp.cost.cloud_pricer import (
    CloudPricer,
    PerClassUsage,
    PricingDependencyError,
)


class _FakeLitellm:
    """Stand-in for litellm: prices input at $1/token, output at $2/token,
    cache_read at $0.1/token; unknown models return (0.0, 0.0)."""

    __version__ = "fake-1"

    def cost_per_token(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> tuple[float, float]:
        if model == "unknown-model":
            return 0.0, 0.0
        uncached = prompt_tokens - cache_read_input_tokens - cache_creation_input_tokens
        prompt_cost = uncached * 1.0 + cache_read_input_tokens * 0.1 + (
            cache_creation_input_tokens * 1.25
        )
        return prompt_cost, completion_tokens * 2.0


def test_measured_prices_with_cache_split(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLitellm()
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: fake)
    pricer = CloudPricer()
    usage = PerClassUsage(
        input_tokens=100,          # uncached billable
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=900,
        usage_source="measured",
    )
    price = pricer.price_case(usage, model="claude-x")
    # prompt_tokens passed inclusive = 100 + 900 = 1000; cost = 100*1 + 900*0.1 = 190
    # + output 10*2 = 20 => 210
    assert price.usd == Decimal("210.0")
    assert price.usage_source == "measured"
    assert price.price_unknown is False
    assert price.cost_unknown is False
    assert price.pricing_scope == "cloud"
    assert price.price_map_version.startswith("litellm-fake-1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cost/test_cloud_pricer.py::test_measured_prices_with_cache_split -v`
Expected: FAIL — `ModuleNotFoundError: atp.cost.cloud_pricer`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Cache-aware cloud-$ pricer over stored per-class token usage (ADR-ECO-003d).

Pure and path-agnostic: given a normalized PerClassUsage and a model string, it
returns a derived cloud-$ using litellm as a *pricer* (never a re-counter). Only
the cloud class is priced; local models are out of scope (003c D4).

Normalized token contract `cloud_pricing_usage_v1`: input_tokens is billable
UNCACHED prompt; cache_* are additive and mutually exclusive from input_tokens.
litellm.cost_per_token expects prompt_tokens INCLUSIVE of cache classes (pinned
by tests/unit/cost/test_cloud_pricer_contract.py), so we reconstruct the total.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

USAGE_CONTRACT = "cloud_pricing_usage_v1"
PRICING_INSTALL_HINT = (
    "cloud-$ pricing needs litellm — external: 'pip install atp-platform[pricing]', "
    "in-repo dev: 'uv sync --extra pricing'"
)


class PricingDependencyError(RuntimeError):
    """Raised when the pricer is used without the [pricing] extra installed."""


@dataclass(frozen=True)
class PerClassUsage:
    """Per-class token usage under the cloud_pricing_usage_v1 contract."""

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    usage_source: str | None


@dataclass(frozen=True)
class CasePrice:
    """Derived price for one case (or None when not priceable)."""

    usd: Decimal | None
    usage_source: str | None
    price_unknown: bool
    cache_price_unknown: bool
    cost_unknown: bool
    pricing_scope: str  # "cloud" | "local_excluded"
    price_map_version: str


class _LitellmLike(Protocol):
    __version__: str

    def cost_per_token(self, **kwargs: Any) -> tuple[float, float]: ...


def _import_litellm() -> _LitellmLike:
    try:
        import litellm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise PricingDependencyError(PRICING_INSTALL_HINT) from exc
    return litellm


class CloudPricer:
    """Prices normalized per-class usage into cloud-$ via litellm."""

    def __init__(self, overrides: "PriceOverrides | None" = None) -> None:
        self._overrides = overrides
        self._litellm = _import_litellm()
        self._overrides_sha = overrides.sha8 if overrides is not None else "none"
        if overrides is not None:
            overrides.register(self._litellm)

    @property
    def price_map_version(self) -> str:
        return f"litellm-{self._litellm.__version__}+overrides-{self._overrides_sha}"

    def _total_prompt_tokens(self, usage: PerClassUsage) -> int:
        # litellm expects prompt_tokens inclusive of cache classes (Task 1 pin).
        return usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens

    def price_case(
        self, usage: PerClassUsage, model: str, *, is_local: bool = False
    ) -> CasePrice:
        pmv = self.price_map_version
        if is_local:
            return CasePrice(
                usd=None, usage_source=usage.usage_source, price_unknown=False,
                cache_price_unknown=False, cost_unknown=True,
                pricing_scope="local_excluded", price_map_version=pmv,
            )
        if usage.usage_source != "measured":
            return CasePrice(
                usd=None, usage_source=usage.usage_source, price_unknown=False,
                cache_price_unknown=False, cost_unknown=True,
                pricing_scope="cloud", price_map_version=pmv,
            )
        prompt_cost, completion_cost = self._litellm.cost_per_token(
            model=model,
            prompt_tokens=self._total_prompt_tokens(usage),
            completion_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_tokens,
            cache_creation_input_tokens=usage.cache_creation_tokens,
        )
        total = prompt_cost + completion_cost
        billable = usage.input_tokens + usage.output_tokens + usage.cache_read_tokens
        if total == 0.0 and billable > 0:
            return CasePrice(
                usd=None, usage_source="measured", price_unknown=True,
                cache_price_unknown=False, cost_unknown=False,
                pricing_scope="cloud", price_map_version=pmv,
            )
        return CasePrice(
            usd=Decimal(str(total)), usage_source="measured", price_unknown=False,
            cache_price_unknown=False, cost_unknown=False,
            pricing_scope="cloud", price_map_version=pmv,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cost/test_cloud_pricer.py::test_measured_prices_with_cache_split -v`
Expected: PASS.

- [ ] **Step 5: Add the not-measured, local, and dependency-missing cases**

Append to `tests/unit/cost/test_cloud_pricer.py`:

```python
def test_not_measured_is_cost_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer()
    usage = PerClassUsage(0, 0, 0, 0, usage_source=None)
    price = pricer.price_case(usage, model="claude-x")
    assert price.usd is None
    assert price.cost_unknown is True
    assert price.price_unknown is False


def test_local_is_excluded_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer()
    usage = PerClassUsage(100, 10, 0, 0, usage_source="measured")
    price = pricer.price_case(usage, model="llama3.2:3b", is_local=True)
    assert price.usd is None
    assert price.cost_unknown is True
    assert price.pricing_scope == "local_excluded"


def test_silent_zero_is_price_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer()
    usage = PerClassUsage(100, 10, 0, 0, usage_source="measured")
    price = pricer.price_case(usage, model="unknown-model")
    assert price.usd is None
    assert price.price_unknown is True
    assert price.cost_unknown is False


def test_missing_litellm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise PricingDependencyError("nope")

    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", _boom)
    with pytest.raises(PricingDependencyError):
        CloudPricer()
```

- [ ] **Step 6: Run all pricer unit tests**

Run: `uv run pytest tests/unit/cost/test_cloud_pricer.py -v`
Expected: 4 PASS.

- [ ] **Step 7: Format, lint, type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean (a `PriceOverrides` forward-ref is fine; it lands in Task 3).

- [ ] **Step 8: Commit**

```bash
git add packages/atp-core/atp/cost/cloud_pricer.py tests/unit/cost/test_cloud_pricer.py
git commit -m "feat(pricing): CloudPricer measured/not-measured/local/silent-zero paths"
```

---

## Task 3: Price overrides loader (register_model, provenance, cache_price_unknown, local set)

**Files:**
- Modify: `packages/atp-core/atp/cost/cloud_pricer.py` (add `PriceOverrides`)
- Create: `method/price_overrides.toml`
- Modify: `tests/unit/cost/test_cloud_pricer.py` (override tests)

**Interfaces:**
- Produces: `PriceOverrides` with classmethod `from_toml(path: Path) -> PriceOverrides`,
  attributes `sha8: str`, `local_models: frozenset[str]`,
  method `register(litellm: _LitellmLike) -> None`,
  method `cache_pricing_known(model: str) -> bool`, `is_local(model: str) -> bool`.
- Consumes: `_LitellmLike` from Task 2.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from atp.cost.cloud_pricer import PriceOverrides


def test_overrides_register_and_flags(tmp_path: Path) -> None:
    toml = tmp_path / "price_overrides.toml"
    toml.write_text(
        """
        [models."mimo-7b"]
        input_cost_per_1m = 0.20
        output_cost_per_1m = 0.60
        cache_pricing = "unknown"
        litellm_provider = "openai"
        source = "https://example.com/mimo-pricing"
        effective_date = "2026-07-01"
        currency = "USD"
        unit = "per_1m_tokens"
        notes = "interim"

        [local]
        models = ["llama3.2:3b", "qwen2.5:7b"]
        """,
        encoding="utf-8",
    )
    ov = PriceOverrides.from_toml(toml)
    assert ov.is_local("llama3.2:3b") is True
    assert ov.is_local("mimo-7b") is False
    assert ov.cache_pricing_known("mimo-7b") is False
    assert len(ov.sha8) == 8

    registered: dict = {}
    ov.register(type("L", (), {"register_model": staticmethod(lambda d: registered.update(d))})())
    # per-1M 0.20 => per-token 2e-7
    assert registered["mimo-7b"]["input_cost_per_token"] == 0.20 / 1_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cost/test_cloud_pricer.py::test_overrides_register_and_flags -v`
Expected: FAIL — `ImportError: cannot import name 'PriceOverrides'`.

- [ ] **Step 3: Implement `PriceOverrides` in `cloud_pricer.py`**

Add these imports at the top and the class below `CasePrice`:

```python
import hashlib
import tomllib
from pathlib import Path


@dataclass(frozen=True)
class PriceOverrides:
    """Interim open-tail prices + provenance, folded into the catalog contour later."""

    sha8: str
    local_models: frozenset[str]
    _models: dict[str, dict[str, Any]]      # model -> litellm register_model params
    _cache_known: frozenset[str]            # models whose cache tariff is trustworthy

    @classmethod
    def from_toml(cls, path: Path) -> "PriceOverrides":
        raw_bytes = path.read_bytes()
        sha8 = hashlib.sha256(raw_bytes).hexdigest()[:8]
        data = tomllib.loads(raw_bytes.decode("utf-8"))
        models: dict[str, dict[str, Any]] = {}
        cache_known: set[str] = set()
        for name, entry in (data.get("models") or {}).items():
            # Provenance is required — refuse silently-sourced prices.
            for field in ("source", "effective_date", "currency", "unit"):
                if field not in entry:
                    raise ValueError(f"price override '{name}' missing '{field}'")
            params: dict[str, Any] = {
                "input_cost_per_token": entry["input_cost_per_1m"] / 1_000_000,
                "output_cost_per_token": entry["output_cost_per_1m"] / 1_000_000,
                "litellm_provider": entry.get("litellm_provider", "openai"),
                "mode": "chat",
            }
            if entry.get("cache_pricing") == "known":
                if "cache_read_cost_per_1m" in entry:
                    params["cache_read_input_token_cost"] = (
                        entry["cache_read_cost_per_1m"] / 1_000_000
                    )
                cache_known.add(name)
            models[name] = params
        local = frozenset((data.get("local") or {}).get("models", []))
        return cls(sha8, local, models, frozenset(cache_known))

    def register(self, litellm: "_LitellmLike") -> None:
        if self._models:
            litellm.register_model(self._models)  # type: ignore[attr-defined]

    def cache_pricing_known(self, model: str) -> bool:
        return model in self._cache_known

    def is_local(self, model: str) -> bool:
        return model in self.local_models
```

- [ ] **Step 4: Wire `cache_price_unknown` into `price_case`**

In `price_case`, after computing a successful (non-zero) price, replace the final return with:

```python
        cache_used = usage.cache_read_tokens + usage.cache_creation_tokens
        cache_unknown = cache_used > 0 and self._overrides is not None and (
            not self._overrides.cache_pricing_known(model)
            and model in getattr(self._overrides, "_models", {})
        )
        return CasePrice(
            usd=Decimal(str(total)), usage_source="measured", price_unknown=False,
            cache_price_unknown=cache_unknown, cost_unknown=False,
            pricing_scope="cloud", price_map_version=pmv,
        )
```

- [ ] **Step 5: Author the real overrides file**

Create `method/price_overrides.toml` (fill prices from cited sources during implementation;
one worked entry shown — replicate the shape for glm-5.1, qwen3.6):

```toml
# Interim open-tail model prices for the cloud-$ pricing view (ADR-ECO-003d D5).
# NOT a source of truth — folds into the 003b catalog/discovery contour when it lands.
# Every model entry MUST carry provenance (source/effective_date/currency/unit).

[models."mimo-7b"]
input_cost_per_1m = 0.00     # TODO-DURING-IMPL: real cited value
output_cost_per_1m = 0.00    # TODO-DURING-IMPL: real cited value
cache_pricing = "unknown"
litellm_provider = "openai"
source = "FILL-CITATION-URL"
effective_date = "2026-07-01"
currency = "USD"
unit = "per_1m_tokens"
notes = "interim open-tail; verify against provider price page before paid runs"

[local]
# Local-class models: reported with tokens but excluded from cloud-$ (003c D4).
models = ["llama3.2:1b", "llama3.2:3b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
```

> The `0.00` values are the ONLY placeholders permitted, and only because they are
> cited-data-to-fill, not logic. Fill them from the provider's price page during
> implementation; a `0.00` cloud price with nonzero tokens trips the silent-zero
> `price_unknown` guard, so an unfilled entry fails safe (flagged, not silently free).

- [ ] **Step 6: Run override tests**

Run: `uv run pytest tests/unit/cost/test_cloud_pricer.py -v`
Expected: all PASS.

- [ ] **Step 7: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add packages/atp-core/atp/cost/cloud_pricer.py method/price_overrides.toml tests/unit/cost/test_cloud_pricer.py
git commit -m "feat(pricing): price overrides loader with provenance, register_model, local set"
```

---

## Task 4: Normalize codex_cli usage to `cloud_pricing_usage_v1`

Codex emits `input_tokens` = full prompt (cached included) and `cached_input_tokens` as a
subset under a non-`Metrics` key that `_grade_case` drops. Normalize so the split reaches the
payload: `input_tokens := input − cached`, `cache_read_tokens := cached`. This is the
double-count / over-count fix at the shim edge.

**Files:**
- Modify: `method/spawners/codex_cli_shim.py:198-207` (the `metrics` dict)
- Create: `tests/unit/method/test_codex_usage_normalization.py`

**Interfaces:**
- Produces: codex `metrics` carrying `input_tokens` (uncached), `cache_read_tokens`
  (= cached), `cache_creation_tokens` (0), conforming to the contract. `total_tokens`
  unchanged (= full input + output).

- [ ] **Step 1: Write the failing test**

```python
"""codex_cli usage must be normalized to cloud_pricing_usage_v1 at the shim edge:
cached_input is a SUBSET of input; leaving it inside input over-counts billable
input, and adding it to cache_read as well would double-count."""

from __future__ import annotations

from method.spawners.codex_cli_shim import normalize_usage


def test_cached_split_out_of_input_no_double_count() -> None:
    # Codex raw: input=1000 (incl. 800 cached), output=200, cached=800.
    norm = normalize_usage(input_tokens=1000, output_tokens=200, cached_input=800)
    assert norm["input_tokens"] == 200          # uncached billable
    assert norm["cache_read_tokens"] == 800
    assert norm["cache_creation_tokens"] == 0
    # invariant: input + cache_read + cache_creation == full prompt input
    assert (
        norm["input_tokens"] + norm["cache_read_tokens"] + norm["cache_creation_tokens"]
        == 1000
    )


def test_no_cached_is_identity() -> None:
    norm = normalize_usage(input_tokens=500, output_tokens=50, cached_input=None)
    assert norm["input_tokens"] == 500
    assert norm["cache_read_tokens"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/method/test_codex_usage_normalization.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_usage'`.

- [ ] **Step 3: Add `normalize_usage` and use it in the response builder**

Add near the top of `method/spawners/codex_cli_shim.py`:

```python
def normalize_usage(
    *, input_tokens: int | None, output_tokens: int | None, cached_input: int | None
) -> dict[str, int | None]:
    """Map codex OpenAI-convention usage onto cloud_pricing_usage_v1.

    Codex `input_tokens` is the FULL prompt and `cached_input_tokens` a subset of
    it. The pricing contract wants input_tokens = uncached billable and cache_read
    as an additive, mutually-exclusive class. Subtract cached out of input.
    """
    cached = int(cached_input or 0)
    full_input = input_tokens if input_tokens is not None else None
    uncached = None if full_input is None else max(full_input - cached, 0)
    return {
        "input_tokens": uncached,
        "output_tokens": output_tokens,
        "cache_read_tokens": cached if full_input is not None else None,
        "cache_creation_tokens": 0 if full_input is not None else None,
    }
```

Then in the `metrics` dict (currently lines ~198-207) replace the per-class fields:

```python
            "metrics": {
                "total_tokens": total,
                **normalize_usage(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cached_input=cached_in,
                ),
                # reasoning_output is a subset breakdown; surfaced, not summed.
                "reasoning_output_tokens": reasoning_out,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/method/test_codex_usage_normalization.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Regression — the existing codex shim smoke still passes**

Run: `uv run pytest tests/ -v -k codex -m "not slow"`
Expected: PASS (no existing test asserted the old un-normalized `input_tokens`; if one does,
update it to the normalized contract and note it in the commit).

- [ ] **Step 6: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add method/spawners/codex_cli_shim.py tests/unit/method/test_codex_usage_normalization.py
git commit -m "fix(method): normalize codex_cli usage to cloud_pricing_usage_v1 (cache split)"
```

---

## Task 5: Stamp `usage_contract` in the report_benchmark payload

**Files:**
- Modify: `atp/reporters/benchmark_reporter.py` (payload dict, near the `total_tokens` line)
- Modify: `tests/` — the reporter's existing payload test (locate with grep in Step 1)

**Interfaces:**
- Produces: `report_benchmark` payload with top-level `"usage_contract":
  "cloud_pricing_usage_v1"`. Additive field (schema `Request.additionalProperties: true`) —
  no version bump.

- [ ] **Step 1: Find the reporter payload test**

Run: `grep -rln "total_tokens\|payload_version\|build_report_benchmark" tests/ | head`
Open the test that asserts payload shape; note its path as `<REPORTER_TEST>`.

- [ ] **Step 2: Write the failing assertion**

Add to `<REPORTER_TEST>` (in the test that builds a payload):

```python
from atp.cost.cloud_pricer import USAGE_CONTRACT

def test_payload_stamps_usage_contract(...):  # reuse the existing payload fixture
    payload = build_report_benchmark_payload(...)  # existing call
    assert payload["usage_contract"] == USAGE_CONTRACT
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest <REPORTER_TEST> -k usage_contract -v`
Expected: FAIL — `KeyError: 'usage_contract'`.

- [ ] **Step 4: Add the stamp**

In `atp/reporters/benchmark_reporter.py`, import at top:

```python
from atp.cost.cloud_pricer import USAGE_CONTRACT
```

In the `payload` dict literal, alongside `"total_tokens": ...`, add:

```python
        "usage_contract": USAGE_CONTRACT,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest <REPORTER_TEST> -k usage_contract -v`
Expected: PASS.

- [ ] **Step 6: Confirm the conformance/schema test is still green**

Run: `uv run pytest tests/ -k "report_benchmark or conformance" -m "not slow"`
Expected: PASS (additive field, no version bump).

- [ ] **Step 7: Format, lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add atp/reporters/benchmark_reporter.py <REPORTER_TEST>
git commit -m "feat(pricing): stamp usage_contract=cloud_pricing_usage_v1 in payload"
```

---

## Task 6: The view — `method/price_reports.py`

Reads saved reports, resolves model from `agent_id`, prices per case, aggregates per agent,
emits a numeric reliability block, ignores legacy `total_cost_usd`, flags missing
`usage_contract`, and writes a sidecar + prints a table.

**Files:**
- Create: `method/price_reports.py`
- Create: `tests/unit/method/test_price_reports.py`
- Create: `tests/fixtures/pricing/` (report fixtures)

**Interfaces:**
- Consumes: `CloudPricer`, `PerClassUsage`, `PriceOverrides`, `USAGE_CONTRACT` (Tasks 2-3).
- Produces:
  - `resolve_model(agent_id: str) -> str | None` (`"harness@model"` → `model`; bare → None)
  - `AgentCost` (agent_id, model, measured_usd: Decimal | None, reliability: dict[str, Any])
  - `derive_cost_view(reports: Iterable[dict], overrides_path: Path) -> list[AgentCost]`
  - `main(argv: list[str] | None = None) -> int` (CLI; writes `cost_view.json`)

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from method.price_reports import derive_cost_view, resolve_model


def test_resolve_model_from_agent_id() -> None:
    assert resolve_model("claude_code@claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert resolve_model("legacy_bare_id") is None


def _report(agent_id: str, cases: list[dict], contract: str | None) -> dict:
    d = {"agent_id": agent_id, "per_task": cases}
    if contract is not None:
        d["usage_contract"] = contract
    return d


def test_view_prices_measured_and_flags(tmp_path: Path) -> None:
    overrides = tmp_path / "ov.toml"
    overrides.write_text('[local]\nmodels = ["llama3.2:3b"]\n', encoding="utf-8")
    cloud = _report(
        "codex_cli@gpt-5.5",
        [{
            "input_tokens": 100, "output_tokens": 10,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "usage_source": "measured",
        }],
        contract="cloud_pricing_usage_v1",
    )
    local = _report(
        "ollama@llama3.2:3b",
        [{
            "input_tokens": 100, "output_tokens": 10,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "usage_source": "measured",
        }],
        contract="cloud_pricing_usage_v1",
    )
    view = {a.agent_id: a for a in derive_cost_view([cloud, local], overrides)}
    assert view["codex_cli@gpt-5.5"].measured_usd is not None
    assert view["codex_cli@gpt-5.5"].reliability["reliability_status"] == "ok"
    local_agent = view["ollama@llama3.2:3b"]
    assert local_agent.measured_usd is None
    assert local_agent.reliability["local_cases"] == 1


def test_missing_contract_is_flagged(tmp_path: Path) -> None:
    overrides = tmp_path / "ov.toml"
    overrides.write_text("", encoding="utf-8")
    legacy = _report(
        "codex_cli@gpt-5.5",
        [{"input_tokens": 100, "output_tokens": 10, "cache_creation_tokens": 0,
          "cache_read_tokens": 0, "usage_source": "measured"}],
        contract=None,
    )
    agent = derive_cost_view([legacy], overrides)[0]
    assert agent.reliability["reliability_status"] == "unreliable"
    assert agent.reliability["contract_missing"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/method/test_price_reports.py -v`
Expected: FAIL — `ModuleNotFoundError: method.price_reports`.

- [ ] **Step 3: Implement the view**

```python
"""Derive cloud-$ over saved report_benchmark payloads (ADR-ECO-003d, surface A).

Derived-not-stored: reads stored per-class usage and prices at report time, so a
price change re-derives without a re-sweep. Ignores the legacy total_cost_usd
field; surfaces derived_usd under its own name. Reports without the
cloud_pricing_usage_v1 stamp are flagged, never silently mixed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from atp.cost.cloud_pricer import (
    USAGE_CONTRACT,
    CloudPricer,
    PerClassUsage,
    PriceOverrides,
)


def resolve_model(agent_id: str) -> str | None:
    """`harness@model` -> `model`; a bare id (no '@') -> None (never guess)."""
    if "@" not in agent_id:
        return None
    return agent_id.split("@", 1)[1]


@dataclass(frozen=True)
class AgentCost:
    agent_id: str
    model: str | None
    measured_usd: Decimal | None
    reliability: dict[str, Any]


def _usage(case: dict) -> PerClassUsage:
    return PerClassUsage(
        input_tokens=int(case.get("input_tokens") or 0),
        output_tokens=int(case.get("output_tokens") or 0),
        cache_creation_tokens=int(case.get("cache_creation_tokens") or 0),
        cache_read_tokens=int(case.get("cache_read_tokens") or 0),
        usage_source=case.get("usage_source"),
    )


def _status(counts: dict[str, int], cloud_total: int, contract_missing: bool) -> str:
    if contract_missing:
        return "unreliable"
    if cloud_total == 0:
        return "ok"  # all-local: nothing to price, not a failure
    bad = (
        counts["cost_unknown_cases"]
        + counts["price_unknown_cases"]
        + counts["estimated_cases"]
    )
    if bad * 2 > cloud_total:
        return "unreliable"
    if bad > 0 or counts["cache_pricing_unknown_cases"] > 0:
        return "degraded"
    return "ok"


def _price_agent(report: dict, pricer: CloudPricer, overrides: PriceOverrides) -> AgentCost:
    agent_id = report["agent_id"]
    model = resolve_model(agent_id)
    contract_missing = report.get("usage_contract") != USAGE_CONTRACT
    cases = report.get("per_task") or []
    counts = {
        "total_cases": len(cases), "measured_cases": 0, "cost_unknown_cases": 0,
        "price_unknown_cases": 0, "cache_pricing_unknown_cases": 0,
        "estimated_cases": 0, "local_cases": 0,
    }
    total = Decimal("0")
    any_priced = False
    cloud_total = 0
    for case in cases:
        usage = _usage(case)
        is_local = model is not None and overrides.is_local(model)
        if is_local:
            counts["local_cases"] += 1
            continue
        cloud_total += 1
        if model is None:
            counts["price_unknown_cases"] += 1
            continue
        price = pricer.price_case(usage, model=model, is_local=False)
        if price.usage_source == "estimated":
            counts["estimated_cases"] += 1
        if price.price_unknown:
            counts["price_unknown_cases"] += 1
        if price.cache_price_unknown:
            counts["cache_pricing_unknown_cases"] += 1
        if price.cost_unknown:
            counts["cost_unknown_cases"] += 1
        if price.usd is not None:
            total += price.usd
            any_priced = True
            counts["measured_cases"] += 1
    reliability = {
        **counts,
        "contract_missing": contract_missing,
        "reliability_status": _status(counts, cloud_total, contract_missing),
    }
    return AgentCost(agent_id, model, total if any_priced else None, reliability)


def derive_cost_view(
    reports: Iterable[dict], overrides_path: Path
) -> list[AgentCost]:
    overrides = (
        PriceOverrides.from_toml(overrides_path)
        if overrides_path.exists() and overrides_path.read_text(encoding="utf-8").strip()
        else PriceOverrides("none", frozenset(), {}, frozenset())
    )
    pricer = CloudPricer(overrides=overrides)
    return [_price_agent(r, pricer, overrides) for r in reports]


def _load_reports(reports_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(reports_dir.glob("report_benchmark_*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive cloud-$ over report payloads.")
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument(
        "--overrides", type=Path, default=Path(__file__).parent / "price_overrides.toml"
    )
    args = parser.parse_args(argv)
    view = derive_cost_view(_load_reports(args.reports_dir), args.overrides)
    for agent in view:
        usd = "—" if agent.measured_usd is None else f"${agent.measured_usd:.4f}"
        print(
            f"{agent.agent_id:40s}  derived_usd={usd:>12s}  "
            f"{agent.reliability['reliability_status']}"
        )
    sidecar = args.reports_dir / "cost_view.json"
    sidecar.write_text(
        json.dumps(
            [
                {
                    "agent_id": a.agent_id,
                    "model": a.model,
                    "derived_usd": None if a.measured_usd is None else str(a.measured_usd),
                    "usage_contract": USAGE_CONTRACT,
                    "price_map_version": CloudPricer(
                        overrides=PriceOverrides("none", frozenset(), {}, frozenset())
                    ).price_map_version,
                    "reliability": a.reliability,
                }
                for a in view
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> Note: `CloudPricer` requires litellm at construction. `derive_cost_view` builds one; in
> tests without the `[pricing]` extra, monkeypatch `_import_litellm` as in Task 2, or mark
> these view tests to `importorskip("litellm")`. Prefer the monkeypatch so the view logic is
> tested without the heavy dep — add the same `_FakeLitellm` monkeypatch fixture at the top
> of `test_price_reports.py`.

- [ ] **Step 4: Add the litellm monkeypatch to the view test**

At the top of `tests/unit/method/test_price_reports.py`, add an autouse fixture reusing the
Task 2 fake (import it or redefine a minimal one) so `CloudPricer()` constructs without the
extra:

```python
import pytest

@pytest.fixture(autouse=True)
def _fake_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    class _F:
        __version__ = "fake-1"
        def cost_per_token(self, *, model, prompt_tokens, completion_tokens,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0):
            uncached = prompt_tokens - cache_read_input_tokens - cache_creation_input_tokens
            return uncached * 1.0 + cache_read_input_tokens * 0.1, completion_tokens * 2.0
        def register_model(self, d): ...
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _F())
```

- [ ] **Step 5: Run the view tests**

Run: `uv run pytest tests/unit/method/test_price_reports.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Create report fixtures for an end-to-end CLI check**

Create `tests/fixtures/pricing/report_benchmark_codex.json` (measured, contract present) and
`tests/fixtures/pricing/report_benchmark_ollama.json` (local) matching the shapes used in the
unit test. Then add:

```python
def test_main_writes_sidecar(tmp_path: Path, monkeypatch, capsys) -> None:
    # copy the two fixtures into tmp_path, point overrides at an empty file
    ...
    rc = main([str(tmp_path), "--overrides", str(tmp_path / "ov.toml")])
    assert rc == 0
    assert (tmp_path / "cost_view.json").exists()
```

- [ ] **Step 7: Run, format, lint, type-check, commit**

```bash
uv run pytest tests/unit/method/test_price_reports.py -v
uv run ruff format . && uv run ruff check . && uv run pyrefly check
git add method/price_reports.py tests/unit/method/test_price_reports.py tests/fixtures/pricing/
git commit -m "feat(pricing): derive_cost_view + price_reports CLI over saved payloads"
```

---

## Task 7: Re-sweep under the normalized contract + docs (operational, user-gated)

The prior tasks make the view correct; this task produces authoritative data and wires docs.
**The re-sweep runs live agents and costs money — it MUST be run by the user, not a subagent.**

**Files:**
- Modify: `CLAUDE.md` (component 25 note + `[pricing]` extra mention)
- Modify: `TODO.md` (mark pricing-view done, link the spec/plan)

- [ ] **Step 1: Full-suite green + type-check**

Run: `uv run pytest tests/ -m "not slow" -q && uv run pyrefly check`
Expected: PASS. This is the go/no-go for the code half.

- [ ] **Step 2: Dry-run the view over EXISTING (pre-contract) reports**

Run: `uv run python method/price_reports.py _cowork_output/r07-pipecheck/` (or a dir with
recent `report_benchmark_*.json`).
Expected: every agent flagged `unreliable` with `contract_missing: true` — proving the
lineage guard works and old reports are not silently priced. Confirm before re-sweeping.

- [ ] **Step 3: HAND OFF the paid re-sweep to the user**

Do not run this yourself. Tell the user to run, foreground and chunked (bg sweeps get killed
— see memory), e.g.:

```
uv run python method/run_pipe_check.py --agents codex_cli@gpt-5.5,claude_code@claude-sonnet-4-6 --out-dir _bench_output/pricing-resweep
```

Then re-derive (free, no re-sweep needed thereafter):

```
uv run python method/price_reports.py _bench_output/pricing-resweep
```

Expected after re-sweep: codex prices **below** its pre-normalization value (cache now
discounted), reliability `ok`, sidecar `cost_view.json` written.

- [ ] **Step 4: Update docs**

In `CLAUDE.md` component 25, append one sentence: pricing-view derives cloud-`$` from stored
per-class usage via a cache-aware litellm pricer (`atp/cost/cloud_pricer.py`,
`method/price_reports.py`), gated behind the `[pricing]` extra; `usage_contract =
"cloud_pricing_usage_v1"`. In `TODO.md`, tick pricing-view and link this plan + the spec.

- [ ] **Step 5: Commit docs**

```bash
git add CLAUDE.md TODO.md
git commit -m "docs(pricing): pricing-view usage + resweep note (ADR-003d)"
```

- [ ] **Step 6: Open the PR**

```bash
git push -u origin feat/pricing-view
gh pr create --title "feat(pricing): cloud-\$ pricing view over per-class usage (ADR-003d)" --body "..."
```

---

## Self-Review

**Spec coverage:**
- Core pricer, cache-split, silent-zero, litellm-missing → Task 2 + 3. ✓
- Token-class invariant + codex normalization + no-double-count regression → Task 4. ✓
- `usage_contract` named/placed (payload + sidecar) → Task 5 (payload), Task 6 (sidecar). ✓
- Open-tail `register_model` + provenance + `cache_price_unknown` → Task 3. ✓
- Model resolution from `agent_id`, bare → price_unknown → Task 6. ✓
- Legacy `total_cost_usd` ignored, `derived_usd` distinct → Task 6. ✓
- Formal numeric reliability block + thresholds + local `pricing_scope` → Task 2/6. ✓
- Contract test vs real litellm → Task 1. ✓
- `[pricing]` extra in atp-core, root proxy, dual install hint → Task 1 + 2. ✓
- Data lineage / re-sweep, missing-contract flag → Task 6 (flag) + Task 7 (re-sweep). ✓
- Estimated-fallback deferred (schema-only) → honored; `estimated_cases` counter present,
  no estimation computed. ✓

**Placeholder scan:** the only literals left blank are the cited price values in
`method/price_overrides.toml` (data-to-fill from provider pages, fail-safe via silent-zero),
and `<REPORTER_TEST>` (a grep-located path) / the `...` fixture-copy body in Task 6 Step 6 —
all mechanical, not logic. No "add error handling"/"TBD"/"similar to" placeholders in code.

**Type consistency:** `PerClassUsage`, `CasePrice`, `PriceOverrides`, `CloudPricer.price_case`,
`resolve_model`, `AgentCost`, `derive_cost_view`, `USAGE_CONTRACT` used identically across
Tasks 2-6. `price_map_version` format consistent. `reliability` keys match between `_status`,
`_price_agent`, and the tests.
