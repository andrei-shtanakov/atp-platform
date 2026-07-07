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

import hashlib
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

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
    """Import litellm lazily, raising a friendly error when it is absent."""
    try:
        import litellm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise PricingDependencyError(PRICING_INSTALL_HINT) from exc
    return cast(_LitellmLike, litellm)


@dataclass(frozen=True)
class PriceOverrides:
    """Interim open-tail prices + provenance, folded into the catalog contour later."""

    sha8: str
    local_models: frozenset[str]
    _models: dict[str, dict[str, Any]]  # model -> litellm register_model params
    _cache_known: frozenset[str]  # models whose cache tariff is trustworthy

    @classmethod
    def from_toml(cls, path: Path) -> PriceOverrides:
        """Load price overrides + provenance from a TOML file.

        Args:
            path: Path to the overrides TOML file (see `method/price_overrides.toml`).

        Returns:
            A `PriceOverrides` instance ready to register against litellm.

        Raises:
            ValueError: If a model entry is missing required provenance
                fields (`source`, `effective_date`, `currency`, `unit`).
        """
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

    def register(self, litellm: _LitellmLike) -> None:
        """Register override model prices against the given litellm module."""
        if self._models:
            litellm.register_model(self._models)  # type: ignore[attr-defined]

    def cache_pricing_known(self, model: str) -> bool:
        """Return True when the model's cache tariff is a trusted override."""
        return model in self._cache_known

    def is_local(self, model: str) -> bool:
        """Return True when the model is in the local (non-cloud-priced) set."""
        return model in self.local_models


class CloudPricer:
    """Prices normalized per-class usage into cloud-$ via litellm."""

    def __init__(
        self,
        overrides: PriceOverrides | None = None,
    ) -> None:
        """Create a pricer, optionally applying price overrides.

        Args:
            overrides: Optional price override registry (Task 3). When
                provided, it is registered against the imported litellm
                module and its sha8 is folded into ``price_map_version``.
        """
        self._overrides = overrides
        self._litellm = _import_litellm()
        self._overrides_sha = overrides.sha8 if overrides is not None else "none"
        if overrides is not None:
            overrides.register(self._litellm)

    @property
    def price_map_version(self) -> str:
        """Return a version string identifying the litellm + overrides state."""
        return f"litellm-{self._litellm.__version__}+overrides-{self._overrides_sha}"

    def _total_prompt_tokens(self, usage: PerClassUsage) -> int:
        # litellm expects prompt_tokens inclusive of cache classes (Task 1 pin).
        return (
            usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
        )

    def price_case(
        self, usage: PerClassUsage, model: str, *, is_local: bool = False
    ) -> CasePrice:
        """Price one case's per-class usage into cloud-$.

        Args:
            usage: Normalized per-class token usage.
            model: The model name passed through to litellm's price map.
            is_local: When True, the case is excluded from cloud pricing
                (local models have no cloud-$ cost) rather than reported as
                unpriceable.

        Returns:
            A CasePrice describing the derived usd (or why it is unknown).
        """
        pmv = self.price_map_version
        if is_local:
            return CasePrice(
                usd=None,
                usage_source=usage.usage_source,
                price_unknown=False,
                cache_price_unknown=False,
                cost_unknown=True,
                pricing_scope="local_excluded",
                price_map_version=pmv,
            )
        if usage.usage_source != "measured":
            return CasePrice(
                usd=None,
                usage_source=usage.usage_source,
                price_unknown=False,
                cache_price_unknown=False,
                cost_unknown=True,
                pricing_scope="cloud",
                price_map_version=pmv,
            )
        prompt_cost, completion_cost = self._litellm.cost_per_token(
            model=model,
            prompt_tokens=self._total_prompt_tokens(usage),
            completion_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_tokens,
            cache_creation_input_tokens=usage.cache_creation_tokens,
        )
        total = prompt_cost + completion_cost
        billable = (
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_read_tokens
            + usage.cache_creation_tokens
        )
        if total == 0.0 and billable > 0:
            return CasePrice(
                usd=None,
                usage_source="measured",
                price_unknown=True,
                cache_price_unknown=False,
                cost_unknown=False,
                pricing_scope="cloud",
                price_map_version=pmv,
            )
        cache_used = usage.cache_read_tokens + usage.cache_creation_tokens
        cache_unknown = (
            cache_used > 0
            and self._overrides is not None
            and (
                not self._overrides.cache_pricing_known(model)
                and model in getattr(self._overrides, "_models", {})
            )
        )
        return CasePrice(
            usd=Decimal(str(total)),
            usage_source="measured",
            price_unknown=False,
            cache_price_unknown=cache_unknown,
            cost_unknown=False,
            pricing_scope="cloud",
            price_map_version=pmv,
        )
