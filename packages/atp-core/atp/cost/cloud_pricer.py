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


class CloudPricer:
    """Prices normalized per-class usage into cloud-$ via litellm."""

    def __init__(
        self,
        overrides: PriceOverrides | None = None,  # noqa: F821 # pyrefly: ignore[unknown-name]
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
        billable = usage.input_tokens + usage.output_tokens + usage.cache_read_tokens
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
        return CasePrice(
            usd=Decimal(str(total)),
            usage_source="measured",
            price_unknown=False,
            cache_price_unknown=False,
            cost_unknown=False,
            pricing_scope="cloud",
            price_map_version=pmv,
        )
