"""Unit tests for the cache-aware CloudPricer (monkeypatched litellm)."""

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
        prompt_cost = (
            uncached * 1.0
            + cache_read_input_tokens * 0.1
            + (cache_creation_input_tokens * 1.25)
        )
        return prompt_cost, completion_tokens * 2.0


def test_measured_prices_with_cache_split(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLitellm()
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: fake)
    pricer = CloudPricer()
    usage = PerClassUsage(
        input_tokens=100,  # uncached billable
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
