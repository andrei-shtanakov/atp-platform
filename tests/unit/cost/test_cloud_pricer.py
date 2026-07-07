"""Unit tests for the cache-aware CloudPricer (monkeypatched litellm)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from atp.cost.cloud_pricer import (
    CloudPricer,
    PerClassUsage,
    PriceOverrides,
    PricingDependencyError,
)


class _FakeLitellm:
    """Stand-in for litellm: prices input at $1/token, output at $2/token,
    cache_read at $0.1/token; unknown models return (0.0, 0.0)."""

    __version__ = "fake-1"

    def register_model(self, models: dict) -> None:
        """No-op: cost_per_token below uses a fixed formula, not a price map."""

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


class _RaisingVersionLitellm:
    """Stand-in for real litellm (>=1.91): __version__ access raises
    AttributeError via a lazy module-level __getattr__, instead of returning
    a value like the _FakeLitellm class-attribute fake above."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def cost_per_token(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> tuple[float, float]:
        return 1.0, 1.0


def test_price_map_version_survives_raising_version_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: real litellm's lazy __getattr__ raises AttributeError for
    # __version__ instead of returning a value; price_map_version must not
    # crash and must fall back to the installed-package version (or
    # "unknown"), not the fake-1 class-attribute value from _FakeLitellm.
    monkeypatch.setattr(
        "atp.cost.cloud_pricer._import_litellm", lambda: _RaisingVersionLitellm()
    )
    pricer = CloudPricer()
    pmv = pricer.price_map_version
    assert pmv.startswith("litellm-")
    version_part = pmv.removeprefix("litellm-").split("+overrides-", 1)[0]
    assert version_part != ""
    assert "+overrides-" in pmv


def test_missing_litellm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise PricingDependencyError("nope")

    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", _boom)
    with pytest.raises(PricingDependencyError):
        CloudPricer()


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
    ov.register(
        type(
            "L", (), {"register_model": staticmethod(lambda d: registered.update(d))}
        )()
    )
    # per-1M 0.20 => per-token 2e-7
    assert registered["mimo-7b"]["input_cost_per_token"] == 0.20 / 1_000_000


def test_overrides_missing_provenance_raises(tmp_path: Path) -> None:
    toml = tmp_path / "price_overrides.toml"
    toml.write_text(
        """
        [models."bad-model"]
        input_cost_per_1m = 0.20
        output_cost_per_1m = 0.60
        cache_pricing = "unknown"
        litellm_provider = "openai"
        effective_date = "2026-07-01"
        currency = "USD"
        unit = "per_1m_tokens"
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source"):
        PriceOverrides.from_toml(toml)


def test_cache_price_unknown_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        """,
        encoding="utf-8",
    )
    ov = PriceOverrides.from_toml(toml)
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer(overrides=ov)
    usage = PerClassUsage(
        input_tokens=100,
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=50,
        usage_source="measured",
    )
    price = pricer.price_case(usage, model="mimo-7b")
    assert price.cache_price_unknown is True
    assert price.price_unknown is False


def test_cache_price_known_not_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    toml = tmp_path / "price_overrides.toml"
    toml.write_text(
        """
        [models."known-cache-model"]
        input_cost_per_1m = 0.20
        output_cost_per_1m = 0.60
        cache_read_cost_per_1m = 0.02
        cache_pricing = "known"
        litellm_provider = "openai"
        source = "https://example.com/known-cache-pricing"
        effective_date = "2026-07-01"
        currency = "USD"
        unit = "per_1m_tokens"
        notes = "cache tariff confirmed"
        """,
        encoding="utf-8",
    )
    ov = PriceOverrides.from_toml(toml)
    assert ov.cache_pricing_known("known-cache-model") is True
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer(overrides=ov)
    usage = PerClassUsage(
        input_tokens=100,
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=50,
        usage_source="measured",
    )
    price = pricer.price_case(usage, model="known-cache-model")
    assert price.cache_price_unknown is False


def test_cache_price_unknown_false_when_no_cache_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        """,
        encoding="utf-8",
    )
    ov = PriceOverrides.from_toml(toml)
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer(overrides=ov)
    usage = PerClassUsage(
        input_tokens=100,
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        usage_source="measured",
    )
    price = pricer.price_case(usage, model="mimo-7b")
    assert price.cache_price_unknown is False


def test_cache_price_unknown_false_for_model_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A model not present in overrides at all must not be flagged
    # cache_price_unknown — that field is scoped to override entries only.
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer()
    usage = PerClassUsage(
        input_tokens=100,
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=900,
        usage_source="measured",
    )
    price = pricer.price_case(usage, model="claude-x")
    assert price.cache_price_unknown is False


def test_cache_creation_only_trips_silent_zero_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: billable must include cache_creation_tokens too, otherwise a
    # cache-creation-only usage against an unknown model silently reports $0.
    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _FakeLitellm())
    pricer = CloudPricer()
    usage = PerClassUsage(
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=100,
        cache_read_tokens=0,
        usage_source="measured",
    )
    price = pricer.price_case(usage, model="unknown-model")
    assert price.usd is None
    assert price.price_unknown is True
