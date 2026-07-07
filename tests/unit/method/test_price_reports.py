"""Tests for method/price_reports.py — the pricing-view CLI/library surface.

Derived-not-stored: prices re-derive from stored report_benchmark payloads at
report time (ADR-ECO-003d). These tests never touch real litellm — an autouse
fixture swaps in a small fake pricer backend (Task 2 pattern) so the view
logic is exercised without the [pricing] extra.
"""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from method.price_reports import derive_cost_view, main, resolve_model

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "pricing"


@pytest.fixture(autouse=True)
def _fake_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    class _F:
        __version__ = "fake-1"

        def cost_per_token(
            self,
            *,
            model,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ):
            uncached = (
                prompt_tokens - cache_read_input_tokens - cache_creation_input_tokens
            )
            return (
                uncached * 1.0 + cache_read_input_tokens * 0.1,
                completion_tokens * 2.0,
            )

        def register_model(self, d): ...

    monkeypatch.setattr("atp.cost.cloud_pricer._import_litellm", lambda: _F())


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
        [
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "usage_source": "measured",
            }
        ],
        contract="cloud_pricing_usage_v1",
    )
    local = _report(
        "ollama@llama3.2:3b",
        [
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "usage_source": "measured",
            }
        ],
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
        [
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "usage_source": "measured",
            }
        ],
        contract=None,
    )
    agent = derive_cost_view([legacy], overrides)[0]
    assert agent.reliability["reliability_status"] == "unreliable"
    assert agent.reliability["contract_missing"] is True


def test_estimated_cases_not_double_counted_as_cost_unknown(tmp_path: Path) -> None:
    overrides = tmp_path / "ov.toml"
    overrides.write_text("", encoding="utf-8")
    measured_case = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "usage_source": "measured",
    }
    estimated_case = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "usage_source": "estimated",
    }
    cases = [estimated_case] * 3 + [measured_case] * 7
    report = _report("codex_cli@gpt-5.5", cases, contract="cloud_pricing_usage_v1")
    agent = derive_cost_view([report], overrides)[0]
    reliability = agent.reliability
    assert reliability["estimated_cases"] == 3
    assert reliability["cost_unknown_cases"] == 0
    assert reliability["measured_cases"] == 7
    assert reliability["reliability_status"] == "degraded"


def test_main_writes_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Copy the two committed fixtures (measured cloud + local) into a scratch
    # reports dir alongside an empty overrides file, then run the CLI.
    for name in (
        "report_benchmark_codex.json",
        "report_benchmark_ollama.json",
    ):
        shutil.copy(FIXTURES / name, tmp_path / name)
    overrides = tmp_path / "ov.toml"
    overrides.write_text('[local]\nmodels = ["llama3.2:3b"]\n', encoding="utf-8")

    rc = main([str(tmp_path), "--overrides", str(overrides)])

    assert rc == 0
    sidecar = tmp_path / "cost_view.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    by_agent = {d["agent_id"]: d for d in data}
    assert by_agent["codex_cli@gpt-5.5"]["derived_usd"] is not None
    assert Decimal(by_agent["codex_cli@gpt-5.5"]["derived_usd"]) > 0
    assert by_agent["codex_cli@gpt-5.5"]["reliability"]["reliability_status"] == "ok"
    assert by_agent["ollama@llama3.2:3b"]["derived_usd"] is None
    assert by_agent["ollama@llama3.2:3b"]["reliability"]["local_cases"] == 1

    out = capsys.readouterr().out
    assert "codex_cli@gpt-5.5" in out
    assert "ollama@llama3.2:3b" in out
