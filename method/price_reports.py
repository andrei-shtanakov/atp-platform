"""Derive cloud-$ over saved report_benchmark payloads (ADR-ECO-003d, surface A).

Derived-not-stored: reads stored per-class usage and prices at report time, so a
price change re-derives without a re-sweep. Ignores the legacy total_cost_usd
field; surfaces derived_usd under its own name. Reports without the
cloud_pricing_usage_v1 stamp are flagged, never silently mixed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
    """Aggregated derived cloud-$ + reliability signal for one agent."""

    agent_id: str
    model: str | None
    measured_usd: Decimal | None
    reliability: dict[str, Any]


def _usage(case: dict[str, Any]) -> PerClassUsage:
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


def _price_agent(
    report: dict[str, Any], pricer: CloudPricer, overrides: PriceOverrides
) -> AgentCost:
    agent_id = report["agent_id"]
    model = resolve_model(agent_id)
    contract_missing = report.get("usage_contract") != USAGE_CONTRACT
    cases = report.get("per_task") or []
    counts = {
        "total_cases": len(cases),
        "measured_cases": 0,
        "cost_unknown_cases": 0,
        "price_unknown_cases": 0,
        "cache_pricing_unknown_cases": 0,
        "estimated_cases": 0,
        "local_cases": 0,
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
        if price.usd is not None:
            counts["measured_cases"] += 1
            if price.cache_price_unknown:
                counts["cache_pricing_unknown_cases"] += 1
            total += price.usd
            any_priced = True
        else:
            # Unpriced — exactly one reason, mutually exclusive so `bad`
            # (in `_status`) never double-counts a single case.
            if price.price_unknown:
                counts["price_unknown_cases"] += 1
            elif price.usage_source == "estimated":
                counts["estimated_cases"] += 1
            else:
                counts["cost_unknown_cases"] += 1
    reliability = {
        **counts,
        "contract_missing": contract_missing,
        "reliability_status": _status(counts, cloud_total, contract_missing),
    }
    # Lineage guard: a report missing the usage_contract stamp has
    # un-normalized per-class usage, so cache-split math cannot be trusted —
    # withhold the headline price even if some cases nominally priced.
    measured_usd = None if contract_missing else (total if any_priced else None)
    return AgentCost(agent_id, model, measured_usd, reliability)


def _load_overrides(overrides_path: Path) -> PriceOverrides:
    if overrides_path.exists() and overrides_path.read_text(encoding="utf-8").strip():
        return PriceOverrides.from_toml(overrides_path)
    return PriceOverrides("none", frozenset(), {}, frozenset())


def derive_cost_view(
    reports: Iterable[dict[str, Any]], overrides_path: Path
) -> list[AgentCost]:
    """Derive per-agent cloud-$ + reliability over saved report payloads.

    Args:
        reports: Parsed `report_benchmark_*.json` payloads.
        overrides_path: Path to a price-overrides TOML file (Task 3); an
            empty or missing file falls back to no overrides.

    Returns:
        One `AgentCost` per report, in input order.
    """
    overrides = _load_overrides(overrides_path)
    # litellm prints "Provider List: ..." noise to stdout — both when
    # CloudPricer registers open-tail override models (mimo/glm/qwen, not in
    # its built-in cost map) and when it prices cases for them — so both the
    # construction and the pricing loop must run under the redirect. Swallow
    # only that stdout chatter here, not the CLI's own table printed later in
    # main().
    with contextlib.redirect_stdout(io.StringIO()):
        pricer = CloudPricer(overrides=overrides)
        return [_price_agent(r, pricer, overrides) for r in reports]


def _load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("report_benchmark_*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: derive cost view over a reports directory, write sidecar."""
    parser = argparse.ArgumentParser(
        description="Derive cloud-$ over report_benchmark payloads."
    )
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path(__file__).parent / "price_overrides.toml",
    )
    args = parser.parse_args(argv)
    reports_dir: Path = args.reports_dir
    overrides_path: Path = args.overrides

    view = derive_cost_view(_load_reports(reports_dir), overrides_path)
    for agent in view:
        usd = "—" if agent.measured_usd is None else f"${agent.measured_usd:.4f}"
        print(
            f"{agent.agent_id:40s}  derived_usd={usd:>12s}  "
            f"{agent.reliability['reliability_status']}"
        )

    # Same registration noise as above (not the print loop above it, which
    # must stay on real stdout) — redirect it too.
    with contextlib.redirect_stdout(io.StringIO()):
        price_map_version = CloudPricer(
            overrides=_load_overrides(overrides_path)
        ).price_map_version
    sidecar = reports_dir / "cost_view.json"
    sidecar.write_text(
        json.dumps(
            [
                {
                    "agent_id": a.agent_id,
                    "model": a.model,
                    "derived_usd": (
                        None if a.measured_usd is None else str(a.measured_usd)
                    ),
                    "usage_contract": USAGE_CONTRACT,
                    "price_map_version": price_map_version,
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
