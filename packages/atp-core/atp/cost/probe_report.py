"""Action №0 exposure-probe report (ADR-ECO-003e).

Aggregates the JSONL written by JsonlUsageCapture into the runtime
acceptance-gate evidence: which adapter paths carry token usage, whether
model ids and cost_usd are ever populated, and token volume per path.

Usage: uv run python -m atp.cost.probe_report <usage.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _AdapterAgg:
    calls: int = 0
    with_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    statuses: dict[str, int] = field(default_factory=dict)


def load_rows(path: Path) -> list[dict]:
    """Read one UsageRecord dict per JSONL line."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_report(rows: list[dict]) -> str:
    """Render the markdown evidence table for Action №0."""
    aggs: dict[str, _AdapterAgg] = defaultdict(_AdapterAgg)
    model_known = 0
    cost_populated = 0
    for r in rows:
        agg = aggs[r["adapter_type"]]
        agg.calls += 1
        status = r.get("status", "?")
        agg.statuses[status] = agg.statuses.get(status, 0) + 1
        usage = r.get("usage")
        if usage is not None:
            agg.with_usage += 1
            agg.input_tokens += usage["input_tokens"]
            agg.output_tokens += usage["output_tokens"]
            agg.cache_read_tokens += usage["cache_read_tokens"]
            agg.cache_creation_tokens += usage["cache_creation_tokens"]
        if r.get("model") is not None:
            model_known += 1
        if r.get("reported_cost_usd") is not None:
            cost_populated += 1

    total = len(rows)
    lines = [
        "# 003e Action №0 — usage-capture exposure report",
        "",
        f"- records: {total}",
        f"- model known: {model_known}/{total}",
        f"- cost_usd populated: {cost_populated}/{total}",
        "",
        "| adapter | calls | with_usage | tokens_total | input | output |",
        "|---|---|---|---|---|---|",
    ]
    for adapter in sorted(aggs):
        a = aggs[adapter]
        tokens_total = (
            a.input_tokens
            + a.output_tokens
            + a.cache_read_tokens
            + a.cache_creation_tokens
        )
        lines.append(
            f"| {adapter} | {a.calls} | {a.with_usage} "
            f"| {tokens_total} | {a.input_tokens} | {a.output_tokens} |"
        )
    lines.append("")
    for adapter in sorted(aggs):
        statuses = ", ".join(
            f"{k}={v}" for k, v in sorted(aggs[adapter].statuses.items())
        )
        lines.append(f"- {adapter} statuses: {statuses}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m atp.cost.probe_report <usage.jsonl>")
        return 2
    print(build_report(load_rows(Path(args[0]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
