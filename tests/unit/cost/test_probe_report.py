"""Aggregation for the Action №0 exposure probe (ADR-ECO-003e)."""

import json
from pathlib import Path

from atp.cost.probe_report import build_report, load_rows


def row(
    adapter: str,
    usage: dict | None,
    model: str | None = None,
    cost: float | None = None,
    call_id: str = "c1",
) -> dict:
    return {
        "call_id": call_id,
        "timestamp": "2026-07-15T00:00:00+00:00",
        "adapter_type": adapter,
        "status": "completed",
        "model": model,
        "provider": None,
        "usage": usage,
        "reported_cost_usd": cost,
        "test_id": "t",
    }


USAGE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 25,
    "usage_source": "measured",
}


def test_load_rows_reads_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(
        json.dumps(row("cli", USAGE)) + "\n" + json.dumps(row("http", None)) + "\n"
    )
    assert len(load_rows(p)) == 2


def test_report_groups_by_adapter_and_counts_coverage() -> None:
    rows = [
        row("cli", USAGE, call_id="a"),
        row("cli", None, call_id="b"),
        row("http", None, call_id="c"),
    ]
    report = build_report(rows)
    # cli: 2 calls, 1 with usage; http: 1 call, 0 with usage
    assert "| cli | 2 | 1 |" in report
    assert "| http | 1 | 0 |" in report


def test_report_flags_cost_usd_and_model_coverage() -> None:
    rows = [row("cli", USAGE, model=None, cost=None, call_id="a")]
    report = build_report(rows)
    assert "cost_usd populated: 0/1" in report
    assert "model known: 0/1" in report


def test_report_sums_tokens() -> None:
    # USAGE per record: input 100 + output 50 + cache_read 25 = 175 total.
    rows = [row("cli", USAGE, call_id="a"), row("cli", USAGE, call_id="b")]
    report = build_report(rows)
    assert "| cli | 2 | 2 | 350 | 200 | 100 |" in report
