"""Smoke: a report_benchmark-v1 payload INSERTs into a benchmark_runs-shaped
table with idempotency (mirrors arbiter db.rs ON CONFLICT DO NOTHING). This
stands in for the arbiter MCP import until the separate arbiter plan lands."""

import json
import sqlite3

from atp.reporters.benchmark_reporter import build_report_benchmark_payload

DDL = """
CREATE TABLE benchmark_runs (
    run_id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    score REAL NOT NULL,
    score_components TEXT,
    per_task TEXT,
    total_tokens INTEGER,
    total_cost_usd REAL,
    duration_seconds REAL,
    ts TEXT
);
"""


def _insert(conn, p):
    conn.execute(
        "INSERT INTO benchmark_runs(run_id, benchmark_id, agent_id, score, "
        "score_components, per_task, total_tokens, total_cost_usd, "
        "duration_seconds, ts) "
        "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO NOTHING",
        (
            p["run_id"],
            p["benchmark_id"],
            p["agent_id"],
            p["score"],
            json.dumps(p["score_components"]),
            json.dumps(p["per_task"]),
            p["total_tokens"],
            p["total_cost_usd"],
            p["duration_seconds"],
            p["ts"],
        ),
    )
    conn.commit()


def test_payload_inserts_and_is_idempotent() -> None:
    payload = build_report_benchmark_payload(
        run_id="run-smoke",
        benchmark_id="code-review",
        agent_id="claude_code",
        ts="2026-06-13T10:00:00Z",
        case_results=[
            {
                "case_id": "c",
                "axis_level": "moderate",
                "critical_pass": True,
                "rubric_score": 0.8,
                "tokens": 920,
                "cost_usd": 0.0123,
                "duration_seconds": 4.2,
                "error_class": None,
            },
        ],
    )
    conn = sqlite3.connect(":memory:")
    conn.executescript(DDL)
    _insert(conn, payload)
    _insert(conn, payload)  # duplicate run_id -> no second row
    (n,) = conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()
    assert n == 1
    (score, bid) = conn.execute(
        "SELECT score, benchmark_id FROM benchmark_runs WHERE run_id='run-smoke'"
    ).fetchone()
    assert bid == "code-review" and 0.0 <= score <= 1.0
