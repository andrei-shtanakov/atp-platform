#!/usr/bin/env python3
"""R-07 Task 6 — code-review pipe-check harness (PAID when not --dry-run).

Drives the `code-review-planted-defect` agent-eval-case family through the ATP
CLI adapter against one or more spawner shims, grades each case with the
deterministic findings_match gate, and emits a `report_benchmark-v1` payload per
agent — proving the eval→contract pipe carries a *differentiating* routing signal
on live models. This is the run wiring the unit smokes stood in for.

Routing signal = `critical_pass_rate` + `malformed_rate` + `breakpoint_axis_level`
(all deterministic; no judge needed). The rubric is non-gating and OFF by default
(`--with-rubric` enables it, using the default LLM judge — OpenAI when no
ANTHROPIC_API_KEY is present).

Agents (two spawners, by design):
  - `claude_code`    — `claude -p ... --output-format json` (product harness;
                       uses the Claude Code session login, no API key needed).
  - `anthropic_api`  — raw Anthropic Messages API (the "harness vs raw API"
                       baseline; needs ANTHROPIC_API_KEY). LABELED BASELINE ONLY —
                       never substitutes the CLI agent_id in arbiter routing.

Usage:
  uv run python method/run_pipe_check.py --dry-run        # show plan, no calls
  uv run python method/run_pipe_check.py                  # both agents (PAID)
  uv run python method/run_pipe_check.py --agents claude_code
  uv run python method/run_pipe_check.py --with-rubric --db /tmp/bench.db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from atp_method.evaluators import AgentEvalCaseEvaluator
from atp_method.loader import METHOD_CRITICAL_CHECK, METHOD_RUBRIC, load_suite
from atp_method.taxonomy import benchmark_id_for

from atp.adapters import create_adapter
from atp.reporters.benchmark_reporter import build_report_benchmark_payload
from atp.runner import TestOrchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent

# agent_id -> shim path (relative to repo root). Both run via the current
# interpreter so the raw-API shim sees the installed `anthropic` SDK.
SHIMS: dict[str, str] = {
    "claude_code": "method/spawners/claude_code_shim.py",
    "anthropic_api": "method/spawners/anthropic_api_shim.py",
}

# Env vars the shims need, passed through the adapter's filtered inheritance.
# API-key-shaped names are blocked by default, so they MUST be allowlisted.
ALLOWED_ENV = ["ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_BIN", "API_MAX_TOKENS"]

_BENCH_DDL = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
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


def _axis_by_id(case_dir: Path) -> dict[str, str]:
    """Map each case id to its axis_level (read straight from the YAML)."""
    out: dict[str, str] = {}
    for f in sorted(case_dir.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text())
        if isinstance(doc, dict) and "id" in doc:
            out[doc["id"]] = doc.get("axis_level", "unknown")
    return out


def _preflight(agent_id: str) -> str | None:
    """Return a skip-reason if the agent can't run here, else None."""
    if agent_id == "claude_code":
        # The shim invokes CLAUDE_BIN (default "claude"), which may be an absolute
        # path or a "python .../fake.py" command — not necessarily on PATH. Check
        # the first token resolves (on PATH or as an existing file).
        claude_bin = os.environ.get("CLAUDE_BIN", "claude")
        parts = shlex.split(claude_bin) if claude_bin else ["claude"]
        binary = parts[0] if parts else "claude"
        if shutil.which(binary) is None and not Path(binary).exists():
            return f"claude binary not found (CLAUDE_BIN={claude_bin!r})"
    if agent_id == "anthropic_api" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set"
    return None


async def _grade_case(
    evaluator: AgentEvalCaseEvaluator,
    test_result: Any,
    axis_level: str,
    with_rubric: bool,
) -> dict[str, Any]:
    """Evaluate one case's run into a reporter case_result dict."""
    test_def = test_result.test
    run = test_result.runs[0] if test_result.runs else None
    response = run.response if run else None

    metrics = getattr(response, "metrics", None)
    tokens = int(getattr(metrics, "total_tokens", None) or 0)
    # Track whether cost is actually known: the raw-API baseline reports
    # cost_usd=null, which must NOT be flattened to 0.0 ("free" ≠ "unknown").
    # We still feed 0.0 to the reporter (which sums), then null the aggregate in
    # _run_agent when any case's cost is unknown.
    raw_cost = getattr(metrics, "cost_usd", None)
    cost_known = raw_cost is not None
    duration = float(getattr(run, "duration_seconds", None) or 0.0)

    base: dict[str, Any] = {
        "case_id": test_def.id,
        "axis_level": axis_level,
        "critical_pass": False,
        "malformed": False,
        "rubric_score": 0.0,
        "tokens": tokens,
        "cost_usd": float(raw_cost) if cost_known else 0.0,
        "cost_known": cost_known,
        "duration_seconds": duration,
        "error_class": None,
    }

    # Agent-level failure (timeout / shim error / no output) is NOT "malformed
    # findings" — it's an infra error. Record it and stop.
    if response is None or response.status.value != "completed":
        status = response.status.value if response is not None else "no_run"
        base["error_class"] = status
        return base

    for assertion in test_def.assertions:
        if assertion.type == METHOD_CRITICAL_CHECK:
            res = await evaluator.evaluate(test_def, response, run.events, assertion)
            check = res.checks[0]
            base["critical_pass"] = bool(check.passed)
            base["malformed"] = bool((check.details or {}).get("malformed", False))
        elif assertion.type == METHOD_RUBRIC and with_rubric:
            res = await evaluator.evaluate(test_def, response, run.events, assertion)
            base["rubric_score"] = float(res.checks[0].score)

    return base


async def _run_agent(
    agent_id: str,
    case_dir: Path,
    axis_by_id: dict[str, str],
    runs: int,
    with_rubric: bool,
    timeout_s: float,
    benchmark_id: str,
) -> dict[str, Any]:
    """Run the family against one agent and build its report_benchmark payload."""
    suite = load_suite(str(case_dir))
    adapter = create_adapter(
        "cli",
        {
            "command": sys.executable,
            "args": [str(REPO_ROOT / SHIMS[agent_id])],
            "inherit_environment": True,
            "allowed_env_vars": ALLOWED_ENV,
            "timeout_seconds": timeout_s,
        },
    )
    # findings_match needs no judge; rubric uses the default judge (OpenAI when
    # ANTHROPIC_API_KEY is absent) only when --with-rubric is set.
    evaluator = AgentEvalCaseEvaluator()

    async with TestOrchestrator(adapter=adapter, runs_per_test=runs) as orch:
        result = await orch.run_suite(suite, agent_name=agent_id, runs_per_test=runs)

    case_results = [
        await _grade_case(
            evaluator, tr, axis_by_id.get(tr.test.id, "unknown"), with_rubric
        )
        for tr in result.tests
    ]
    payload = build_report_benchmark_payload(
        run_id=str(uuid.uuid4()),
        benchmark_id=benchmark_id,
        agent_id=agent_id,
        ts=datetime.now(tz=UTC).isoformat(),
        case_results=case_results,
    )
    # If any case's cost was unknown (e.g. the raw-API baseline), the summed
    # total is meaningless — report null per the contract (total_cost_usd allows
    # null) rather than a misleading 0.0/partial sum.
    if any(not c["cost_known"] for c in case_results):
        payload["total_cost_usd"] = None
    return payload


def _insert(db_path: Path, payload: dict[str, Any]) -> None:
    """Insert a payload into a local benchmark_runs table (idempotent)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_BENCH_DDL)
        conn.execute(
            "INSERT INTO benchmark_runs(run_id, benchmark_id, agent_id, score, "
            "score_components, per_task, total_tokens, total_cost_usd, "
            "duration_seconds, ts) VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id) DO NOTHING",
            (
                payload["run_id"],
                payload["benchmark_id"],
                payload["agent_id"],
                payload["score"],
                json.dumps(payload["score_components"]),
                json.dumps(payload["per_task"]),
                payload["total_tokens"],
                payload["total_cost_usd"],
                payload["duration_seconds"],
                payload["ts"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _summary_line(payload: dict[str, Any]) -> str:
    sc = payload["score_components"]
    bp = payload.get("breakpoint_axis_level", "—")
    cost = payload["total_cost_usd"]
    cost_str = "unknown" if cost is None else f"${cost:.4f}"
    return (
        f"  {payload['agent_id']:<14} "
        f"critical_pass_rate={sc['critical_pass_rate']:.3f} "
        f"malformed_rate={sc['malformed_rate']:.3f} "
        f"mean_rubric={sc['mean_rubric']:.3f} "
        f"breakpoint={bp} "
        f"tokens={payload['total_tokens']} "
        f"cost={cost_str}"
    )


async def _main_async(args: argparse.Namespace) -> int:
    case_dir = (REPO_ROOT / args.case_dir).resolve()
    axis_by_id = _axis_by_id(case_dir)
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in agents if a not in SHIMS]
    if unknown:
        print(f"Unknown agent(s): {unknown}. Known: {list(SHIMS)}", file=sys.stderr)
        return 2

    try:
        benchmark_id = benchmark_id_for(args.task_type)
    except ValueError as exc:
        # Surface a single-line CLI error, not a traceback (matches the
        # unknown-agent path above).
        print(str(exc), file=sys.stderr)
        return 2

    n_cases = len(axis_by_id)
    rubric = "on" if args.with_rubric else "off"
    print(f"Pipe-check: {n_cases} case(s) in {case_dir} | task_type={args.task_type}")
    print(f"Agents: {agents} | runs={args.runs} | rubric={rubric}")

    runnable: list[str] = []
    for agent_id in agents:
        reason = _preflight(agent_id)
        if reason:
            print(f"  SKIP {agent_id}: {reason}")
        else:
            runnable.append(agent_id)

    if args.dry_run:
        print("\n[dry-run] Would run (PAID):", runnable or "(nothing)")
        return 0
    if not runnable:
        print(
            "\nNothing runnable. Set ANTHROPIC_API_KEY / install `claude`.",
            file=sys.stderr,
        )
        return 1

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = (REPO_ROOT / args.db).resolve() if args.db else None

    print("\n--- PAID RUN: invoking live models ---")
    payloads: list[dict[str, Any]] = []
    for agent_id in runnable:
        print(f"Running {agent_id} ...")
        payload = await _run_agent(
            agent_id,
            case_dir,
            axis_by_id,
            args.runs,
            args.with_rubric,
            args.timeout,
            benchmark_id,
        )
        out_file = out_dir / f"report_benchmark_{agent_id}.json"
        out_file.write_text(json.dumps(payload, indent=2))
        payloads.append(payload)
        if db_path is not None:
            _insert(db_path, payload)

    print("\n=== report_benchmark payloads ===")
    for p in payloads:
        print(_summary_line(p))
    print(f"\nPayloads written to {out_dir}")
    if db_path is not None:
        print(f"Inserted into {db_path} (benchmark_runs)")

    # Pipe-check verdict hint: a valid signal needs the agents to DIFFER on the
    # deterministic gate (else the tube carries no routing information).
    if len(payloads) >= 2:
        rates = {
            p["agent_id"]: p["score_components"]["critical_pass_rate"] for p in payloads
        }
        spread = max(rates.values()) - min(rates.values())
        print(f"\ncritical_pass_rate spread across agents: {spread:.3f} ({rates})")
        print("(Phase-0 anti-pattern: spread ~0 => no differentiating signal.)")
    return 0


def main() -> int:
    """Parse args and run the pipe-check."""
    # Load a repo-root .env (e.g. ANTHROPIC_API_KEY for the anthropic_api shim),
    # matching the `atp` CLI. Real env vars win (override=False).
    from dotenv import load_dotenv

    load_dotenv(override=False)
    p = argparse.ArgumentParser(description="R-07 Task 6 code-review pipe-check")
    p.add_argument("--case-dir", default="method/cases/code-review")
    p.add_argument(
        "--task-type",
        default="review",
        help="internal task_type; benchmark_id is derived for the arbiter export",
    )
    p.add_argument("--agents", default=",".join(SHIMS))
    p.add_argument("--runs", type=int, default=1)
    p.add_argument(
        "--with-rubric", action="store_true", help="grade the LLM rubric too"
    )
    p.add_argument("--out-dir", default="_cowork_output/r07-pipecheck")
    p.add_argument("--db", default=None, help="optional sqlite path for benchmark_runs")
    p.add_argument(
        "--timeout", type=float, default=700.0, help="per-case adapter timeout"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="show the plan, make no calls"
    )
    return asyncio.run(_main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
