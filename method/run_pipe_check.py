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
  uv run python method/run_pipe_check.py                  # all agents (PAID)
  uv run python method/run_pipe_check.py --agents claude_code@claude-sonnet-4-6
  uv run python method/run_pipe_check.py --with-rubric --db /tmp/bench.db
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
import tomllib
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from atp_method.evaluators import AgentEvalCaseEvaluator
from atp_method.loader import METHOD_CRITICAL_CHECK, METHOD_RUBRIC, load_suite
from atp_method.taxonomy import benchmark_id_for

from atp.adapters import create_adapter
from atp.reporters.benchmark_reporter import (
    build_report_benchmark_payload,
    normalize_report_error_class,
)
from atp.runner import TestOrchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent

# Vendored pinned copy of the ecosystem SSOT agents-catalog (ADR-ECO-003). The
# canonical file lives in `_cowork_output/contracts/agents-catalog.toml`; this
# in-repo copy keeps atp-platform self-contained. devtools CI-conformance checks
# the two stay byte-for-byte in sync. The catalog is the single source for the
# <harness>@<model> roster — edit it there, not the literals below (there are
# none anymore). claude_code@claude-sonnet-4-6 and codex_cli@gpt-5.5 carry
# routable=true (arbiter's join keys); a mismatch there = silent re-rank no-op.
CATALOG_PATH = REPO_ROOT / "method" / "agents-catalog.toml"


def _load_agent_catalog(
    path: Path = CATALOG_PATH,
) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]:
    """Load HARNESSES + AGENT_MODELS from the vendored SSOT catalog.

    Returns ``(harnesses, agent_models)`` where ``harnesses`` maps a harness
    name to ``(shim_path, model_env)`` and ``agent_models`` is the ordered list
    of ``(harness, model)`` pairs with ``tested = true`` — the ATP sweep set.
    """
    import tomllib

    try:
        with path.open("rb") as fh:
            catalog = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"vendored agents-catalog not found at {path} — re-vendor from "
            "_cowork_output/contracts/agents-catalog.toml"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"agents-catalog {path} is not valid TOML: {exc}") from exc

    try:
        harnesses: dict[str, tuple[str, str]] = {
            name: (spec["shim"], spec["model_env"])
            for name, spec in catalog["harnesses"].items()
        }
        agent_models: list[tuple[str, str]] = [
            (entry["harness"], entry["model"])
            for entry in catalog["agents"]
            if entry.get("tested", False)
        ]
    except KeyError as exc:
        raise KeyError(
            f"agents-catalog {path} missing required key {exc} (expected "
            "[harnesses.*].shim/model_env and [[agents]].harness/model)"
        ) from exc

    # A tested agent whose harness is undeclared would KeyError far from the
    # cause when AGENTS is built below — fail loudly here instead.
    undeclared = sorted({h for h, _ in agent_models if h not in harnesses})
    if undeclared:
        raise ValueError(
            f"agents-catalog {path}: tested agents reference undeclared "
            f"harness(es) {undeclared}"
        )

    return harnesses, agent_models


# harness -> (shim path relative to repo root, env var that pins the model);
# and the default (harness, model) sweep matrix. agent_id = f"{harness}@{model}"
# (the faithful provider id). Both are projected from the vendored catalog.
HARNESSES, AGENT_MODELS = _load_agent_catalog()

# agent_id -> resolved spec. The id is the routing key (faithful, with '@').
AGENTS: dict[str, dict[str, str]] = {
    f"{harness}@{model}": {
        "shim": HARNESSES[harness][0],
        "model_env": HARNESSES[harness][1],
        "model": model,
        "harness": harness,
    }
    for harness, model in AGENT_MODELS
}


def safe_agent_id(agent_id: str) -> str:
    """Filesystem-safe rendering of an agent_id for output file names.

    The faithful id (with '@', ':', '.') stays in the payload/dashboard/key;
    only file names use this form. The mapping is lossy, so callers writing
    per-agent files must guard against collisions via ``_safe_id_collision``.
    """
    return re.sub(r"[@:.]", "_", agent_id)


def _safe_id_collision(agent_ids: list[str]) -> tuple[str, str] | None:
    """First pair of agent_ids that collapse to the same ``safe_agent_id``.

    ``safe_agent_id`` is not injective, so two distinct ids could map to one
    file stem and silently overwrite each other's reports. Returns the first
    colliding (earlier, later) pair, or None when every id is distinct.
    """
    seen: dict[str, str] = {}
    for agent_id in agent_ids:
        stem = safe_agent_id(agent_id)
        if stem in seen:
            return (seen[stem], agent_id)
        seen[stem] = agent_id
    return None


# Env vars the shims need, passed through the adapter's filtered inheritance.
# API-key-shaped names are blocked by default, so they MUST be allowlisted.
ALLOWED_ENV = [
    "ANTHROPIC_API_KEY",
    "CLAUDE_MODEL",
    "CLAUDE_BIN",
    "API_MAX_TOKENS",
    "OLLAMA_MODEL",
    "OLLAMA_HOST",
    "OPENAI_API_KEY",
    "CODEX_MODEL",
    "CODEX_BIN",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_HOST",
    "MIMO_API_KEY",
    "MIMO_HOST",
    "MIMO_MODEL",
    "QWEN_API_KEY",
    "QWEN_HOST",
    "QWEN_MODEL",
    "PI_BIN",
    "PI_MODEL",
    "OPENCODE_BIN",
    "OPENCODE_MODEL",
    "OPENCODE_GLM_API_KEY",
]

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


# --------------------------------------------------------------------------- #
#  Golden-suite lock (ADR-ECO-003a D3)
#  Freeze the exact case set + BYTE content of a case dir so a re-sweep / future
#  model A/B runs on a byte-identical suite. Strict hash: any edit to a case
#  file trips the guard (re-lock intentionally via --write-suite-lock).
# --------------------------------------------------------------------------- #
SUITE_LOCK_NAME = "SUITE.lock.toml"


class SuiteLockError(Exception):
    """The golden-suite lock exists but is unreadable/malformed.

    A corrupt lock (merge conflict, hand edit, partial write) must refuse the
    run — not silently proceed (that defeats the guard) and not traceback.
    """


def _case_entries(case_dir: Path) -> list[dict[str, Any]]:
    """One entry per case yaml: {id, version, sha256}, sorted by id.

    sha256 is over the raw file bytes (strict): a content edit changes it even
    when the id set is unchanged, which a list-of-ids pin would miss.
    """
    entries: list[dict[str, Any]] = []
    for f in sorted(case_dir.glob("*.yaml")):
        raw = f.read_bytes()
        doc = yaml.safe_load(raw)
        if not isinstance(doc, dict) or "id" not in doc:
            continue
        entries.append(
            {
                "id": doc["id"],
                "version": int(doc.get("version", 0)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    entries.sort(key=lambda e: e["id"])
    return entries


def _suite_fingerprint(entries: list[dict[str, Any]]) -> str:
    """Order-independent hash over all per-case (id, sha256) pairs."""
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e["id"]):
        h.update(f"{e['id']}:{e['sha256']}\n".encode())
    return "sha256:" + h.hexdigest()


def _serialize_lock(benchmark_id: str, entries: list[dict[str, Any]]) -> str:
    """Render a deterministic SUITE.lock.toml body."""
    lines = [
        "# GENERATED by `run_pipe_check.py --write-suite-lock`. Do NOT hand-edit.",
        "# Golden-suite pin (ADR-ECO-003a D3): freezes the exact case set + byte",
        "# content so a re-sweep / future model A/B runs on a byte-identical suite.",
        "# run_pipe_check verifies the live case dir against this lock and FAILS",
        "# LOUD on drift; re-lock intentionally with --write-suite-lock.",
        "schema_version = 1",
        f'benchmark_id   = "{benchmark_id}"',
        f"case_count     = {len(entries)}",
        f'suite_hash     = "{_suite_fingerprint(entries)}"',
        "",
    ]
    for e in entries:
        lines += [
            "[[cases]]",
            f'id      = "{e["id"]}"',
            f"version = {e['version']}",
            f'sha256  = "{e["sha256"]}"',
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _write_suite_lock(case_dir: Path, benchmark_id: str) -> Path:
    """(Re)generate the golden-suite lock from the live case dir."""
    entries = _case_entries(case_dir)
    path = case_dir / SUITE_LOCK_NAME
    path.write_text(_serialize_lock(benchmark_id, entries), encoding="utf-8")
    return path


def _load_lock(case_dir: Path) -> dict[str, Any] | None:
    """Parse+validate the lock, or None when the dir has no golden-suite pin.

    Raises SuiteLockError on a present-but-malformed lock so the caller fails
    fast with a one-line message instead of a traceback.
    """
    path = case_dir / SUITE_LOCK_NAME
    if not path.is_file():
        return None
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise SuiteLockError(f"malformed {SUITE_LOCK_NAME}: {exc}") from exc
    for key in ("schema_version", "benchmark_id", "suite_hash", "cases"):
        if key not in lock:
            raise SuiteLockError(f"{SUITE_LOCK_NAME} missing required key: {key!r}")
    if not isinstance(lock["cases"], list):
        raise SuiteLockError(f"{SUITE_LOCK_NAME}: 'cases' must be a list")
    for i, case in enumerate(lock["cases"]):
        if not isinstance(case, dict) or "id" not in case or "sha256" not in case:
            raise SuiteLockError(
                f"{SUITE_LOCK_NAME}: case #{i} needs 'id' and 'sha256'"
            )
    return lock


def _diff_against_lock(case_dir: Path, lock: dict[str, Any]) -> list[str]:
    """Human-readable drift messages between the live dir and the lock ([] = ok)."""
    live = {e["id"]: e for e in _case_entries(case_dir)}
    locked = {c["id"]: c for c in lock.get("cases", [])}
    msgs: list[str] = []
    for cid in sorted(set(locked) - set(live)):
        msgs.append(f"missing case (in lock, not on disk): {cid}")
    for cid in sorted(set(live) - set(locked)):
        msgs.append(f"new case (on disk, not in lock): {cid}")
    for cid in sorted(set(live) & set(locked)):
        if live[cid]["sha256"] != locked[cid]["sha256"]:
            msgs.append(
                f"changed case content: {cid} "
                f"(lock {locked[cid]['sha256'][:12]}… vs disk "
                f"{live[cid]['sha256'][:12]}…)"
            )
    return msgs


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
    spec = AGENTS.get(agent_id)
    if spec is None:
        return f"unknown agent: {agent_id}"
    harness = spec["harness"]
    if harness == "claude_code":
        claude_bin = os.environ.get("CLAUDE_BIN", "claude")
        parts = shlex.split(claude_bin) if claude_bin else ["claude"]
        binary = parts[0] if parts else "claude"
        if shutil.which(binary) is None and not Path(binary).exists():
            return f"claude binary not found (CLAUDE_BIN={claude_bin!r})"
    if harness == "codex_cli":
        codex_bin = os.environ.get("CODEX_BIN", "codex")
        parts = shlex.split(codex_bin) if codex_bin else ["codex"]
        binary = parts[0] if parts else "codex"
        if shutil.which(binary) is None and not Path(binary).exists():
            return f"codex binary not found (CODEX_BIN={codex_bin!r})"
    if harness == "anthropic_api" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set"
    if harness == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY not set"
    if harness == "mimo" and not os.environ.get("MIMO_API_KEY"):
        return "MIMO_API_KEY not set"
    if harness == "qwen" and not os.environ.get("QWEN_API_KEY"):
        return "QWEN_API_KEY not set"
    if harness == "pi":
        binary = os.environ.get("PI_BIN", "pi")
        parts = shlex.split(binary) if binary else ["pi"]
        head = parts[0] if parts else "pi"
        if shutil.which(head) is None and not Path(head).exists():
            return f"pi binary not found (PI_BIN={binary!r})"
    if harness == "opencode":
        binary = os.environ.get("OPENCODE_BIN", "opencode")
        parts = shlex.split(binary) if binary else ["opencode"]
        head = parts[0] if parts else "opencode"
        if shutil.which(head) is None and not Path(head).exists():
            return f"opencode binary not found (OPENCODE_BIN={binary!r})"
    if harness == "ollama":
        return _preflight_ollama(spec["model"])
    return None


def _preflight_ollama(model: str) -> str | None:
    """Skip-reason if Ollama is unreachable or the model isn't pulled.

    Best-effort: any unexpected failure becomes a skip-reason rather than
    crashing the whole run.
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError):
        return f"ollama not reachable at {host}"
    except (ValueError, TypeError):
        return f"ollama not reachable at {host}"
    names = {m.get("name", "") for m in data.get("models", [])}
    if model not in names:
        return f"ollama model not pulled: {model}"
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
        "recall": None,
        "precision": None,
        "fp_count": None,
        "rubric_score": 0.0,
        "tokens": tokens,
        "cost_usd": float(raw_cost) if cost_known else 0.0,
        "cost_known": cost_known,
        "duration_seconds": duration,
        "error_class": None,
    }

    # Agent-level failure (timeout / shim error / no output) is NOT "malformed
    # findings" — it's an infra error. Record it and stop. Guarding on ``run`` too
    # narrows it to non-None for the ``run.events`` access below (response is
    # derived from run, so response-not-None already implies run-not-None).
    if run is None or response is None or response.status.value != "completed":
        status = response.status.value if response is not None else "no_run"
        base["error_class"] = normalize_report_error_class(status)
        return base

    for assertion in test_def.assertions:
        if assertion.type == METHOD_CRITICAL_CHECK:
            res = await evaluator.evaluate(test_def, response, run.events, assertion)
            check = res.checks[0]
            base["critical_pass"] = bool(check.passed)
            d = check.details or {}
            base["malformed"] = bool(d.get("malformed", False))
            base["recall"] = d.get("recall")
            base["precision"] = d.get("precision")
            base["fp_count"] = d.get("fp_count")
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the family against one agent and build its report_benchmark payload.

    Returns the payload plus the per-case grading dicts (which carry the
    continuous recall/precision/fp_count a downstream filter reads).
    """
    suite = load_suite(str(case_dir))
    spec = AGENTS[agent_id]
    adapter_env: dict[str, str] = {spec["model_env"]: spec["model"]}
    adapter = create_adapter(
        "cli",
        {
            "command": sys.executable,
            "args": [str(REPO_ROOT / spec["shim"])],
            "inherit_environment": True,
            "allowed_env_vars": ALLOWED_ENV,
            "environment": adapter_env,
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
    return payload, case_results


def _write_case_details(path: Path, case_results: list[dict[str, Any]]) -> None:
    """Write one JSON object per case_result as JSONL (downstream filter input)."""
    path.write_text("\n".join(json.dumps(c) for c in case_results))


async def _export_to_dashboard(out_dir: Path, replace: bool) -> None:
    """Import the reports just written to ``out_dir`` into the dashboard store.

    Reuses ``method/import_pipecheck_to_dashboard.py`` so a paid sweep lands in
    the dashboard's ``/ui/eval-*`` views without a separate manual import step.
    The dashboard package is optional (this harness otherwise has no DB
    dependency); a clear message is printed and the run still succeeds if it is
    absent.
    """
    # Load the sibling bridge by explicit path — no global sys.path mutation.
    # It must be registered in sys.modules before exec (its frozen dataclass
    # resolves string annotations via sys.modules[cls.__module__]).
    import importlib.util

    name = "import_pipecheck_to_dashboard"
    bridge = sys.modules.get(name)
    if bridge is None:
        bridge_path = Path(__file__).resolve().parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, bridge_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise RuntimeError(f"cannot load bridge at {bridge_path}")
        bridge = importlib.util.module_from_spec(spec)
        sys.modules[name] = bridge
        spec.loader.exec_module(bridge)

    reports = bridge.discover_reports(out_dir)
    if not reports:
        print("--to-dashboard: no reports to import.")
        return
    try:
        imported, skipped = await bridge.import_reports(reports, replace=replace)
    except SystemExit as exc:  # dashboard package not installed
        print(f"--to-dashboard: skipped — {exc}")
        return
    print(f"--to-dashboard: imported {imported} run(s), skipped {skipped}.")


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
    unknown = [a for a in agents if a not in AGENTS]
    if unknown:
        print(f"Unknown agent(s): {unknown}. Known: {list(AGENTS)}", file=sys.stderr)
        return 2

    try:
        benchmark_id = benchmark_id_for(args.task_type)
    except ValueError as exc:
        # Surface a single-line CLI error, not a traceback (matches the
        # unknown-agent path above).
        print(str(exc), file=sys.stderr)
        return 2

    if args.write_suite_lock:
        path = _write_suite_lock(case_dir, benchmark_id)
        entries = _case_entries(case_dir)
        print(
            f"Wrote golden-suite lock: {path} "
            f"({len(entries)} cases, {_suite_fingerprint(entries)})"
        )
        return 0

    # Golden-suite guard (ADR-ECO-003a D3): refuse a paid sweep on a case set
    # that has drifted from the pinned lock. Absent lock => notice, proceed;
    # malformed lock => refuse (can't verify the suite).
    try:
        lock = _load_lock(case_dir)
    except SuiteLockError as exc:
        print(f"{exc}. Re-lock with --write-suite-lock.", file=sys.stderr)
        return 2
    if lock is None:
        print(f"  [suite-lock] no {SUITE_LOCK_NAME} in {case_dir} — skipping pin check")
    else:
        drift = _diff_against_lock(case_dir, lock)
        if drift:
            print(
                "Suite drift vs golden lock — refusing to run (ADR-ECO-003a D3):",
                file=sys.stderr,
            )
            for d in drift:
                print(f"  - {d}", file=sys.stderr)
            print(
                "Re-lock intentionally with: "
                f"python method/run_pipe_check.py --case-dir {args.case_dir} "
                "--write-suite-lock",
                file=sys.stderr,
            )
            return 2
        print(
            f"  [suite-lock] OK — {len(lock['cases'])} cases match {lock['suite_hash']}"
        )

    n_cases = len(axis_by_id)
    rubric = "on" if args.with_rubric else "off"
    print(f"Pipe-check: {n_cases} case(s) in {case_dir} | task_type={args.task_type}")
    print(f"Agents: {agents} | runs={args.runs} | rubric={rubric}")

    collision = _safe_id_collision(agents)
    if collision:
        print(
            f"agent_id filename collision: {collision[0]!r} and {collision[1]!r} "
            f"both map to file stem {safe_agent_id(collision[0])!r}; "
            "rename one model to avoid silent report overwrite.",
            file=sys.stderr,
        )
        return 2

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
        payload, case_results = await _run_agent(
            agent_id,
            case_dir,
            axis_by_id,
            args.runs,
            args.with_rubric,
            args.timeout,
            benchmark_id,
        )
        safe = safe_agent_id(agent_id)
        out_file = out_dir / f"report_benchmark_{safe}.json"
        out_file.write_text(json.dumps(payload, indent=2))
        _write_case_details(out_dir / f"case_details_{safe}.jsonl", case_results)
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

    if args.to_dashboard:
        await _export_to_dashboard(out_dir, replace=args.dashboard_replace)
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
    p.add_argument("--agents", default=",".join(AGENTS))
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
    p.add_argument(
        "--write-suite-lock",
        action="store_true",
        help="(re)generate the golden-suite lock (SUITE.lock.toml) from the "
        "case dir and exit; the intentional re-lock path when cases change",
    )
    p.add_argument(
        "--to-dashboard",
        action="store_true",
        help="after the run, import the reports into the dashboard SP-1 store "
        "(reuses import_pipecheck_to_dashboard.py; needs the dashboard package)",
    )
    p.add_argument(
        "--dashboard-replace",
        action="store_true",
        help="with --to-dashboard, purge prior pipe-check rows for these "
        "suites before importing (supersede partial data)",
    )
    args = p.parse_args()
    if args.dashboard_replace and not args.to_dashboard:
        print(
            "--dashboard-replace requires --to-dashboard.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
