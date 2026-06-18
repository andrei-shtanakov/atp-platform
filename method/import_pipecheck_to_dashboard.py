"""Import pipe-check ``report_benchmark`` results into the dashboard SP-1 store.

``method/run_pipe_check.py`` emits ``report_benchmark-v1`` payloads to JSON (and a
standalone ``benchmark_runs`` sqlite) for arbiter routing — those never reach the
ATP dashboard, so ``/ui/eval-leaderboard`` and ``/ui/eval-trends`` render nothing
for code-review / req-extraction runs (TODO: "R-07 визуализация результатов").

This bridge reads the already-produced JSON reports and writes one completed
``SuiteExecution`` per report into the dashboard database (the same store
``atp test`` writes to and ``init_database`` resolves), so the existing eval views
light up **without re-running the agents** (the paid sweep is reused as-is).

Idempotent: a report whose ``run_id`` already exists as a ``SuiteExecution.run_uuid``
is skipped, so re-running the import never duplicates rows.

Usage::

    uv run python method/import_pipecheck_to_dashboard.py --dry-run
    uv run python method/import_pipecheck_to_dashboard.py
    uv run python method/import_pipecheck_to_dashboard.py \
        --results-dir _cowork_output/r07-pipecheck

Then open ``atp dashboard`` → http://localhost:8080/ui/eval-leaderboard .
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "_cowork_output" / "r07-pipecheck"

# benchmark_id (suite) → task_type, used only when a report carries no per_task rows.
_TASK_TYPE_BY_BENCHMARK = {
    "code-review": "review",
    "req-extraction": "req-extraction",
}


@dataclass(frozen=True)
class ParsedReport:
    """The subset of a ``report_benchmark`` payload the dashboard store needs."""

    run_uuid: str
    suite_name: str
    agent_name: str
    started_at: datetime
    duration_seconds: float
    critical_pass_rate: float | None
    malformed_rate: float | None
    mean_rubric: float | None
    task_type: str | None
    breakpoint_axis_level: str | None
    source_file: Path


def _task_type_for(report: dict[str, Any], benchmark_id: str) -> str | None:
    """Derive task_type from the first per-task row, falling back to the suite map."""
    per_task = report.get("per_task") or []
    if per_task and isinstance(per_task[0], dict):
        tt = per_task[0].get("task_type")
        if tt:
            return str(tt)
    return _TASK_TYPE_BY_BENCHMARK.get(benchmark_id)


def parse_report(path: Path) -> ParsedReport | None:
    """Parse one ``report_benchmark_*.json`` file; return None if it is not one."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    run_uuid = data.get("run_id")
    benchmark_id = data.get("benchmark_id")
    agent_id = data.get("agent_id")
    ts = data.get("ts")
    if not (run_uuid and benchmark_id and agent_id and ts):
        return None
    comps = data.get("score_components") or {}
    bp = data.get("breakpoint_axis_level")
    return ParsedReport(
        run_uuid=str(run_uuid),
        suite_name=str(benchmark_id),
        agent_name=str(agent_id),
        started_at=datetime.fromisoformat(ts),
        duration_seconds=float(data.get("duration_seconds") or 0.0),
        critical_pass_rate=comps.get("critical_pass_rate"),
        malformed_rate=comps.get("malformed_rate"),
        mean_rubric=comps.get("mean_rubric"),
        task_type=_task_type_for(data, str(benchmark_id)),
        breakpoint_axis_level=str(bp) if bp is not None else None,
        source_file=path,
    )


def discover_reports(results_dir: Path) -> list[ParsedReport]:
    """Find and parse every report_benchmark JSON under results_dir (recursive)."""
    reports: list[ParsedReport] = []
    for path in sorted(results_dir.rglob("report_benchmark_*.json")):
        parsed = parse_report(path)
        if parsed is not None:
            reports.append(parsed)
    return reports


def case_details_path_for(report_path: Path) -> Path:
    """Sibling ``case_details_<agent>.jsonl`` for a report_benchmark file.

    ``report_benchmark_<agent>.json`` -> ``case_details_<agent>.jsonl`` in the
    same directory.
    """
    stem = report_path.name.replace("report_benchmark_", "case_details_", 1)
    stem = stem.rsplit(".json", 1)[0]
    return report_path.with_name(f"{stem}.jsonl")


def parse_case_details(path: Path) -> list[dict[str, Any]]:
    """Parse a ``case_details_<agent>.jsonl`` file into per-case dicts.

    Returns [] if the file is missing or unreadable. Skips blank and malformed
    lines so a single bad line never sinks the whole import.
    """
    try:
        text = path.read_text()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


async def import_reports(
    reports: list[ParsedReport], db_url: str | None = None
) -> tuple[int, int]:
    """Write reports into the dashboard store. Returns (imported, skipped)."""
    try:
        from atp.dashboard import ResultStorage, init_database
        from atp.dashboard.models import SuiteExecution
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "Dashboard package not installed. Install with: uv add atp-dashboard"
        ) from exc
    from sqlalchemy import select

    db = await init_database(url=db_url)
    imported = 0
    skipped = 0
    async with db.session() as session:
        storage = ResultStorage(session)
        for r in reports:
            existing = (
                await session.execute(
                    select(SuiteExecution).where(SuiteExecution.run_uuid == r.run_uuid)
                )
            ).scalar_one_or_none()
            if existing is not None:
                skipped += 1
                continue
            execution = await storage.create_suite_execution_by_name(
                suite_name=r.suite_name,
                agent_name=r.agent_name,
                started_at=r.started_at,
                adapter="pipe-check",
                model=r.agent_name,
            )
            execution.run_uuid = r.run_uuid
            await storage.update_suite_execution(
                execution,
                completed_at=r.started_at + timedelta(seconds=r.duration_seconds),
                status="completed",
                aggregates={
                    "critical_pass_rate": r.critical_pass_rate,
                    "malformed_rate": r.malformed_rate,
                    "mean_rubric": r.mean_rubric,
                    "task_type": r.task_type,
                    "breakpoint_axis_level": r.breakpoint_axis_level,
                },
            )
            imported += 1
        await session.commit()
    return imported, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to scan recursively (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Dashboard DB URL override (default: ATP_DATABASE_URL or ~/.atp).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and list reports without writing to the database",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        return 1

    reports = discover_reports(results_dir)
    if not reports:
        print(f"No report_benchmark_*.json found under {results_dir}")
        return 0

    print(f"Found {len(reports)} report(s) under {results_dir}:")
    for r in reports:
        cpr = "n/a" if r.critical_pass_rate is None else f"{r.critical_pass_rate:.3f}"
        print(
            f"  [{r.suite_name:14s}] {r.agent_name:18s} "
            f"crit_pass={cpr}  {r.started_at.isoformat()}"
        )

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    imported, skipped = asyncio.run(import_reports(reports, db_url=args.db))
    print(f"\nImported {imported} new run(s); skipped {skipped} already present.")
    print("Open: atp dashboard → /ui/eval-leaderboard and /ui/eval-trends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
