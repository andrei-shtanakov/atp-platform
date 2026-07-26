#!/usr/bin/env python3
"""Check that cross-project claims in the plan files are still true.

`TODO.md` and `CLAUDE.md` restate facts owned by other repos. Those
restatements drift: in 2026-07 both files claimed R-06b was blocked on
Maestro R-03 for ~3 months after R-03 shipped, because every source cited
the other rather than the owner (PR #263).

Three deterministic checks, run over the plan files:

1. **Citation liveness** (error) — every ``path:line`` / ``path:a-b`` /
   ``path:a,b`` citation resolves and the lines exist.
2. **Blocker freshness** (warning) — an ``@blocked_by:<repo>#<slug>`` whose
   slug appears only on completed (``[x]``) lines in the sibling's TODO.
3. **Pointer freshness** (warning) — a cited vault status note that is no
   longer the newest one.

Checks 2 and 3 read sibling repos (`../prograph-vault`, `../<repo>/TODO.md`),
which exist only in the polyrepo dev workspace — see the dev-only sibling
rule in CLAUDE.md. When a sibling is absent the check reports *skipped* and
never fails, so a clone without the workspace stays green.

Usage::

    uv run python scripts/ci/check_plan_citations.py            # TODO.md, CLAUDE.md
    uv run python scripts/ci/check_plan_citations.py FILE...    # explicit set
    uv run python scripts/ci/check_plan_citations.py --strict   # warnings fail too

Explicit FILE arguments resolve against the current directory, as any CLI
tool would. The citations *inside* a document always resolve against the
repo root, whatever the document's own location — that convention is what
makes a citation re-checkable by the next reader (CLAUDE.md, "reconcile
cross-project claims").

Exit codes: 0 ok, 1 problems found.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FILES = ("TODO.md", "CLAUDE.md")
REPO_ROOT = Path(__file__).resolve().parents[2]

# `path/to/file.py:12`, `:12-34`, `:12,34` — inside markdown backticks.
CITATION = re.compile(r"`([\w./-]+\.\w+):(\d+(?:[-,]\d+)?)`")
BLOCKED_BY = re.compile(r"@blocked_by:([\w.-]+)#([\w.-]+)")
STATUS_POINTER = re.compile(r"`\.\./prograph-vault/[\w./-]*/status/([\w.-]+\.md)`")
CHECKED_ITEM = re.compile(r"^\s*[-*]\s*\[[xX]\]")
LEADING_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
STATUS_DIR = Path("../prograph-vault/authored/notes/status")


@dataclass
class Report:
    """Accumulated findings across all checked files."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def ok(self, *, strict: bool) -> bool:
        """True when nothing blocking was found."""
        return not self.errors and not (strict and self.warnings)


def cited_lines(spec: str) -> list[int] | None:
    """Line numbers in a citation suffix (``12``, ``1-3``, ``1,3``).

    Returns None for a spec that cannot describe real lines: line 0, or a
    reversed range like ``9-1``. The point of the check is "these lines
    exist", so a nonsensical spec is a defect, not something to skip.
    """
    if "-" in spec:
        start, end = (int(part) for part in spec.split("-", 1))
        if start < 1 or end < start:
            return None
        return [start, end]
    numbers = [int(part) for part in spec.split(",")]
    if any(number < 1 for number in numbers):
        return None
    return numbers


def within_workspace(target: Path) -> bool:
    """True if target stays inside the repo or its polyrepo workspace.

    Citations are authored text; one that resolves outside the workspace is
    a broken citation, and following it would make the hook read files it
    has no business reading.
    """
    workspace = REPO_ROOT.parent.resolve()
    return target == workspace or workspace in target.parents


def check_citations(doc: Path, text: str, report: Report) -> None:
    """Verify every ``path:line`` citation resolves to existing lines."""
    seen_missing_siblings: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        for path_str, spec in CITATION.findall(line):
            where = f"{doc.name}:{lineno}"
            target = (REPO_ROOT / path_str).resolve()
            is_sibling = path_str.startswith("../")

            wanted_lines = cited_lines(spec)
            if wanted_lines is None:
                report.errors.append(f"{where} — invalid line spec: {path_str}:{spec}")
                continue
            if not within_workspace(target):
                report.errors.append(
                    f"{where} — citation escapes the workspace: {path_str}"
                )
                continue
            if not target.exists():
                if is_sibling:
                    if path_str not in seen_missing_siblings:
                        seen_missing_siblings.add(path_str)
                        report.skipped.append(f"{where} — sibling absent: {path_str}")
                else:
                    report.errors.append(f"{where} — no such file: {path_str}")
                continue
            try:
                total = len(target.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError) as exc:
                report.errors.append(f"{where} — cannot read {path_str}: {exc}")
                continue
            for wanted in wanted_lines:
                if wanted > total:
                    report.errors.append(
                        f"{where} — {path_str}:{spec} out of range "
                        f"(file has {total} lines)"
                    )
                    break


def check_blockers(doc: Path, text: str, report: Report) -> None:
    """Flag ``@blocked_by`` targets that look already completed upstream."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        for repo, slug in BLOCKED_BY.findall(line):
            sibling = (REPO_ROOT / ".." / repo / "TODO.md").resolve()
            where = f"{doc.name}:{lineno}"
            if not sibling.exists():
                report.skipped.append(f"{where} — no sibling TODO for '{repo}'")
                continue
            matches = [
                item
                for item in sibling.read_text(encoding="utf-8").splitlines()
                if slug in item and item.lstrip().startswith(("-", "*"))
            ]
            if not matches:
                report.warnings.append(
                    f"{where} — '{slug}' not found in {repo}/TODO.md; "
                    "blocker may be renamed or gone"
                )
            elif all(CHECKED_ITEM.match(item) for item in matches):
                report.warnings.append(
                    f"{where} — '{slug}' is checked off in {repo}/TODO.md; "
                    "blocker looks closed"
                )


def note_date(name: str) -> str | None:
    """Leading ``YYYY-MM-DD`` of a vault note filename, or None."""
    match = LEADING_DATE.match(name)
    return match.group(1) if match else None


def check_status_pointer(doc: Path, text: str, report: Report) -> None:
    """Flag a cited vault status note older than the newest available one.

    Compares dates, not filenames: several notes can share a day
    (``2026-07-08-status.md`` and ``2026-07-08-1228-status.md``), and
    picking between same-day variants is an editorial call, not a defect.
    """
    status_dir = (REPO_ROOT / STATUS_DIR).resolve()
    if not status_dir.is_dir():
        report.skipped.append(f"{doc.name} — vault status dir absent")
        return
    dates = [d for d in (note_date(p.name) for p in status_dir.glob("*.md")) if d]
    if not dates:
        return
    newest = max(dates)
    for lineno, line in enumerate(text.splitlines(), start=1):
        for cited in STATUS_POINTER.findall(line):
            cited_date = note_date(cited)
            if cited_date and cited_date < newest:
                report.warnings.append(
                    f"{doc.name}:{lineno} — cites a note from {cited_date}, "
                    f"newest vault status is {newest}"
                )


def check_file(doc: Path, report: Report) -> None:
    """Run every check over one plan file."""
    text = doc.read_text(encoding="utf-8")
    check_citations(doc, text, report)
    check_blockers(doc, text, report)
    check_status_pointer(doc, text, report)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "files", nargs="*", help="plan files (default: TODO.md CLAUDE.md)"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    args = parser.parse_args(argv)

    targets = [Path(name) for name in args.files] or [
        REPO_ROOT / name for name in DEFAULT_FILES
    ]
    report = Report()
    for doc in targets:
        if not doc.exists():
            report.errors.append(f"plan file missing: {doc}")
            continue
        check_file(doc, report)

    for note in report.skipped:
        print(f"skip:  {note}")
    for note in report.warnings:
        print(f"WARN:  {note}")
    for note in report.errors:
        print(f"ERROR: {note}")

    if report.ok(strict=args.strict):
        print(
            f"plan citations ok "
            f"({len(report.warnings)} warning(s), {len(report.skipped)} skipped)"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
