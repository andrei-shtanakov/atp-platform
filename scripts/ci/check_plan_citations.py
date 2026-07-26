#!/usr/bin/env python3
"""Check that cross-project claims in the plan files are still true.

`TODO.md` and `CLAUDE.md` restate facts owned by other repos. Those
restatements drift: in 2026-07 both files claimed R-06b was blocked on
Maestro R-03 for ~3 months after R-03 shipped, because every source cited
the other rather than the owner (PR #263).

The governing rule: **a cross-repo status is a cache, not a source of
truth.** A claim about another project must name its owner, its evidence,
when it was observed, and when it stops being trusted. These files are then
caches with a TTL rather than mutually-citing authorities.

Deterministic checks, run over the plan files:

1. **Citation liveness** (error) — every ``path:line`` / ``path:a-b`` /
   ``path:a,b`` citation resolves and the lines exist.
2. **Blocker freshness** (warning) — an ``@blocked_by:<repo>#<slug>`` whose
   slug appears only on completed (``[x]``) lines in the sibling's TODO.
3. **Pointer freshness** (warning) — a cited vault status note that is no
   longer the newest one.
4. **Owner form** (error) — ``@owner:`` is ``github:<login>`` or
   ``github-team:<org>/<team>``; no display names, no free text; at most one
   per item. ``@owner:TBD`` requires ``@owner-decision-by:<date>`` and fails
   once that date has passed. Whether the login *exists* is deliberately not
   checked here — that needs the network, which would make the hook brittle
   and unusable offline; it belongs in CI.
5. **Status freshness** (error/warning) — the metadata set
   ``@source-owner`` / ``@source-ref`` / ``@observed-at`` / ``@recheck-by``
   is all-or-nothing (a partial set is the dangerous case), and an expired
   ``@recheck-by`` fails. An ``@blocked_by:`` with no metadata warns.
6. **Owner coverage** (warning) — open actionable items with no ``@owner``,
   reported as one count while the backlog backfill is pending.

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
from datetime import date
from pathlib import Path

DEFAULT_FILES = ("TODO.md", "CLAUDE.md")
REPO_ROOT = Path(__file__).resolve().parents[2]

# `path/to/file.py:12`, `:12-34`, `:12,34` — inside markdown backticks.
CITATION = re.compile(r"`([\w./-]+\.\w+):(\d+(?:[-,]\d+)?)`")
BLOCKED_BY = re.compile(r"@blocked_by:([\w.-]+)#([\w.-]+)")
STATUS_POINTER = re.compile(r"`\.\./prograph-vault/[\w./-]*/status/([\w.-]+\.md)`")
CHECKED_ITEM = re.compile(r"^\s*[-*]\s*\[[xX]\]")
ANY_ITEM = re.compile(r"^\s*[-*]\s*\[[ xX]\]")
LEADING_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
STATUS_DIR = Path("../prograph-vault/authored/notes/status")

# Owner identity is a handle, never a display name: an accountable GitHub
# user or team. `TBD` is a temporary escape with an expiry.
OWNER_TAG = re.compile(r"@owner:(\S+)")
OWNER_VALUE = re.compile(
    r"^(?:github:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"|github-team:[A-Za-z0-9._-]+/[A-Za-z0-9._-]+"
    r"|TBD)$"
)
DATED_TAGS = ("@owner-decision-by", "@observed-at", "@recheck-by")
# All four travel together; a partial set is worse than none, because it
# looks sourced without being re-checkable.
FRESHNESS_TAGS = ("@source-owner", "@source-ref", "@observed-at", "@recheck-by")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def today() -> date:
    """Current date, isolated so tests can pin it."""
    return date.today()


def tag_values(block: str, tag: str) -> list[str]:
    """Every value given for ``tag`` inside an item block."""
    return re.findall(rf"{re.escape(tag)}:(\S+)", block)


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


@dataclass
class Item:
    """One checkbox item and the indented lines that belong to it."""

    lineno: int
    block: str
    is_open: bool


def iter_items(text: str) -> list[Item]:
    """Split a plan file into checkbox items with their continuation lines.

    Tags routinely sit on a continuation line rather than the checkbox line
    itself, so checks must see the whole block, not one line.

    Single linear pass: slicing the remaining lines per item copied the tail
    of the file for every checkbox, which is quadratic on a long plan file.
    """
    items: list[Item] = []
    current: Item | None = None
    block: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ANY_ITEM.match(line):
            if current is not None:
                current.block = "\n".join(block)
                items.append(current)
            current = Item(
                lineno=lineno, block="", is_open=not CHECKED_ITEM.match(line)
            )
            block = [line]
        elif current is not None:
            block.append(line)
    if current is not None:
        current.block = "\n".join(block)
        items.append(current)
    return items


def collect_owners(items: list[Item]) -> list[str]:
    """Well-formed owner handles, as ``user <login>`` / ``team <org>/<slug>``.

    Exists so CI resolves the *same* handles this file's grammar accepts.
    A second, looser parser in shell would silently truncate a handle at the
    first character its character class omits and then cheerfully validate a
    different, real account.
    """
    seen: list[str] = []
    for item in items:
        for owner in OWNER_TAG.findall(item.block):
            if not OWNER_VALUE.match(owner) or owner == "TBD":
                continue
            kind, _, value = owner.partition(":")
            line = f"{'user' if kind == 'github' else 'team'} {value}"
            if line not in seen:
                seen.append(line)
    return seen


def check_dates(where: str, block: str, report: Report) -> dict[str, date]:
    """Validate every dated tag in a block; return the ones that parsed."""
    parsed: dict[str, date] = {}
    for tag in DATED_TAGS:
        for value in tag_values(block, tag):
            if not ISO_DATE.match(value):
                report.errors.append(
                    f"{where} — {tag} must be YYYY-MM-DD, got '{value}'"
                )
                continue
            parsed[tag] = date.fromisoformat(value)
    return parsed


def check_owner_form(doc: Path, report: Report, items: list[Item]) -> None:
    """Enforce the owner handle grammar and the TBD expiry."""
    for item in items:
        where = f"{doc.name}:{item.lineno}"
        owners = OWNER_TAG.findall(item.block)
        if len(owners) > 1:
            report.errors.append(
                f"{where} — {len(owners)} @owner tags; an item has exactly one"
            )
        dates = check_dates(where, item.block, report)
        for owner in owners:
            if not OWNER_VALUE.match(owner):
                report.errors.append(
                    f"{where} — bad @owner '{owner}'; use github:<login>, "
                    "github-team:<org>/<team>, or TBD"
                )
                continue
            if owner != "TBD":
                continue
            deadline = dates.get("@owner-decision-by")
            if deadline is None:
                report.errors.append(
                    f"{where} — @owner:TBD needs @owner-decision-by:YYYY-MM-DD"
                )
            elif deadline < today():
                report.errors.append(
                    f"{where} — @owner:TBD expired on {deadline.isoformat()}; "
                    "name an owner"
                )


def check_status_freshness(doc: Path, report: Report, items: list[Item]) -> None:
    """A cached cross-repo status must be attributed, dated, and expirable."""
    for item in items:
        where = f"{doc.name}:{item.lineno}"
        present = [tag for tag in FRESHNESS_TAGS if tag_values(item.block, tag)]
        if present and len(present) < len(FRESHNESS_TAGS):
            missing = ", ".join(tag for tag in FRESHNESS_TAGS if tag not in present)
            report.errors.append(
                f"{where} — incomplete status metadata, missing: {missing}"
            )
            continue
        if not present:
            if item.is_open and BLOCKED_BY.search(item.block):
                report.warnings.append(
                    f"{where} — @blocked_by with no @source-ref/@observed-at/"
                    "@recheck-by; a cross-repo status is a cache, not a source"
                )
            continue
        dates = check_dates(where, item.block, report)
        recheck = dates.get("@recheck-by")
        if recheck is not None and recheck < today():
            report.errors.append(
                f"{where} — status went stale on {recheck.isoformat()}; "
                "re-check against the owner and update @observed-at"
            )


def check_owner_coverage(doc: Path, report: Report, items: list[Item]) -> None:
    """Count open items with no owner (one line, not one per item)."""
    unowned = [
        item for item in items if item.is_open and not OWNER_TAG.search(item.block)
    ]
    if unowned:
        report.warnings.append(
            f"{doc.name} — {len(unowned)} open item(s) without @owner "
            f"(first at line {unowned[0].lineno}); backfill pending"
        )


def check_file(doc: Path, report: Report) -> None:
    """Run every check over one plan file."""
    text = doc.read_text(encoding="utf-8")
    items = iter_items(text)
    check_citations(doc, text, report)
    check_blockers(doc, text, report)
    check_status_pointer(doc, text, report)
    check_owner_form(doc, report, items)
    check_status_freshness(doc, report, items)
    check_owner_coverage(doc, report, items)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "files", nargs="*", help="plan files (default: TODO.md CLAUDE.md)"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    parser.add_argument(
        "--print-owners",
        action="store_true",
        help="list well-formed owner handles (for the CI existence check) and exit",
    )
    args = parser.parse_args(argv)

    targets = [Path(name) for name in args.files] or [
        REPO_ROOT / name for name in DEFAULT_FILES
    ]

    if args.print_owners:
        for doc in targets:
            if doc.exists():
                for handle in collect_owners(iter_items(doc.read_text("utf-8"))):
                    print(handle)
        return 0
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
