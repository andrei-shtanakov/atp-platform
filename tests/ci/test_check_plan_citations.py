"""Test the plan-citation checker by importing it and driving it on fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_plan_citations.py"


def _load_module():
    """Import the script as a module (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("check_plan_citations", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Fake repo root, nested so siblings land in this test's own tmp_path.

    Rooting it at tmp_path directly would put siblings in tmp_path.parent,
    which pytest shares across the whole run — one test's fixtures would
    leak into the next.
    """
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def checker(repo: Path, monkeypatch: pytest.MonkeyPatch):
    """The checker module rooted at an isolated fake repo."""
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", repo)
    return module


def _write(path: Path, text: str) -> Path:
    """Write text to path, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestCitationLiveness:
    """Check 1 — every path:line citation resolves."""

    def test_valid_citation_passes(self, checker, repo: Path, tmp_path: Path) -> None:
        _write(repo / "src/mod.py", "a\nb\nc\n")
        doc = _write(repo / "TODO.md", "see `src/mod.py:2` for detail\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert report.errors == []

    def test_missing_file_is_an_error(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "see `src/gone.py:2`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert len(report.errors) == 1
        assert "no such file" in report.errors[0]

    def test_line_past_end_of_file_is_an_error(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        _write(repo / "src/mod.py", "a\nb\n")
        doc = _write(repo / "TODO.md", "see `src/mod.py:99`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "out of range" in report.errors[0]

    def test_range_end_past_end_of_file_is_an_error(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        _write(repo / "src/mod.py", "a\nb\n")
        doc = _write(repo / "TODO.md", "see `src/mod.py:1-9`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "out of range" in report.errors[0]

    def test_absent_sibling_is_skipped_not_failed(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        """A clone without the dev workspace must stay green."""
        doc = _write(repo / "TODO.md", "see `../maestro/TODO.md:12`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert report.errors == []
        assert len(report.skipped) == 1


class TestBlockerFreshness:
    """Check 2 — @blocked_by targets that already shipped upstream."""

    def test_open_blocker_is_quiet(self, checker, repo: Path, tmp_path: Path) -> None:
        _write(tmp_path / "maestro/TODO.md", "- [ ] **R-03** MCP client\n")
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-03\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert report.warnings == []

    def test_closed_blocker_warns(self, checker, repo: Path, tmp_path: Path) -> None:
        """The exact regression from PR #263."""
        _write(tmp_path / "maestro/TODO.md", "- [x] **R-03** MCP client\n")
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-03\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert len(report.warnings) == 1
        assert "looks closed" in report.warnings[0]

    def test_unknown_slug_warns(self, checker, repo: Path, tmp_path: Path) -> None:
        _write(tmp_path / "maestro/TODO.md", "- [ ] something else\n")
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-99\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert "not found" in report.warnings[0]

    def test_absent_sibling_is_skipped(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:nowhere#X\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert report.warnings == []
        assert len(report.skipped) == 1


class TestStatusPointer:
    """Check 3 — cited vault status note older than the newest."""

    def _seed(self, tmp_path: Path, *names: str) -> None:
        for name in names:
            _write(tmp_path / "prograph-vault/authored/notes/status" / name, "x")

    def test_stale_pointer_warns(self, checker, repo: Path, tmp_path: Path) -> None:
        self._seed(tmp_path, "2026-04-10-status.md", "2026-07-08-status.md")
        doc = _write(
            repo / "TODO.md",
            "status: `../prograph-vault/authored/notes/status/2026-04-10-status.md`\n",
        )
        report = checker.Report()
        checker.check_status_pointer(doc, doc.read_text(), report)
        assert "newest vault status is 2026-07-08" in report.warnings[0]

    def test_same_day_variant_is_not_a_defect(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        """Picking between same-day notes is editorial, not stale."""
        self._seed(tmp_path, "2026-07-08-status.md", "2026-07-08-1228-status.md")
        doc = _write(
            repo / "TODO.md",
            "s: `../prograph-vault/authored/notes/status/2026-07-08-1228-status.md`\n",
        )
        report = checker.Report()
        checker.check_status_pointer(doc, doc.read_text(), report)
        assert report.warnings == []

    def test_absent_vault_is_skipped(self, checker, repo: Path, tmp_path: Path) -> None:
        doc = _write(
            repo / "TODO.md",
            "s: `../prograph-vault/authored/notes/status/2026-04-10-status.md`\n",
        )
        report = checker.Report()
        checker.check_status_pointer(doc, doc.read_text(), report)
        assert report.warnings == []
        assert len(report.skipped) == 1


class TestExitCodes:
    """The strict flag decides whether warnings block."""

    def test_warnings_pass_by_default(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        report = checker.Report(warnings=["something"])
        assert report.ok(strict=False)

    def test_warnings_fail_under_strict(
        self, checker, repo: Path, tmp_path: Path
    ) -> None:
        report = checker.Report(warnings=["something"])
        assert not report.ok(strict=True)

    def test_errors_always_fail(self, checker, repo: Path, tmp_path: Path) -> None:
        report = checker.Report(errors=["boom"])
        assert not report.ok(strict=False)
