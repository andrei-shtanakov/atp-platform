"""Test the plan-citation checker by importing it and driving it on fixtures."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_plan_citations.py"


def _load_module() -> ModuleType:
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
def checker(repo: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
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

    def test_valid_citation_passes(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        _write(repo / "src/mod.py", "a\nb\nc\n")
        doc = _write(repo / "TODO.md", "see `src/mod.py:2` for detail\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert report.errors == []

    def test_missing_file_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "see `src/gone.py:2`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert len(report.errors) == 1
        assert "no such file" in report.errors[0]

    def test_line_past_end_of_file_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        _write(repo / "src/mod.py", "a\nb\n")
        doc = _write(repo / "TODO.md", "see `src/mod.py:99`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "out of range" in report.errors[0]

    def test_range_end_past_end_of_file_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        _write(repo / "src/mod.py", "a\nb\n")
        doc = _write(repo / "TODO.md", "see `src/mod.py:1-9`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "out of range" in report.errors[0]

    def test_absent_sibling_is_skipped_not_failed(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """A clone without the dev workspace must stay green."""
        doc = _write(repo / "TODO.md", "see `../maestro/TODO.md:12`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert report.errors == []
        assert len(report.skipped) == 1

    @pytest.mark.parametrize("spec", ["0", "9-1", "2,0"])
    def test_impossible_line_spec_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path, spec: str
    ) -> None:
        """Line 0 and reversed ranges cannot describe real lines."""
        _write(repo / "src/mod.py", "a\nb\nc\n" * 10)
        doc = _write(repo / "TODO.md", f"see `src/mod.py:{spec}`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "invalid line spec" in report.errors[0]

    def test_citation_escaping_the_workspace_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """A citation must not send the hook outside the workspace.

        Note the extension: the citation regex requires one, so a bare
        `/etc/hosts` never parses as a citation in the first place.
        """
        doc = _write(repo / "TODO.md", "see `../../../outside/secret.txt:1`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "escapes the workspace" in report.errors[0]

    def test_undecodable_target_is_reported_not_raised(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """A binary target must produce a finding, not a traceback."""
        blob = repo / "src/blob.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"\xff\xfe\x00\x01")
        doc = _write(repo / "TODO.md", "see `src/blob.bin:1`\n")
        report = checker.Report()
        checker.check_citations(doc, doc.read_text(), report)
        assert "cannot read" in report.errors[0]


class TestBlockerFreshness:
    """Check 2 — @blocked_by targets that already shipped upstream."""

    def test_open_blocker_is_quiet(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        _write(tmp_path / "maestro/TODO.md", "- [ ] **R-03** MCP client\n")
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-03\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert report.warnings == []

    def test_closed_blocker_warns(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """The exact regression from PR #263."""
        _write(tmp_path / "maestro/TODO.md", "- [x] **R-03** MCP client\n")
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-03\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert len(report.warnings) == 1
        assert "looks closed" in report.warnings[0]

    def test_unknown_slug_warns(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        _write(tmp_path / "maestro/TODO.md", "- [ ] something else\n")
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-99\n")
        report = checker.Report()
        checker.check_blockers(doc, doc.read_text(), report)
        assert "not found" in report.warnings[0]

    def test_absent_sibling_is_skipped(
        self, checker: ModuleType, repo: Path, tmp_path: Path
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

    def test_stale_pointer_warns(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        self._seed(tmp_path, "2026-04-10-status.md", "2026-07-08-status.md")
        doc = _write(
            repo / "TODO.md",
            "status: `../prograph-vault/authored/notes/status/2026-04-10-status.md`\n",
        )
        report = checker.Report()
        checker.check_status_pointer(doc, doc.read_text(), report)
        assert "newest vault status is 2026-07-08" in report.warnings[0]

    def test_same_day_variant_is_not_a_defect(
        self, checker: ModuleType, repo: Path, tmp_path: Path
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

    def test_absent_vault_is_skipped(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
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
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        report = checker.Report(warnings=["something"])
        assert report.ok(strict=False)

    def test_warnings_fail_under_strict(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        report = checker.Report(warnings=["something"])
        assert not report.ok(strict=True)

    def test_errors_always_fail(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        report = checker.Report(errors=["boom"])
        assert not report.ok(strict=False)


class TestOwnerForm:
    """Check 4 — owner is an accountable handle, not a display name."""

    @pytest.mark.parametrize(
        "owner",
        ["github:andrei-shtanakov", "github-team:acme/platform"],
    )
    def test_valid_handles_pass(
        self, checker: ModuleType, repo: Path, tmp_path: Path, owner: str
    ) -> None:
        doc = _write(repo / "TODO.md", f"- [ ] thing @owner:{owner}\n")
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert report.errors == []

    @pytest.mark.parametrize(
        "owner", ["Andrei", "atp-platform", "github:", "team:acme/platform"]
    )
    def test_display_names_and_repos_are_rejected(
        self, checker: ModuleType, repo: Path, tmp_path: Path, owner: str
    ) -> None:
        """An owner is a person or team — never a repository, never prose."""
        doc = _write(repo / "TODO.md", f"- [ ] thing @owner:{owner}\n")
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert "bad @owner" in report.errors[0]

    def test_two_owners_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(
            repo / "TODO.md",
            "- [ ] thing @owner:github:a @owner:github:b\n",
        )
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert "exactly one" in report.errors[0]

    def test_tbd_requires_a_decision_date(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "- [ ] thing @owner:TBD\n")
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert "@owner-decision-by" in report.errors[0]

    def test_expired_tbd_fails(
        self,
        checker: ModuleType,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(checker, "today", lambda: date(2026, 7, 26))
        doc = _write(
            repo / "TODO.md",
            "- [ ] thing @owner:TBD @owner-decision-by:2026-07-01\n",
        )
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert "expired" in report.errors[0]

    def test_unexpired_tbd_passes(
        self,
        checker: ModuleType,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(checker, "today", lambda: date(2026, 7, 26))
        doc = _write(
            repo / "TODO.md",
            "- [ ] thing @owner:TBD @owner-decision-by:2026-09-01\n",
        )
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert report.errors == []

    def test_malformed_date_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(
            repo / "TODO.md",
            "- [ ] thing @owner:TBD @owner-decision-by:26-07-01\n",
        )
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert "YYYY-MM-DD" in report.errors[0]

    def test_tag_on_a_continuation_line_is_seen(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """Tags routinely sit under the checkbox line, not on it."""
        doc = _write(
            repo / "TODO.md",
            "- [ ] thing\n  more prose\n  @owner:not-a-handle\n",
        )
        report = checker.Report()
        checker.check_owner_form(doc, report, checker.iter_items(doc.read_text()))
        assert "bad @owner" in report.errors[0]


class TestStatusFreshness:
    """Check 5 — a cached cross-repo status must be attributable and expirable."""

    FULL = (
        "@source-owner:maestro @source-ref:maestro@07d408d "
        "@observed-at:2026-07-26 @recheck-by:2026-10-26"
    )

    def test_complete_and_current_metadata_passes(
        self,
        checker: ModuleType,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(checker, "today", lambda: date(2026, 7, 26))
        doc = _write(repo / "TODO.md", f"- [ ] thing {self.FULL}\n")
        report = checker.Report()
        checker.check_status_freshness(doc, report, checker.iter_items(doc.read_text()))
        assert report.errors == []

    def test_partial_metadata_is_an_error(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """Half a citation looks sourced without being re-checkable."""
        doc = _write(repo / "TODO.md", "- [ ] thing @source-owner:maestro\n")
        report = checker.Report()
        checker.check_status_freshness(doc, report, checker.iter_items(doc.read_text()))
        assert "incomplete status metadata" in report.errors[0]

    def test_expired_recheck_by_is_an_error(
        self,
        checker: ModuleType,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(checker, "today", lambda: date(2027, 1, 1))
        doc = _write(repo / "TODO.md", f"- [ ] thing {self.FULL}\n")
        report = checker.Report()
        checker.check_status_freshness(doc, report, checker.iter_items(doc.read_text()))
        assert "went stale" in report.errors[0]

    def test_blocked_by_without_metadata_warns(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "- [ ] thing @blocked_by:maestro#R-03\n")
        report = checker.Report()
        checker.check_status_freshness(doc, report, checker.iter_items(doc.read_text()))
        assert "cache, not a source" in report.warnings[0]

    def test_closed_item_without_metadata_is_quiet(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """A done item's blocker no longer needs a TTL."""
        doc = _write(repo / "TODO.md", "- [x] thing @blocked_by:maestro#R-03\n")
        report = checker.Report()
        checker.check_status_freshness(doc, report, checker.iter_items(doc.read_text()))
        assert report.warnings == []


class TestOwnerCoverage:
    """Check 6 — unowned open items counted once, not spammed per item."""

    def test_unowned_open_items_produce_one_warning(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "- [ ] a\n- [ ] b\n- [x] done\n")
        report = checker.Report()
        checker.check_owner_coverage(doc, report, checker.iter_items(doc.read_text()))
        assert len(report.warnings) == 1
        assert "2 open item(s) without @owner" in report.warnings[0]

    def test_fully_owned_file_is_quiet(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(repo / "TODO.md", "- [ ] a @owner:github:someone\n")
        report = checker.Report()
        checker.check_owner_coverage(doc, report, checker.iter_items(doc.read_text()))
        assert report.warnings == []


class TestOwnerCollection:
    """CI must resolve exactly the handles this grammar accepts."""

    def test_handles_are_emitted_in_a_kind_value_form(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(
            repo / "TODO.md",
            "- [ ] a @owner:github:andrei-shtanakov\n"
            "- [ ] b @owner:github-team:acme/platform\n",
        )
        handles = checker.collect_owners(checker.iter_items(doc.read_text()))
        assert handles == ["user andrei-shtanakov", "team acme/platform"]

    def test_underscored_handle_is_not_truncated_into_a_different_account(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """A shell character class stopped at `_` and validated `foo` instead."""
        doc = _write(repo / "TODO.md", "- [ ] a @owner:github:foo_bar\n")
        assert checker.collect_owners(checker.iter_items(doc.read_text())) == []

    def test_tbd_and_malformed_owners_are_not_sent_to_the_api(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(
            repo / "TODO.md",
            "- [ ] a @owner:TBD @owner-decision-by:2099-01-01\n- [ ] b @owner:Andrei\n",
        )
        assert checker.collect_owners(checker.iter_items(doc.read_text())) == []

    def test_duplicate_handles_are_emitted_once(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        doc = _write(
            repo / "TODO.md",
            "- [ ] a @owner:github:x\n- [ ] b @owner:github:x\n",
        )
        assert checker.collect_owners(checker.iter_items(doc.read_text())) == ["user x"]


class TestItemSplitting:
    """iter_items is a single linear pass over the file."""

    def test_blocks_carry_their_continuation_lines(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        items = checker.iter_items("- [ ] a\n  tail-a\n- [x] b\n  tail-b\n")
        assert [(i.lineno, i.is_open) for i in items] == [(1, True), (3, False)]
        assert "tail-a" in items[0].block and "tail-a" not in items[1].block

    def test_preamble_before_the_first_item_is_ignored(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        items = checker.iter_items("# Heading\nprose\n- [ ] a\n")
        assert len(items) == 1 and items[0].lineno == 3

    def test_scales_linearly_on_a_long_tail(
        self, checker: ModuleType, repo: Path, tmp_path: Path
    ) -> None:
        """One item followed by a long tail must not copy the tail repeatedly."""
        text = "- [ ] a\n" + "\n".join(f"  line {n}" for n in range(20_000))
        items = checker.iter_items(text)
        assert len(items) == 1
        assert items[0].block.count("\n") == 20_000
