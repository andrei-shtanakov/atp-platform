"""Test the CHANGELOG-gate script via subprocess against fixture git repos."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_changelog.sh"


def _git(repo: Path, *args: str) -> str:
    """Run a git command in repo and return stdout (raises on failure)."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    """Init a fresh repo with one initial commit and a baseline CHANGELOG.md."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n",
        encoding="utf-8",
    )
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run_script(repo: Path, base: str, head: str, labels: list[str]) -> int:
    env = {
        **os.environ,
        "BASE_SHA": base,
        "HEAD_SHA": head,
        "PR_LABELS": json.dumps(labels),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode


def _commit_changelog(repo: Path, new_text: str, msg: str) -> str:
    (repo / "CHANGELOG.md").write_text(new_text, encoding="utf-8")
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _commit_other(repo: Path, msg: str = "other change") -> str:
    other = repo / "other.txt"
    other.write_text((other.read_text() if other.exists() else "") + "x\n")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    return _init_repo(tmp_path)


def test_no_relevant_label_passes_with_no_changelog_diff(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, []) == 0


def test_chore_label_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, ["chore"]) == 0


def test_bug_label_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, ["bug"]) == 0


def test_feat_label_without_changelog_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit_other(repo)
    assert _run_script(repo, base, head, ["feat"]) == 1


def test_breaking_label_with_unreleased_addition_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "- new breaking bullet\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add breaking bullet")
    assert _run_script(repo, base, head, ["breaking"]) == 0


def test_feat_label_with_unreleased_addition_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "- new feat bullet\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add feat bullet")
    assert _run_script(repo, base, head, ["feat"]) == 0


def test_feat_label_editing_only_1_0_0_section_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
        "- typo fix in 1.0.0\n"
    )
    head = _commit_changelog(repo, new, "fix typo in 1.0.0")
    assert _run_script(repo, base, head, ["feat"]) == 1


def test_both_labels_with_unreleased_addition_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "- combined bullet\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add combined bullet")
    assert _run_script(repo, base, head, ["feat", "breaking"]) == 0


def test_feat_label_addition_under_2_0_0_only_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n"
        "- new bullet that landed in 2.0.0 section\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet\n"
    )
    head = _commit_changelog(repo, new, "add bullet in wrong section")
    assert _run_script(repo, base, head, ["feat"]) == 1


def test_feat_label_typo_fix_in_1_0_0_only_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    new = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [2.0.0] - 2026-05-07\n\n"
        "- existing 2.0.0 bullet\n\n"
        "## [1.0.0] - 2026-02-13\n\n"
        "- existing 1.0.0 bullet typo-fixed\n"
    )
    head = _commit_changelog(repo, new, "typo fix in 1.0.0")
    assert _run_script(repo, base, head, ["feat"]) == 1
