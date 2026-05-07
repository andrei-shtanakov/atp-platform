"""Static checks that release artefacts (CHANGELOG, migration guides, per-package
CHANGELOGs) are present and structurally well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"
MIGRATIONS_DIR = REPO_ROOT / "docs" / "migrations"

EXPECTED_MIGRATIONS = {
    "2026-04-el-farol-intervals.md",
    "2026-05-el-farol-scoring.md",
    "2026-04-mcp-purpose-gating.md",
    "2026-04-legacy-agents-endpoint.md",
}

EXPECTED_HEADINGS = {
    "## What changed",
    "## Before",
    "## After",
    "## How to migrate",
}

PACKAGE_CHANGELOGS = [
    REPO_ROOT / "packages" / "atp-core" / "CHANGELOG.md",
    REPO_ROOT / "packages" / "atp-adapters" / "CHANGELOG.md",
    REPO_ROOT / "packages" / "atp-dashboard" / "CHANGELOG.md",
    REPO_ROOT / "packages" / "atp-sdk" / "CHANGELOG.md",
]


def test_root_changelog_has_unreleased_and_2_0_0() -> None:
    text = ROOT_CHANGELOG.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert "## [2.0.0]" in text
    assert "## [1.0.0]" in text


def test_root_changelog_linkbacks_use_correct_org() -> None:
    text = ROOT_CHANGELOG.read_text(encoding="utf-8")
    assert "github.com/andrei-shtanakov/atp-platform" in text, (
        "CHANGELOG.md linkbacks must use the andrei-shtanakov org "
        "(matching `git remote get-url origin`)."
    )
    assert "github.com/anthropics" not in text, (
        "CHANGELOG.md still references the stale 'anthropics' org "
        "in linkbacks; update to andrei-shtanakov."
    )


def test_migrations_dir_exists() -> None:
    assert MIGRATIONS_DIR.is_dir(), f"{MIGRATIONS_DIR} must exist"


@pytest.mark.parametrize("name", sorted(EXPECTED_MIGRATIONS))
def test_migration_guide_present_and_well_formed(name: str) -> None:
    path = MIGRATIONS_DIR / name
    assert path.is_file(), f"missing migration guide: {path}"
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    assert line_count >= 30, f"{path} is only {line_count} lines; expected >= 30"
    for heading in EXPECTED_HEADINGS:
        assert heading in text, f"{path} missing heading {heading!r}"


def test_migration_guide_no_unfilled_placeholders() -> None:
    """Catch the MCP guide if it ships with `<placeholder>` markers."""
    for name in EXPECTED_MIGRATIONS:
        path = MIGRATIONS_DIR / name
        text = path.read_text(encoding="utf-8")
        # Allow Markdown emphasis/HTML like `<br>` or `</em>` if needed,
        # but reject anything matching `<lowercase-with-dashes>` which is
        # how the spec template marks placeholders.
        import re

        bad = re.findall(r"<[a-z][a-z0-9-]*(?:-[a-z0-9]+)*>", text)
        assert not bad, f"{path} still contains placeholder markers: {bad}"


@pytest.mark.parametrize("path", PACKAGE_CHANGELOGS, ids=lambda p: p.parent.name)
def test_package_changelog_exists_and_has_unreleased(path: Path) -> None:
    assert path.is_file(), f"missing package CHANGELOG: {path}"
    text = path.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text, f"{path} missing `## [Unreleased]`"
