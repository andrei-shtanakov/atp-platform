"""Contract tests: openprose.receipt.v1 verification over vendored fixtures.

Corpus runs and broken fixtures are pinned copies of upstream open-prose
(method/contract/openprose/PROVENANCE.md). Broken-fixture outcomes are
asserted through each fixture's own upstream expected.json, so a future
fixture refresh stays mechanical.
"""

import json
from pathlib import Path

import pytest

from atp.evaluators.openprose_receipts.reader import verify_run

_FIXTURES = (
    Path(__file__).parent.parent.parent
    / "method"
    / "contract"
    / "fixtures"
    / "openprose"
)
_CORPUS = sorted(p for p in (_FIXTURES / "runs").iterdir() if p.is_dir())
_BROKEN = sorted(p for p in (_FIXTURES / "broken").iterdir() if p.is_dir())


def test_fixture_tree_is_present() -> None:
    """Guard against a silently-empty parametrization."""
    assert len(_CORPUS) == 4
    assert len(_BROKEN) == 4


@pytest.mark.parametrize("run_dir", _CORPUS, ids=lambda p: p.name)
def test_corpus_run_verifies_clean(run_dir: Path) -> None:
    result = verify_run(run_dir)
    assert result.ok, [i.message for i in result.errors]
    assert result.errors == []
    assert result.warnings == []
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert result.receipt_count == manifest["receipt_count"]


@pytest.mark.parametrize("fixture_dir", _BROKEN, ids=lambda p: p.name)
def test_broken_fixture_matches_upstream_expectation(fixture_dir: Path) -> None:
    expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    result = verify_run(fixture_dir)
    assert result.ok == expected["ok"]
    if "error_contains" in expected:
        assert any(expected["error_contains"] in i.message for i in result.errors), [
            i.message for i in result.errors
        ]
    if "warning_contains" in expected:
        assert any(
            expected["warning_contains"] in i.message for i in result.warnings
        ), [i.message for i in result.warnings]
