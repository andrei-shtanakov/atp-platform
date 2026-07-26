"""Package-direction guard: the dashboard must not reach into atp-platform.

`atp-dashboard` declares `atp-core` as its only ATP dependency. Evaluator
implementations, the runner and the CLI live in `atp-platform`. Importing them
from the dashboard works in this monorepo — everything is installed together —
which is exactly why the breach went unnoticed: `upload.py` pulled in
`EvaluatorRegistry` inside a function, wrapped in `try/except`, so it neither
failed at import time nor showed up in a top-of-file import scan.

Parsing with `ast` rather than grepping imports is the point: a function-local
import is still a dependency, and it is the form a violation takes when
someone works around a declared boundary.

When benchmark scoring gains real evaluators, they must arrive by injection
from the composition root — never by the dashboard importing them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "packages" / "atp-dashboard"
CORE = REPO_ROOT / "packages" / "atp-core"

#: Top-level `atp.*` modules that belong to atp-platform, not atp-core.
PLATFORM_ONLY = {
    "atp.evaluators",
    "atp.runner",
    "atp.cli",
    "atp.reporters",
    "atp.baseline",
    "atp.generator",
    "atp.tui",
}

#: Pre-existing breaches, recorded rather than hidden. Running this guard for
#: the first time (2026-07-26) turned up three `atp.generator` imports nobody
#: had noticed — suite generation reached from dashboard routes. They are out
#: of scope for the evaluation work that motivated the guard, and weakening
#: the rule to hide them would defeat its purpose, so they are listed here
#: with an expiry instead. Shrink this set; never grow it.
KNOWN_VIOLATIONS: dict[str, set[str]] = {
    "atp/dashboard/v2/routes/definitions.py": {"atp.generator"},
    "atp/dashboard/v2/routes/templates.py": {"atp.generator"},
    "atp/dashboard/v2/services/export_service.py": {"atp.generator"},
}


def _python_files(root: Path) -> list[Path]:
    """Every python source file under root, excluding caches."""
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imported_modules(tree: ast.AST) -> set[str]:
    """Module names imported anywhere in the tree, including inside functions."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports cannot cross a package boundary.
            if node.level == 0 and node.module:
                modules.add(node.module)
    return modules


def _violations(path: Path) -> list[str]:
    """Platform-only modules imported by this file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for module in sorted(_imported_modules(tree)):
        for forbidden in PLATFORM_ONLY:
            if module == forbidden or module.startswith(f"{forbidden}."):
                found.append(module)
    return found


@pytest.mark.parametrize(
    "source", _python_files(DASHBOARD), ids=lambda p: str(p.relative_to(DASHBOARD))
)
def test_dashboard_does_not_import_platform_modules(source: Path) -> None:
    """atp-dashboard depends on atp-core only."""
    relative = str(source.relative_to(DASHBOARD))
    allowed = KNOWN_VIOLATIONS.get(relative, set())
    found = [
        module
        for module in _violations(source)
        if not any(module.startswith(prefix) for prefix in allowed)
    ]
    assert not found, (
        f"{source.relative_to(REPO_ROOT)} imports {found}, which lives in "
        "atp-platform. Inject it from the composition root, or move the "
        "neutral part of it into atp-core."
    )


@pytest.mark.parametrize(
    "source", _python_files(CORE), ids=lambda p: str(p.relative_to(CORE))
)
def test_core_does_not_import_platform_modules(source: Path) -> None:
    """atp-core is the bottom of the graph and may depend on nothing above it.

    Added after the dashboard's "imports without atp-platform" test started
    actually blocking: it failed, and the culprit was not the dashboard at all
    but `atp/scoring/aggregator.py` importing `EvalResult` from
    `atp.evaluators.base` -- a re-export of a class atp-core already owns. The
    guard scanned only the dashboard, so the deeper violation was invisible.
    """
    found = _violations(source)
    assert not found, (
        f"{source.relative_to(REPO_ROOT)} imports {found} from atp-platform. "
        "atp-core sits below it; move the needed piece down or invert the "
        "dependency."
    )


def test_known_violations_are_still_real() -> None:
    """Delete an entry once it is fixed; a stale exemption re-opens the door."""
    stale = [
        relative
        for relative in KNOWN_VIOLATIONS
        if not _violations(DASHBOARD / relative)
    ]
    assert not stale, f"fixed — remove from KNOWN_VIOLATIONS: {stale}"


def test_evaluator_implementations_are_never_exempt() -> None:
    """The breach this guard was built for admits no exemptions."""
    for modules in KNOWN_VIOLATIONS.values():
        assert not any(m.startswith("atp.evaluators") for m in modules)


class TestTheGuardItself:
    """A boundary test that cannot see a violation protects nothing."""

    def test_detects_a_module_level_import(self, tmp_path: Path) -> None:
        source = tmp_path / "m.py"
        source.write_text("from atp.evaluators import EvaluatorRegistry\n")
        assert _violations(source) == ["atp.evaluators"]

    def test_detects_a_function_local_import(self, tmp_path: Path) -> None:
        """The form the real violation took."""
        source = tmp_path / "m.py"
        source.write_text(
            "def f():\n"
            "    try:\n"
            "        from atp.evaluators import EvaluatorRegistry\n"
            "    except Exception:\n"
            "        return None\n"
        )
        assert _violations(source) == ["atp.evaluators"]

    def test_detects_a_submodule_import(self, tmp_path: Path) -> None:
        source = tmp_path / "m.py"
        source.write_text("import atp.runner.sandbox\n")
        assert _violations(source) == ["atp.runner.sandbox"]

    def test_allows_core_modules(self, tmp_path: Path) -> None:
        source = tmp_path / "m.py"
        source.write_text(
            "from atp.evaluation.vocabulary import known_assertion_types\n"
            "from atp.core.results import EvalResult\n"
            "from atp.scoring import ScoreAggregator\n"
        )
        assert _violations(source) == []

    def test_does_not_confuse_a_prefix_match(self, tmp_path: Path) -> None:
        """`atp.evaluation` is core; `atp.evaluators` is not."""
        source = tmp_path / "m.py"
        source.write_text("import atp.evaluation\n")
        assert _violations(source) == []
