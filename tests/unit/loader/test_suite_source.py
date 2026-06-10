"""Unit tests for the suite-source registry (alternate source formats)."""

from pathlib import Path

from atp.loader.models import TestDefinition, TestSuite
from atp.loader.suite_source import SuiteSourceRegistry, get_suite_source_registry


def _suite() -> TestSuite:
    return TestSuite(
        test_suite="s",
        tests=[TestDefinition(id="t1", name="t1", task={"description": "x"})],
    )


def test_empty_returns_none(tmp_path: Path) -> None:
    reg = SuiteSourceRegistry()
    f = tmp_path / "s.yaml"
    f.write_text("x")
    assert reg.find_loader(f) is None


def test_register_and_find(tmp_path: Path) -> None:
    reg = SuiteSourceRegistry()
    loader = lambda p: _suite()  # noqa: E731
    reg.register("demo", lambda p: True, loader)
    f = tmp_path / "s.yaml"
    f.write_text("x")
    assert reg.find_loader(f) is loader


def test_detector_exception_skipped(tmp_path: Path) -> None:
    reg = SuiteSourceRegistry()

    def boom(_p: Path) -> bool:
        raise ValueError("bad")

    loader = lambda p: _suite()  # noqa: E731
    reg.register("boom", boom, loader)
    reg.register("ok", lambda p: True, loader)
    f = tmp_path / "s.yaml"
    f.write_text("x")
    assert reg.find_loader(f) is loader


def test_singleton() -> None:
    assert get_suite_source_registry() is get_suite_source_registry()
