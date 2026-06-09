"""Unit tests for the suite format-dispatch registry."""

from pathlib import Path

from atp.loader.format_dispatch import SuiteFormatRegistry, get_suite_format_registry


async def _handler(**_kwargs: object) -> bool:
    return True


def test_empty_registry_returns_none(tmp_path: Path) -> None:
    """No registered formats → find_handler returns None."""
    reg = SuiteFormatRegistry()
    f = tmp_path / "s.yaml"
    f.write_text("type: native")
    assert reg.find_handler(f) is None


def test_register_and_find(tmp_path: Path) -> None:
    """A matching detector routes to its handler."""
    reg = SuiteFormatRegistry()
    reg.register("demo", lambda p: p.read_text().strip() == "type: demo", _handler)
    f = tmp_path / "s.yaml"
    f.write_text("type: demo")
    assert reg.find_handler(f) is _handler


def test_no_match_returns_none(tmp_path: Path) -> None:
    """A non-matching file returns None even with formats registered."""
    reg = SuiteFormatRegistry()
    reg.register("demo", lambda p: False, _handler)
    f = tmp_path / "s.yaml"
    f.write_text("type: other")
    assert reg.find_handler(f) is None


def test_register_override_by_name(tmp_path: Path) -> None:
    """Re-registering a name replaces the previous entry (plugin override)."""
    reg = SuiteFormatRegistry()

    async def _other(**_kwargs: object) -> bool:
        return False

    reg.register("demo", lambda p: True, _handler)
    reg.register("demo", lambda p: True, _other)
    f = tmp_path / "s.yaml"
    f.write_text("x")
    assert reg.find_handler(f) is _other
    assert reg.names() == ["demo"]


def test_detector_exception_is_skipped(tmp_path: Path) -> None:
    """A detector that raises is skipped, not fatal."""
    reg = SuiteFormatRegistry()

    def _boom(_p: Path) -> bool:
        raise ValueError("bad file")

    reg.register("boom", _boom, _handler)
    reg.register("ok", lambda p: True, _handler)
    f = tmp_path / "s.yaml"
    f.write_text("x")
    assert reg.find_handler(f) is _handler


def test_singleton_accessor() -> None:
    """get_suite_format_registry returns a stable singleton."""
    assert get_suite_format_registry() is get_suite_format_registry()
