"""Tests for the atp.plugins entry-point loader (hermetic — entry points stubbed)."""

from unittest.mock import patch

import atp.plugins.entrypoints as ep_mod


class _FakeEntryPoint:
    def __init__(self, name: str, hook) -> None:
        self.name = name
        self._hook = hook

    def load(self):
        return self._hook


def test_runs_each_register_hook() -> None:
    """Each discovered entry point's register hook is loaded and called."""
    calls: list[str] = []
    ep = _FakeEntryPoint("demo", lambda: calls.append("ran"))
    ep_mod._loaded = False
    with patch.object(ep_mod, "entry_points", return_value=[ep]) as mock_eps:
        result = ep_mod.load_entrypoint_plugins(force=True)
    mock_eps.assert_called_once_with(group="atp.plugins")
    assert result == ["demo"]
    assert calls == ["ran"]


def test_idempotent() -> None:
    """A second, non-forced call is a no-op once loaded."""
    ep_mod._loaded = False
    with patch.object(ep_mod, "entry_points", return_value=[]):
        assert ep_mod.load_entrypoint_plugins(force=True) == []
        assert ep_mod.load_entrypoint_plugins() == []


def test_broken_hook_is_skipped() -> None:
    """A register hook that raises is logged and skipped, not fatal."""

    def boom() -> None:
        raise RuntimeError("bad plugin")

    ep = _FakeEntryPoint("broken", boom)
    ep_mod._loaded = False
    with patch.object(ep_mod, "entry_points", return_value=[ep]):
        result = ep_mod.load_entrypoint_plugins(force=True)
    assert result == []
