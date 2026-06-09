"""Tests for the atp.plugins entry-point loader."""

from atp.plugins.entrypoints import load_entrypoint_plugins


def test_load_is_idempotent() -> None:
    """First (forced) load runs hooks; a subsequent non-forced call is a no-op."""
    first = load_entrypoint_plugins(force=True)
    assert isinstance(first, list)
    second = load_entrypoint_plugins()  # already loaded → no-op
    assert second == []


def test_force_reruns() -> None:
    """force=True re-runs the hooks and returns the registered names."""
    result = load_entrypoint_plugins(force=True)
    assert isinstance(result, list)
