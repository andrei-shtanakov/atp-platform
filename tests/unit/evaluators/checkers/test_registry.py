"""Tests for the deterministic checker registry (Phase A-1)."""

import pytest

from atp.core.results import CaseVerdict
from atp.evaluators.checkers.registry import (
    get_checker,
    list_checkers,
    register_checker,
)


def _dummy(config: dict, text: str | None) -> CaseVerdict:
    return CaseVerdict(critical_pass=True, grader_version="dummy@1")


def test_register_and_get() -> None:
    register_checker("dummy", _dummy)
    fn = get_checker("dummy")
    assert fn is not None
    assert fn({}, None).grader_version == "dummy@1"
    assert "dummy" in list_checkers()


def test_unknown_checker_returns_none() -> None:
    assert get_checker("does-not-exist") is None


def test_register_rejects_duplicate() -> None:
    register_checker("dup", _dummy)
    with pytest.raises(ValueError, match="already registered"):
        register_checker("dup", _dummy)
