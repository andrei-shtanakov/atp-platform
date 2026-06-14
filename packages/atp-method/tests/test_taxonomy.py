"""Tests for the task_type <-> benchmark_id taxonomy registry (Phase A-2)."""

import pytest

from atp_method.taxonomy import TASK_TYPE_TO_BENCHMARK_ID, benchmark_id_for


def test_review_maps_to_code_review() -> None:
    assert benchmark_id_for("review") == "code-review"


def test_registry_is_the_source() -> None:
    assert TASK_TYPE_TO_BENCHMARK_ID["review"] == "code-review"


def test_unknown_task_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown task_type"):
        benchmark_id_for("does-not-exist")
