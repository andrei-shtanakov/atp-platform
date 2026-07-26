"""Contract tests for the benchmark-plane score semantics.

These lock the promises made to an external consumer (Maestro's
`finalize()`): the number is always accompanied by a versioned statement of
what it means, and the statement says plainly that it is completion rather
than quality. Breaking any of these is a wire-contract change, not a
refactor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atp.dashboard.benchmark.schemas import RunResponse, RunStatusResponse
from atp.dashboard.benchmark.score_contract import (
    SCORE_CONTRACT_VERSION,
    empty_score_components,
    run_score_semantics,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "benchmark_score_contract"


def _load(name: str) -> dict[str, Any]:
    """Read a published contract fixture."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _status(**overrides: Any) -> RunStatusResponse:
    """A minimal run-status response."""
    payload: dict[str, Any] = {
        "id": 1,
        "status": "completed",
        "current_task_index": 1,
        "tasks_count": 1,
        "total_score": 100.0,
        "completed_tasks": [],
    }
    payload.update(overrides)
    return RunStatusResponse(**payload)


class TestSemanticsArePresentAndHonest:
    """The number never travels bare, and never claims to be quality."""

    def test_status_response_carries_semantics_and_components(self) -> None:
        dumped = _status().model_dump()
        assert dumped["score_semantics"]["schema_version"] == SCORE_CONTRACT_VERSION
        assert dumped["score_components"] == {}

    def test_run_response_carries_them_too(self) -> None:
        """An unqualified score anywhere reproduces the defect."""
        dumped = RunResponse(
            id=1,
            benchmark_id=1,
            agent_name="a",
            adapter_type="sdk",
            status="completed",
            current_task_index=1,
            total_score=100.0,
            started_at="2026-07-26T00:00:00",
            finished_at=None,
        ).model_dump()
        assert dumped["score_semantics"]["schema_version"] == SCORE_CONTRACT_VERSION

    def test_current_score_is_declared_not_a_quality_signal(self) -> None:
        semantics = run_score_semantics()
        assert semantics["quality_signal"] is False
        assert semantics["kind"] == "completion_rate"

    def test_the_two_levels_are_described_separately(self) -> None:
        """Run-level percentage and per-task boolean are different quantities."""
        semantics = run_score_semantics()
        assert semantics["level"] == "run"
        assert semantics["unit"] == "percent"
        assert semantics["range"] == {"min": 0.0, "max": 100.0}
        assert semantics["aggregation"] == {"function": "mean", "over": "task_score"}
        assert semantics["task_score"]["level"] == "task"
        assert semantics["task_score"]["values"] == [0.0, 100.0]

    @pytest.mark.parametrize("caveat", ["null_until_finalized", "zero_is_ambiguous"])
    def test_known_traps_are_stated_on_the_wire(self, caveat: str) -> None:
        """A 0.0 from an empty run must not silently read as total failure."""
        assert any(c.startswith(caveat) for c in run_score_semantics()["caveats"])

    def test_semantics_are_not_shared_mutable_state(self) -> None:
        first = _status().model_dump()
        first["score_semantics"]["kind"] = "mutated"
        assert _status().model_dump()["score_semantics"]["kind"] == "completion_rate"

    def test_components_default_is_a_fresh_map(self) -> None:
        components = empty_score_components()
        components["x"] = 1
        assert empty_score_components() == {}


class TestPublishedFixtures:
    """The fixtures Maestro pins must keep matching what we serialize."""

    def test_completion_only_fixture_matches_current_serialization(self) -> None:
        published = _load("run_status_completion_only.json")
        live = _status(
            id=42,
            current_task_index=3,
            tasks_count=3,
            total_score=66.66666666666667,
            completed_tasks=published["completed_tasks"],
        ).model_dump()
        assert live == published

    def test_forward_compat_fixture_parses_into_the_model(self) -> None:
        """Unknown components and unknown semantics keys must not break parsing."""
        future = _load("run_status_forward_compat.json")
        parsed = RunStatusResponse(**future)
        assert "some_future_axis_atp_has_not_invented_yet" in parsed.score_components
        assert parsed.score_semantics["some_future_key"] == "consumers must ignore this"

    def test_forward_compat_fixture_actually_exercises_the_future(self) -> None:
        """Guard against the fixture silently degrading to today's payload."""
        future = _load("run_status_forward_compat.json")
        assert future["score_components"] != {}
        assert future["score_semantics"]["quality_signal"] is True
        assert future["score_semantics"]["kind"] != run_score_semantics()["kind"]


class TestNoNewPersistence:
    """The contract is an API view; it must not have grown a column."""

    def test_run_model_has_no_score_component_columns(self) -> None:
        from atp.dashboard.benchmark.models import Run

        columns = set(Run.__table__.columns.keys())
        assert "score_components" not in columns
        assert "score_semantics" not in columns
