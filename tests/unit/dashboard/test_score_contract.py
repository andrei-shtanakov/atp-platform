"""Contract tests for the benchmark-plane score semantics.

These lock the promises made to an external consumer (Maestro's
`finalize()`): the number is always accompanied by a versioned statement of
what it means, and the statement says plainly that it is completion rather
than quality. Breaking any of these is a wire-contract change, not a
refactor.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from atp.dashboard.benchmark.schemas import RunResponse, RunStatusResponse
from atp.dashboard.benchmark.score_contract import (
    AGGREGATED_EVALUATION,
    COMPLETION_RATE,
    SCORE_CONTRACT_VERSION,
    empty_score_components,
    run_score_semantics,
)
from atp.dashboard.benchmark.scoring import (
    EVALUATION_RECORD_VERSION,
    RecordStatus,
    derive_run_score_view,
)

#: One applied and one withheld assertion, as `submit` stores them.
EVALUATED_RECORDS: list[dict[str, Any]] = [
    {
        "record_version": EVALUATION_RECORD_VERSION,
        "assertion_type": "contains",
        "status": RecordStatus.APPLIED,
        "evaluator": "artifact",
        "score": 0.5,
        "passed": False,
        "critical": False,
        "checks": [{"name": "contains", "passed": False, "score": 0.5}],
    },
    {
        "record_version": EVALUATION_RECORD_VERSION,
        "assertion_type": "pytest",
        "status": RecordStatus.SKIPPED,
        "reason": "not_allowed_by_policy",
        "detail": "policy 'untrusted_submission' does not permit this evaluator",
    },
]

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "benchmark_score_contract"
#: The handoff a consumer vendors against; its pins are asserted below.
DOC = Path(__file__).resolve().parents[3] / "docs" / "maestro-score-contract-handoff.md"

#: The fixture whose `score_semantics` block is the canonical one. It is the
#: payload of a real completion-only run, `coverage` included — not the bare
#: `run_score_semantics()` default, which omits coverage and hashes to
#: something else.
CANONICAL_SOURCE = "run_status_completion_only.json"


def _load(name: str) -> dict[str, Any]:
    """Read a published contract fixture."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _fixture_digest(name: str) -> str:
    """sha256 of a fixture's bytes, exactly as a consumer vendors them."""
    return hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


def canonical_semantics_digest() -> str:
    """sha256 of the canonical `score_semantics`, per the documented recipe.

    JSON with sorted keys and no whitespace, over the block published in
    `CANONICAL_SOURCE`. Kept as one function so the document and the test can
    never disagree about which bytes are being hashed.
    """
    canonical = json.dumps(
        _load(CANONICAL_SOURCE)["score_semantics"],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _status(
    task_records: list[list[dict[str, Any]] | None] | None = None,
    **overrides: Any,
) -> RunStatusResponse:
    """A run-status response built the way the route builds it.

    Going through `derive_run_score_view` rather than hand-writing the map is
    the point: a fixture assembled by different code than the endpoint uses is
    a fixture that can stay green while the endpoint changes.
    """
    semantics, components = derive_run_score_view(
        task_records if task_records is not None else [],
        overrides.get("tasks_count", 1),
    )
    payload: dict[str, Any] = {
        "id": 1,
        "status": "completed",
        "current_task_index": 1,
        "tasks_count": 1,
        "total_score": 100.0,
        "score_semantics": semantics,
        "score_components": components,
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
            score_semantics=run_score_semantics(),
            score_components=empty_score_components(),
            started_at="2026-07-26T00:00:00",
            finished_at=None,
        ).model_dump()
        assert dumped["score_semantics"]["schema_version"] == SCORE_CONTRACT_VERSION

    @pytest.mark.parametrize("field", ["score_semantics", "score_components"])
    def test_the_fields_have_no_default(self, field: str) -> None:
        """A default would let a call site publish a label it never derived.

        That is how the number gets a confident description for free — the one
        failure mode this whole contract exists to prevent.
        """
        assert RunStatusResponse.model_fields[field].is_required()
        assert RunResponse.model_fields[field].is_required()

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

    def test_an_evaluated_run_is_labelled_as_evaluation(self) -> None:
        """The other half of the contract: a real evaluation says so."""
        dumped = _status([EVALUATED_RECORDS]).model_dump()
        assert dumped["score_semantics"]["kind"] == "aggregated_evaluation"
        assert dumped["score_semantics"]["quality_signal"] is True
        assert dumped["score_components"] == {"contains": 50.0}

    def test_components_default_is_a_fresh_map(self) -> None:
        components = empty_score_components()
        components["x"] = 1
        assert empty_score_components() == {}


class TestPublishedFixtures:
    """The fixtures Maestro pins must keep matching what we serialize."""

    def test_completion_only_fixture_matches_current_serialization(self) -> None:
        published = _load("run_status_completion_only.json")
        live = _status(
            [None, None, None],
            id=42,
            current_task_index=3,
            tasks_count=3,
            total_score=66.66666666666667,
            completed_tasks=published["completed_tasks"],
        ).model_dump()
        assert live == published

    def test_evaluated_fixture_matches_current_serialization(self) -> None:
        """Dropping a measured component changes this file, loudly."""
        published = _load("run_status_evaluated.json")
        live = _status(
            [EVALUATED_RECORDS],
            id=43,
            current_task_index=1,
            tasks_count=1,
            total_score=50.0,
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


class TestHandoffPinsAreRecomputed:
    """The handoff document's pins are checked, not merely written down.

    A sha256 typed into prose is a guarantee nobody verifies: it drifts on the
    exact commit that changes the bytes, and says nothing while it does. That
    is what happened between `f58ff7f` and `05bd939` — the latter added
    `coverage`, a third fixture and `kind: aggregated_evaluation`, leaving two
    of three published pins wrong and one fixture undeclared, which the
    consumer only found while vendoring (issue #298). These tests move the
    pins under the same CI that guards the fixtures themselves: change the
    bytes without updating the table and this fails.
    """

    #: Every fixture line in the document's pins table: name, sha256.
    _ROW = re.compile(r"^\|\s*`([^`]+\.json)`\s*\|\s*`([0-9a-f]{64})`\s*\|", re.M)
    #: The lone canonical-semantics digest, in its own fenced block.
    _CANONICAL = re.compile(r"^sha256\s+([0-9a-f]{64})\s*$", re.M)

    @staticmethod
    def _doc() -> str:
        """The handoff text a consumer vendors against."""
        return DOC.read_text(encoding="utf-8")

    def test_canonical_semantics_digest_matches_the_document(self) -> None:
        """The recipe is stated in the document; here it is executed."""
        declared = self._CANONICAL.findall(self._doc())
        assert len(declared) == 1, "expected exactly one canonical digest"
        assert declared[0] == canonical_semantics_digest()

    def test_every_fixture_pin_matches_the_bytes_on_disk(self) -> None:
        pinned = dict(self._ROW.findall(self._doc()))
        assert pinned, "the pins table lost its rows"
        for name, digest in pinned.items():
            assert digest == _fixture_digest(name), f"stale pin for {name}"

    def test_no_fixture_is_published_without_a_pin(self) -> None:
        """The failure that let `run_status_evaluated.json` ship undeclared."""
        pinned = set(dict(self._ROW.findall(self._doc())))
        on_disk = {path.name for path in FIXTURES.glob("*.json")}
        assert pinned == on_disk

    def test_the_document_still_describes_the_current_kinds(self) -> None:
        """Prose drifts the same way pins do; the load-bearing words are few."""
        doc = self._doc()
        for kind in (COMPLETION_RATE, AGGREGATED_EVALUATION):
            assert kind in doc
        assert "coverage" in doc
