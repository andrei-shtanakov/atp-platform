"""Contract tests for LearningEvent v1 (RD-007 M1a).

The schema is the contractual protection against silent self-modification:
events are observational, graduation happens only via reviewed PR
(docs/2026-07-12-rd-007-learning-event-design.md §0, §6).
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

_CONTRACT_DIR = Path(__file__).parent.parent.parent / "method" / "contract"
_SCHEMA_PATH = _CONTRACT_DIR / "learning-event-v1.schema.json"
_FIXTURES_DIR = _CONTRACT_DIR / "fixtures" / "learning-event"

_ULID = "01KX9ZD3F8Q2R7T4V6W8X0Y2Z3"


def _schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _valid_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": "1",
        "event_id": _ULID,
        "producer": "robin-runtime",
        "kind": "gap",
        "ts": "2026-07-12T10:00:00Z",
        "source": {"store": "var/gaps.jsonl", "id": "gap-2026-07-12-0042"},
        "payload": {"query": "...", "failure": "zero-retrieval"},
    }
    event.update(overrides)
    return event


def test_schema_compiles() -> None:
    jsonschema.Draft7Validator.check_schema(_schema())


def test_minimal_event_validates() -> None:
    jsonschema.validate(_valid_event(), _schema())


def test_full_event_with_target_and_evidence_validates() -> None:
    event = _valid_event(
        proposed_target="eval-case",
        evidence_refs=[{"kind": "log", "pipeline_id": "01KX8V7Z9DHBKYWGSN2KTWM8AB"}],
    )
    jsonschema.validate(event, _schema())


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": "2"},  # unknown version
        {"event_id": "not-a-ulid"},
        {"event_id": _ULID.lower()},  # ULID is uppercase Crockford
        {"kind": "skill-write"},  # not in the closed vocabulary
        {"proposed_target": "runtime"},  # not a governed target class
        {"source": {"store": "var/gaps.jsonl"}},  # id missing
        {"evidence_refs": [{"kind": "log"}]},  # pipeline_id required for kind=log
        {"extra_top_level": True},  # additionalProperties: false
    ],
)
def test_invalid_events_rejected(mutation: dict[str, Any]) -> None:
    event = _valid_event()
    event.update(mutation)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(event, _schema())


@pytest.mark.parametrize(
    "missing",
    ["schema_version", "event_id", "producer", "kind", "ts", "source", "payload"],
)
def test_required_fields(missing: str) -> None:
    event = _valid_event()
    del event[missing]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(event, _schema())


def test_fixtures_validate() -> None:
    fixtures = sorted(_FIXTURES_DIR.glob("*.json"))
    assert fixtures, f"no fixtures under {_FIXTURES_DIR}"
    for path in fixtures:
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), _schema())


def test_vendored_evidence_ref_matches_maestro_shape() -> None:
    """The inline evidence_ref definition is a vendored pinned copy.

    Byte-conformance CI is M2; this guards the essentials so the vendored
    copy cannot silently drop the kind-conditional key requirements.
    """
    definition = _schema()["definitions"]["evidence_ref"]
    kinds = set(definition["properties"]["kind"]["enum"])
    assert kinds == {
        "trace",
        "log",
        "benchmark",
        "decision",
        "artifact",
        "gate-verdict",
    }
    assert definition["additionalProperties"] is False
    assert any(
        branch["if"]["properties"]["kind"]["const"] == "gate-verdict"
        and set(branch["then"]["required"]) == {"kind", "pipeline_id", "gate_id", "sha"}
        for branch in definition["allOf"]
    )
