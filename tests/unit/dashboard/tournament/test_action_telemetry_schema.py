"""Tests for the optional ``ActionTelemetry`` payload on submit actions.

The schema is opt-in — agents that omit it leave the tier-2 columns NULL
and the dashboard renders "—".  These tests pin the validation rules
that production relies on (``decide_ms`` defaults to ``None`` and rejects
negatives via ``ge=0``) plus the ``extra="forbid"`` posture so typos
fail fast instead of silently dropping fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.dashboard.tournament.schemas import ActionTelemetry, ElFarolAction


def test_action_telemetry_defaults_all_fields_to_none() -> None:
    """Bare-payload construction leaves every field as ``None`` so the
    drawer renders "—" rather than zeroes for un-instrumented agents.
    """
    # GIVEN no telemetry payload supplied
    # WHEN we instantiate ActionTelemetry with no arguments
    tel = ActionTelemetry()

    # THEN every field defaults to None
    assert tel.model_id is None
    assert tel.tokens_in is None
    assert tel.tokens_out is None
    assert tel.cost_usd is None
    assert tel.decide_ms is None


def test_action_telemetry_accepts_decide_ms_zero() -> None:
    """``ge=0`` admits the boundary value — agents that decide
    instantaneously (cached / hard-coded responses) still validate.
    """
    # GIVEN decide_ms at the lower bound
    # WHEN we validate
    tel = ActionTelemetry(decide_ms=0)

    # THEN it accepts zero
    assert tel.decide_ms == 0


def test_action_telemetry_accepts_positive_decide_ms() -> None:
    # GIVEN a realistic positive value (sub-second LLM decode)
    # WHEN we validate
    tel = ActionTelemetry(decide_ms=874)

    # THEN it round-trips verbatim
    assert tel.decide_ms == 874


def test_action_telemetry_rejects_negative_decide_ms() -> None:
    """The ``ge=0`` constraint must reject negatives — a negative
    decide_ms would mean wall-clock went backwards, which is always
    a bug at the source we want to surface immediately.
    """
    # GIVEN a negative decide_ms
    # WHEN we validate
    # THEN pydantic raises with a greater-than-or-equal complaint
    with pytest.raises(ValidationError) as exc:
        ActionTelemetry(decide_ms=-1)
    assert "greater than or equal to 0" in str(exc.value)


def test_action_telemetry_round_trips_full_payload() -> None:
    """All five fields populated together — exact-equality round-trip,
    no coercion / clamping by validators."""
    # GIVEN all telemetry fields set
    # WHEN we validate
    tel = ActionTelemetry(
        model_id="gpt-4o-mini-2024-07-18",
        tokens_in=512,
        tokens_out=128,
        cost_usd=0.000234,
        decide_ms=874,
    )

    # THEN every value is preserved exactly
    assert tel.model_id == "gpt-4o-mini-2024-07-18"
    assert tel.tokens_in == 512
    assert tel.tokens_out == 128
    assert tel.cost_usd == 0.000234
    assert tel.decide_ms == 874


def test_action_telemetry_forbids_unknown_fields() -> None:
    """``extra='forbid'`` is non-negotiable — silent drops on typos
    would mean a misnamed ``decision_ms`` never makes it to the DB and
    the agent operator never finds out."""
    # GIVEN a misspelled field
    # WHEN we validate
    # THEN pydantic rejects with extra-forbidden
    with pytest.raises(ValidationError) as exc:
        ActionTelemetry(decision_ms=874)  # type: ignore[call-arg]
    assert "extra" in str(exc.value).lower() or "forbid" in str(exc.value).lower()


def test_el_farol_action_accepts_full_telemetry_block() -> None:
    """End-to-end: telemetry nested on the wire-shape ElFarolAction
    survives validation. Without this, the ``getattr(typed, 'telemetry', None)``
    in submit_action would always fall back to None and the capture
    path would never fire."""
    # GIVEN an ElFarolAction with a nested telemetry block
    # WHEN we validate
    action = ElFarolAction(
        game_type="el_farol",
        intervals=[[0, 0], [3, 3]],
        telemetry=ActionTelemetry(
            model_id="claude-3-5-sonnet",
            tokens_in=200,
            tokens_out=50,
            cost_usd=0.005,
            decide_ms=1234,
        ),
    )

    # THEN telemetry is attached and decide_ms survives
    assert action.telemetry is not None
    assert action.telemetry.decide_ms == 1234
    assert action.telemetry.model_id == "claude-3-5-sonnet"
