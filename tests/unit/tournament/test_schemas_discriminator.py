"""Tests for discriminated-union action/state schemas (spec §3.4)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from atp.dashboard.tournament.schemas import (
    ElFarolAction,
    ElFarolRoundState,
    PDAction,
    PDRoundState,
    RoundState,
    TournamentAction,
)

ADAPTER = TypeAdapter(TournamentAction)
STATE_ADAPTER = TypeAdapter(RoundState)


def test_parses_pd_action():
    result = ADAPTER.validate_python(
        {"game_type": "prisoners_dilemma", "choice": "cooperate"}
    )
    assert isinstance(result, PDAction)
    assert result.choice == "cooperate"


def test_parses_el_farol_action():
    result = ADAPTER.validate_python(
        {"game_type": "el_farol", "intervals": [[0, 0], [3, 3]]}
    )
    assert isinstance(result, ElFarolAction)
    assert result.intervals == [[0, 0], [3, 3]]


def test_pd_missing_choice_rejected():
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "prisoners_dilemma"})
    assert "choice" in str(exc.value)


def test_pd_discriminator_with_el_farol_fields_rejected():
    """Case 4: extra='forbid' surfaces BOTH missing and extra fields."""
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python(
            {"game_type": "prisoners_dilemma", "intervals": [[0, 0]]}
        )
    text = str(exc.value)
    assert "choice" in text
    assert "intervals" in text


def test_unknown_game_type_rejected():
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "tic_tac_toe", "move": "X1"})
    assert "prisoners_dilemma" in str(exc.value) or "el_farol" in str(exc.value)


def test_pd_roundtrip():
    a = ADAPTER.validate_python({"game_type": "prisoners_dilemma", "choice": "defect"})
    assert ADAPTER.validate_python(a.model_dump()) == a


def test_el_farol_roundtrip():
    a = ADAPTER.validate_python({"game_type": "el_farol", "intervals": [[1, 2]]})
    assert ADAPTER.validate_python(a.model_dump()) == a


def test_parses_pd_round_state():
    s = STATE_ADAPTER.validate_python(
        {
            "game_type": "prisoners_dilemma",
            "tournament_id": 42,
            "your_history": ["cooperate"],
            "opponent_history": ["defect"],
            "your_cumulative_score": 0.0,
            "opponent_cumulative_score": 5.0,
            "round_number": 2,
            "total_rounds": 5,
            "your_turn": True,
            "action_schema": {
                "type": "choice",
                "options": ["cooperate", "defect"],
            },
        }
    )
    assert isinstance(s, PDRoundState)
    assert s.tournament_id == 42


def test_parses_el_farol_round_state():
    s = STATE_ADAPTER.validate_python(
        {
            "game_type": "el_farol",
            "tournament_id": 42,
            "your_history": [[0, 3]],
            "attendance_by_round": [[1, 0, 0, 1]],
            "capacity_threshold": 3,
            "your_cumulative_score": 2.0,
            "all_scores": [2.0, 1.0, -1.0],
            "your_participant_idx": 0,
            "num_slots": 4,
            "round_number": 2,
            "total_rounds": 10,
            "pending_submission": False,
            "action_schema": {
                "type": "list[int]",
                "max_length": 8,
                "value_range": [0, 3],
                "unique": True,
            },
        }
    )
    assert isinstance(s, ElFarolRoundState)
    assert s.pending_submission is False
    assert s.tournament_id == 42


def test_pd_roundstate_missing_your_turn_rejected():
    with pytest.raises(ValidationError):
        STATE_ADAPTER.validate_python(
            {
                "game_type": "prisoners_dilemma",
                "tournament_id": 1,
                "your_history": [],
                "opponent_history": [],
                "your_cumulative_score": 0.0,
                "opponent_cumulative_score": 0.0,
                "round_number": 1,
                "total_rounds": 5,
                "action_schema": {},
                # your_turn missing
            }
        )
