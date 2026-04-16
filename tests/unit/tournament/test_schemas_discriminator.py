"""Tests for discriminated-union action/state schemas (spec §3.4)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from atp.dashboard.tournament.schemas import (
    ElFarolAction,
    PDAction,
    TournamentAction,
)

ADAPTER = TypeAdapter(TournamentAction)


def test_parses_pd_action():
    result = ADAPTER.validate_python(
        {"game_type": "prisoners_dilemma", "choice": "cooperate"}
    )
    assert isinstance(result, PDAction)
    assert result.choice == "cooperate"


def test_parses_el_farol_action():
    result = ADAPTER.validate_python({"game_type": "el_farol", "slots": [0, 3]})
    assert isinstance(result, ElFarolAction)
    assert result.slots == [0, 3]


def test_pd_missing_choice_rejected():
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "prisoners_dilemma"})
    assert "choice" in str(exc.value)


def test_pd_discriminator_with_el_farol_fields_rejected():
    """Case 4: extra='forbid' surfaces BOTH missing and extra fields."""
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "prisoners_dilemma", "slots": [0]})
    text = str(exc.value)
    assert "choice" in text
    assert "slots" in text


def test_unknown_game_type_rejected():
    with pytest.raises(ValidationError) as exc:
        ADAPTER.validate_python({"game_type": "tic_tac_toe", "move": "X1"})
    assert "prisoners_dilemma" in str(exc.value) or "el_farol" in str(exc.value)


def test_pd_roundtrip():
    a = ADAPTER.validate_python({"game_type": "prisoners_dilemma", "choice": "defect"})
    assert ADAPTER.validate_python(a.model_dump()) == a


def test_el_farol_roundtrip():
    a = ADAPTER.validate_python({"game_type": "el_farol", "slots": [1, 2]})
    assert ADAPTER.validate_python(a.model_dump()) == a
