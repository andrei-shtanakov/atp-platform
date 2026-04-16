"""Tests for El Farol game-layer service-facing methods."""

import pytest
from game_envs.core.errors import ValidationError
from game_envs.games.el_farol import MAX_SLOTS_PER_DAY, ElFarolBar, ElFarolConfig


def _game(n: int = 5, num_slots: int = 16) -> ElFarolBar:
    return ElFarolBar(
        ElFarolConfig(
            num_players=n,
            num_slots=num_slots,
            capacity_threshold=3,
        )
    )


def test_validate_action_accepts_sorted_unique_slots():
    g = _game(num_slots=16)
    assert g.validate_action({"slots": [3, 7, 0]}) == {"slots": [0, 3, 7]}


def test_validate_action_accepts_empty_list():
    g = _game()
    assert g.validate_action({"slots": []}) == {"slots": []}


def test_validate_action_rejects_duplicates():
    g = _game()
    with pytest.raises(ValidationError, match="unique"):
        g.validate_action({"slots": [1, 2, 2]})


def test_validate_action_rejects_out_of_range():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"slots": [16]})


def test_validate_action_rejects_negative():
    g = _game(num_slots=16)
    with pytest.raises(ValidationError, match="out of range"):
        g.validate_action({"slots": [-1]})


def test_validate_action_rejects_too_many():
    g = _game(num_slots=16)
    too_many = list(range(MAX_SLOTS_PER_DAY + 1))
    with pytest.raises(ValidationError, match="at most"):
        g.validate_action({"slots": too_many})


def test_validate_action_rejects_non_list():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({"slots": "not-a-list"})


def test_validate_action_rejects_missing_slots():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action({})


def test_validate_action_rejects_non_dict():
    g = _game()
    with pytest.raises(ValidationError):
        g.validate_action([0, 1])  # type: ignore[arg-type]


def test_validate_action_rejects_bool_slots():
    g = _game()
    with pytest.raises(ValidationError, match="not an int"):
        g.validate_action({"slots": [True, False]})
    with pytest.raises(ValidationError, match="not an int"):
        g.validate_action({"slots": [True, True]})


def test_sanitize_is_permissive_where_validate_is_strict():
    """Boundary check from spec §3.1."""
    g = _game(num_slots=16)
    cleaned = g.action_space("player_0").sanitize([1, 2, 2, 99, -1])
    assert set(cleaned).issubset(range(16))
    with pytest.raises(ValidationError):
        g.validate_action({"slots": [1, 2, 2, 99, -1]})
