"""Tests for the _game_for / _el_farol_for dispatch."""

import pytest
from game_envs.games.el_farol import ElFarolBar
from game_envs.games.prisoners_dilemma import PrisonersDilemma

from atp.dashboard.tournament.service import (
    _EL_FAROL_V1_NUM_SLOTS,
    _EL_FAROL_V1_THRESHOLD_RATIO,
    _el_farol_for,
    _game_for,
)


class _FakeTournament:
    def __init__(self, game_type: str, num_players: int = 2) -> None:
        self.game_type = game_type
        self.num_players = num_players


def test_game_for_pd_is_singleton():
    t1 = _FakeTournament("prisoners_dilemma")
    t2 = _FakeTournament("prisoners_dilemma")
    assert _game_for(t1) is _game_for(t2)
    assert isinstance(_game_for(t1), PrisonersDilemma)


def test_game_for_el_farol_caches_by_num_players():
    t5a = _FakeTournament("el_farol", num_players=5)
    t5b = _FakeTournament("el_farol", num_players=5)
    t7 = _FakeTournament("el_farol", num_players=7)
    g5a = _game_for(t5a)
    g5b = _game_for(t5b)
    g7 = _game_for(t7)
    assert g5a is g5b
    assert g5a is not g7
    assert isinstance(g5a, ElFarolBar)


def test_el_farol_capacity_threshold_derived_from_ratio():
    g = _el_farol_for(num_players=10)
    expected = max(1, int(_EL_FAROL_V1_THRESHOLD_RATIO * 10))
    assert g.config.capacity_threshold == expected
    assert g.config.num_slots == _EL_FAROL_V1_NUM_SLOTS
    assert g.config.num_players == 10


def test_el_farol_min_players_yields_valid_threshold():
    g = _el_farol_for(num_players=2)
    assert g.config.capacity_threshold >= 1


def test_game_for_unknown_raises():
    from atp.dashboard.tournament.errors import ValidationError

    t = _FakeTournament("tic_tac_toe")
    with pytest.raises(ValidationError, match="unsupported"):
        _game_for(t)
