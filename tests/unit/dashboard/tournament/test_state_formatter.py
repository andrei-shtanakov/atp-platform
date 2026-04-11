"""Tests for RoundState dataclass and PD format_state_for_player."""

from __future__ import annotations


def test_round_state_to_dict_serializes_all_fields() -> None:
    from atp.dashboard.tournament.state import RoundState

    state = RoundState(
        tournament_id=7,
        round_number=5,
        game_type="prisoners_dilemma",
        your_history=["cooperate", "cooperate", "defect"],
        opponent_history=["cooperate", "defect", "cooperate"],
        your_cumulative_score=8,
        opponent_cumulative_score=10,
        action_schema={"type": "choice", "options": ["cooperate", "defect"]},
        your_turn=True,
        total_rounds=100,
    )

    d = state.to_dict()
    assert d["tournament_id"] == 7
    assert d["round_number"] == 5
    assert d["game_type"] == "prisoners_dilemma"
    assert d["your_history"] == ["cooperate", "cooperate", "defect"]
    assert d["opponent_history"] == ["cooperate", "defect", "cooperate"]
    assert d["your_cumulative_score"] == 8
    assert d["opponent_cumulative_score"] == 10
    assert d["action_schema"] == {
        "type": "choice",
        "options": ["cooperate", "defect"],
    }
    assert d["your_turn"] is True
    assert d["total_rounds"] == 100


def test_pd_format_state_for_player_first_round_no_history() -> None:
    from game_envs.games.prisoners_dilemma import PrisonersDilemma

    game = PrisonersDilemma()
    state = game.format_state_for_player(
        round_number=1,
        total_rounds=3,
        participant_idx=0,
        action_history=[],
        cumulative_scores=[0.0, 0.0],
    )
    assert state["round_number"] == 1
    assert state["game_type"] == "prisoners_dilemma"
    assert state["your_history"] == []
    assert state["opponent_history"] == []
    assert state["your_cumulative_score"] == 0.0
    assert state["opponent_cumulative_score"] == 0.0
    assert state["action_schema"]["options"] == ["cooperate", "defect"]
    assert state["your_turn"] is True
    assert state["total_rounds"] == 3


def test_pd_format_state_for_player_with_history() -> None:
    from game_envs.games.prisoners_dilemma import PrisonersDilemma

    game = PrisonersDilemma()
    history = [
        ["cooperate", "cooperate"],
        ["cooperate", "defect"],
        ["defect", "cooperate"],
    ]
    state_a = game.format_state_for_player(
        round_number=4,
        total_rounds=10,
        participant_idx=0,
        action_history=history,
        cumulative_scores=[8.0, 10.0],
    )
    assert state_a["your_history"] == ["cooperate", "cooperate", "defect"]
    assert state_a["opponent_history"] == ["cooperate", "defect", "cooperate"]
    assert state_a["your_cumulative_score"] == 8.0
    assert state_a["opponent_cumulative_score"] == 10.0

    state_b = game.format_state_for_player(
        round_number=4,
        total_rounds=10,
        participant_idx=1,
        action_history=history,
        cumulative_scores=[8.0, 10.0],
    )
    assert state_b["your_history"] == ["cooperate", "defect", "cooperate"]
    assert state_b["opponent_history"] == ["cooperate", "cooperate", "defect"]
    assert state_b["your_cumulative_score"] == 10.0
    assert state_b["opponent_cumulative_score"] == 8.0
