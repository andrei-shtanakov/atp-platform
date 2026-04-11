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
