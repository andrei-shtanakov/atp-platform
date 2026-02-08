"""Tests for GameHistory."""

from __future__ import annotations

from game_envs.core.history import GameHistory
from game_envs.core.state import Message, RoundResult


class TestGameHistory:
    def test_empty(self) -> None:
        h = GameHistory()
        assert len(h) == 0
        assert h.rounds == []

    def test_add_round(self, sample_round_result: RoundResult) -> None:
        h = GameHistory()
        h.add_round(sample_round_result)
        assert len(h) == 1
        assert h.rounds[0] == sample_round_result

    def test_multiple_rounds(self) -> None:
        h = GameHistory()
        for i in range(5):
            h.add_round(
                RoundResult(
                    round_number=i + 1,
                    actions={"p1": "A", "p2": "B"},
                    payoffs={"p1": float(i), "p2": float(i)},
                )
            )
        assert len(h) == 5
        assert h.rounds[-1].round_number == 5

    def test_for_player_full_observability(self) -> None:
        h = GameHistory()
        rr = RoundResult(
            round_number=1,
            actions={"p1": "A", "p2": "B"},
            payoffs={"p1": 1.0, "p2": 2.0},
        )
        h.add_round(rr)
        result = h.for_player("p1")
        assert len(result) == 1
        assert result[0].actions == {"p1": "A", "p2": "B"}

    def test_for_player_partial_observability(self) -> None:
        h = GameHistory()
        rr = RoundResult(
            round_number=1,
            actions={"p1": "A", "p2": "B", "p3": "C"},
            payoffs={"p1": 1.0, "p2": 2.0, "p3": 3.0},
        )
        h.add_round(rr)
        result = h.for_player("p1", visible_players=["p1"])
        assert len(result) == 1
        assert result[0].actions == {"p1": "A"}
        assert result[0].payoffs == {"p1": 1.0}

    def test_for_player_messages_filtered(self) -> None:
        h = GameHistory()
        rr = RoundResult(
            round_number=1,
            actions={"p1": "A", "p2": "B"},
            payoffs={"p1": 1.0, "p2": 2.0},
            messages=[
                Message("p1", "hello", 1),
                Message("p2", "world", 1),
                Message("p3", "hidden", 1),
            ],
        )
        h.add_round(rr)
        result = h.for_player("p1", visible_players=["p1", "p2"])
        senders = [m.sender for m in result[0].messages]
        # p1 sees own messages and visible players' messages
        assert "p1" in senders
        assert "p2" in senders
        assert "p3" not in senders

    def test_get_player_actions(self) -> None:
        h = GameHistory()
        for i, action in enumerate(["cooperate", "defect", "cooperate"]):
            h.add_round(
                RoundResult(
                    round_number=i + 1,
                    actions={"p1": action, "p2": "defect"},
                    payoffs={"p1": 0.0, "p2": 0.0},
                )
            )
        actions = h.get_player_actions("p1")
        assert actions == [
            "cooperate",
            "defect",
            "cooperate",
        ]

    def test_get_player_payoffs(self) -> None:
        h = GameHistory()
        for i in range(3):
            h.add_round(
                RoundResult(
                    round_number=i + 1,
                    actions={"p1": "A"},
                    payoffs={"p1": float(i + 1)},
                )
            )
        payoffs = h.get_player_payoffs("p1")
        assert payoffs == [1.0, 2.0, 3.0]

    def test_total_payoff(self) -> None:
        h = GameHistory()
        for i in range(3):
            h.add_round(
                RoundResult(
                    round_number=i + 1,
                    actions={"p1": "A"},
                    payoffs={"p1": float(i + 1)},
                )
            )
        assert h.total_payoff("p1") == 6.0

    def test_clear(self) -> None:
        h = GameHistory()
        h.add_round(
            RoundResult(
                round_number=1,
                actions={"p1": "A"},
                payoffs={"p1": 1.0},
            )
        )
        assert len(h) == 1
        h.clear()
        assert len(h) == 0

    def test_roundtrip(self) -> None:
        h = GameHistory()
        h.add_round(
            RoundResult(
                round_number=1,
                actions={"p1": "A", "p2": "B"},
                payoffs={"p1": 1.0, "p2": 2.0},
                messages=[
                    Message("p1", "hello", 1),
                ],
            )
        )
        h.add_round(
            RoundResult(
                round_number=2,
                actions={"p1": "B", "p2": "A"},
                payoffs={"p1": 3.0, "p2": 0.0},
            )
        )
        d = h.to_dict()
        h2 = GameHistory.from_dict(d)
        assert len(h2) == 2
        assert h2.rounds[0].actions == {"p1": "A", "p2": "B"}
        assert h2.rounds[1].payoffs == {
            "p1": 3.0,
            "p2": 0.0,
        }
        assert len(h2.rounds[0].messages) == 1

    def test_from_dict_empty(self) -> None:
        h = GameHistory.from_dict({})
        assert len(h) == 0

    def test_repr(self) -> None:
        h = GameHistory()
        assert "0" in repr(h)
        h.add_round(
            RoundResult(
                round_number=1,
                actions={},
                payoffs={},
            )
        )
        assert "1" in repr(h)

    def test_rounds_returns_copy(self) -> None:
        h = GameHistory()
        h.add_round(
            RoundResult(
                round_number=1,
                actions={"p1": "A"},
                payoffs={"p1": 1.0},
            )
        )
        rounds = h.rounds
        rounds.clear()
        assert len(h) == 1  # Original unaffected
