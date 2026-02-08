"""Tests for communication channel and information sets."""

from __future__ import annotations

import pytest

from game_envs.core.communication import (
    CommunicationChannel,
    CommunicationMode,
    InformationSet,
)
from game_envs.core.game import GameConfig
from game_envs.core.state import Message
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma

# ── CommunicationMode enum ──────────────────────────────


class TestCommunicationMode:
    def test_values(self) -> None:
        assert CommunicationMode.NO_COMMUNICATION == "no_communication"
        assert CommunicationMode.PRE_ACTION == "pre_action"
        assert CommunicationMode.POST_ACTION == "post_action"
        assert CommunicationMode.FREE == "free"

    def test_from_string(self) -> None:
        assert CommunicationMode("pre_action") is CommunicationMode.PRE_ACTION


# ── CommunicationChannel ────────────────────────────────


class TestCommunicationChannelNoComm:
    """No-communication mode blocks all messages."""

    def test_cannot_send(self) -> None:
        ch = CommunicationChannel(CommunicationMode.NO_COMMUNICATION)
        assert not ch.can_send()

    def test_send_raises(self) -> None:
        ch = CommunicationChannel(CommunicationMode.NO_COMMUNICATION)
        ch.begin_round(0)
        with pytest.raises(RuntimeError, match="Cannot send"):
            ch.send_message("p0", "hello", 0)

    def test_phase_stays_idle(self) -> None:
        ch = CommunicationChannel(CommunicationMode.NO_COMMUNICATION)
        ch.begin_round(0)
        assert ch.phase == "idle"
        ch.end_actions(0)
        assert ch.phase == "idle"


class TestCommunicationChannelPreAction:
    """Pre-action mode: messages before actions only."""

    def test_can_send_before_actions(self) -> None:
        ch = CommunicationChannel(
            CommunicationMode.PRE_ACTION,
            player_ids=["p0", "p1"],
        )
        ch.begin_round(0)
        assert ch.can_send()
        assert ch.phase == "pre_action"

    def test_cannot_send_after_actions(self) -> None:
        ch = CommunicationChannel(CommunicationMode.PRE_ACTION)
        ch.begin_round(0)
        ch.end_actions(0)
        assert not ch.can_send()
        assert ch.phase == "idle"

    def test_send_and_receive(self) -> None:
        ch = CommunicationChannel(
            CommunicationMode.PRE_ACTION,
            player_ids=["p0", "p1"],
        )
        ch.begin_round(0)
        msg = ch.send_message("p0", "I will cooperate", 0)
        assert msg.sender == "p0"
        assert msg.content == "I will cooperate"
        assert msg.round_number == 0
        assert msg.timestamp > 0

        # p1 receives the message
        msgs = ch.get_messages(0, receiver="p1")
        assert len(msgs) == 1
        assert msgs[0].content == "I will cooperate"

        # p0 doesn't see own message when filtered
        msgs_p0 = ch.get_messages(0, receiver="p0")
        assert len(msgs_p0) == 0

    def test_unknown_sender_raises(self) -> None:
        ch = CommunicationChannel(
            CommunicationMode.PRE_ACTION,
            player_ids=["p0", "p1"],
        )
        ch.begin_round(0)
        with pytest.raises(ValueError, match="Unknown sender"):
            ch.send_message("unknown", "hi", 0)

    def test_multiple_messages(self) -> None:
        ch = CommunicationChannel(
            CommunicationMode.PRE_ACTION,
            player_ids=["p0", "p1"],
        )
        ch.begin_round(0)
        ch.send_message("p0", "msg1", 0)
        ch.send_message("p1", "msg2", 0)

        # Without filtering, get all messages
        msgs = ch.get_messages(0)
        assert len(msgs) == 2

        # p0 sees only p1's message
        msgs_p0 = ch.get_messages(0, receiver="p0")
        assert len(msgs_p0) == 1
        assert msgs_p0[0].sender == "p1"


class TestCommunicationChannelPostAction:
    """Post-action mode: messages after actions only."""

    def test_cannot_send_before_actions(self) -> None:
        ch = CommunicationChannel(CommunicationMode.POST_ACTION)
        ch.begin_round(0)
        assert not ch.can_send()

    def test_can_send_after_actions(self) -> None:
        ch = CommunicationChannel(CommunicationMode.POST_ACTION)
        ch.begin_round(0)
        ch.end_actions(0)
        assert ch.can_send()
        assert ch.phase == "post_action"

    def test_send_after_actions(self) -> None:
        ch = CommunicationChannel(CommunicationMode.POST_ACTION)
        ch.begin_round(0)
        ch.end_actions(0)
        msg = ch.send_message("p0", "good move", 0)
        assert msg.content == "good move"


class TestCommunicationChannelFree:
    """Free mode: messages at any time."""

    def test_can_send_anytime(self) -> None:
        ch = CommunicationChannel(CommunicationMode.FREE)
        # Even before begin_round, free mode allows
        assert ch.can_send()

    def test_can_send_pre_action(self) -> None:
        ch = CommunicationChannel(CommunicationMode.FREE)
        ch.begin_round(0)
        assert ch.can_send()

    def test_can_send_post_action(self) -> None:
        ch = CommunicationChannel(CommunicationMode.FREE)
        ch.begin_round(0)
        ch.end_actions(0)
        assert ch.can_send()


class TestCommunicationChannelSerialization:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        ch = CommunicationChannel(
            CommunicationMode.PRE_ACTION,
            player_ids=["p0", "p1"],
        )
        ch.begin_round(0)
        ch.send_message("p0", "hello", 0)
        ch.send_message("p1", "hi back", 0)

        data = ch.to_dict()
        assert data["mode"] == "pre_action"
        assert data["player_ids"] == ["p0", "p1"]
        assert len(data["round_messages"]["0"]) == 2

        ch2 = CommunicationChannel.from_dict(data)
        assert ch2.mode == CommunicationMode.PRE_ACTION
        msgs = ch2.get_messages(0)
        assert len(msgs) == 2
        assert msgs[0].content == "hello"

    def test_get_all_messages(self) -> None:
        ch = CommunicationChannel(CommunicationMode.FREE)
        ch.send_message("p0", "round0", 0)
        ch.send_message("p1", "round1", 1)
        all_msgs = ch.get_all_messages()
        assert len(all_msgs) == 2
        assert all_msgs[0].round_number == 0
        assert all_msgs[1].round_number == 1

    def test_clear(self) -> None:
        ch = CommunicationChannel(CommunicationMode.FREE)
        ch.send_message("p0", "msg", 0)
        ch.clear()
        assert ch.get_all_messages() == []
        assert ch.phase == "idle"

    def test_end_round(self) -> None:
        ch = CommunicationChannel(CommunicationMode.PRE_ACTION)
        ch.begin_round(0)
        assert ch.phase == "pre_action"
        ch.end_round()
        assert ch.phase == "idle"


# ── InformationSet ──────────────────────────────────────


class TestInformationSet:
    def test_full_observability(self) -> None:
        """No restrictions = full observability."""
        info_set = InformationSet(player_id="p0")
        state = {"score": 10, "private_value": 50}
        filtered = info_set.filter_state(state)
        assert filtered == state

    def test_visible_state_keys(self) -> None:
        """Only specified keys visible."""
        info_set = InformationSet(
            player_id="p0",
            visible_state_keys=["score", "game"],
        )
        state = {"score": 10, "game": "auction", "secret": 42}
        filtered = info_set.filter_state(state)
        assert filtered == {"score": 10, "game": "auction"}
        assert "secret" not in filtered

    def test_hidden_state_keys(self) -> None:
        """Explicitly hidden keys removed."""
        info_set = InformationSet(
            player_id="p0",
            hidden_state_keys=["private_value"],
        )
        state = {"score": 10, "private_value": 50}
        filtered = info_set.filter_state(state)
        assert filtered == {"score": 10}

    def test_visible_and_hidden_combined(self) -> None:
        """Hidden keys applied after visible filter."""
        info_set = InformationSet(
            player_id="p0",
            visible_state_keys=["a", "b", "c"],
            hidden_state_keys=["b"],
        )
        state = {"a": 1, "b": 2, "c": 3, "d": 4}
        filtered = info_set.filter_state(state)
        assert filtered == {"a": 1, "c": 3}

    def test_visible_players(self) -> None:
        """Visible players controls history filtering."""
        info_set = InformationSet(
            player_id="p0",
            visible_players=["p0"],
        )
        assert info_set.visible_players == ["p0"]

    def test_to_dict_from_dict_roundtrip(self) -> None:
        info_set = InformationSet(
            player_id="p0",
            visible_players=["p0", "p1"],
            visible_state_keys=["score"],
            hidden_state_keys=["secret"],
        )
        data = info_set.to_dict()
        assert data["player_id"] == "p0"
        assert data["visible_players"] == ["p0", "p1"]

        restored = InformationSet.from_dict(data)
        assert restored.player_id == "p0"
        assert restored.visible_players == ["p0", "p1"]
        assert restored.visible_state_keys == ["score"]
        assert restored.hidden_state_keys == ["secret"]

    def test_from_dict_defaults(self) -> None:
        """Missing optional fields use defaults."""
        data = {"player_id": "p0"}
        info_set = InformationSet.from_dict(data)
        assert info_set.visible_players is None
        assert info_set.visible_state_keys is None
        assert info_set.hidden_state_keys == []

    def test_frozen(self) -> None:
        """InformationSet is immutable."""
        info_set = InformationSet(player_id="p0")
        with pytest.raises(AttributeError):
            info_set.player_id = "p1"  # type: ignore[misc]

    def test_empty_state(self) -> None:
        """Filtering empty state returns empty dict."""
        info_set = InformationSet(
            player_id="p0",
            hidden_state_keys=["x"],
        )
        assert info_set.filter_state({}) == {}


# ── GameConfig communication sync ───────────────────────


class TestGameConfigCommunication:
    def test_default_no_communication(self) -> None:
        cfg = GameConfig()
        assert cfg.communication_mode == "no_communication"
        assert cfg.communication is False

    def test_communication_bool_sets_mode(self) -> None:
        """Setting communication=True auto-sets PRE_ACTION."""
        cfg = GameConfig(communication=True)
        assert cfg.communication_mode == CommunicationMode.PRE_ACTION

    def test_explicit_mode_overrides(self) -> None:
        """Explicit mode is used even with communication=False."""
        cfg = GameConfig(
            communication_mode=CommunicationMode.POST_ACTION,
        )
        assert cfg.communication_mode == CommunicationMode.POST_ACTION

    def test_communication_true_with_explicit_mode(self) -> None:
        """Explicit mode is kept when communication=True."""
        cfg = GameConfig(
            communication=True,
            communication_mode=CommunicationMode.FREE,
        )
        assert cfg.communication_mode == CommunicationMode.FREE


# ── Game base class communication integration ───────────


class TestGameCommunication:
    def test_no_channel_by_default(self) -> None:
        """Games without communication have no channel."""
        from tests.conftest import StubGame

        game = StubGame()
        assert game.channel is None

    def test_channel_created_with_mode(self) -> None:
        from tests.conftest import StubGame

        cfg = GameConfig(
            communication_mode=CommunicationMode.PRE_ACTION,
        )
        game = StubGame(config=cfg)
        assert game.channel is not None
        assert game.channel.mode == CommunicationMode.PRE_ACTION

    def test_send_message_no_channel_raises(self) -> None:
        from tests.conftest import StubGame

        game = StubGame()
        with pytest.raises(RuntimeError, match="No communication"):
            game.send_message("p0", "hello")


# ── PD with pre-action communication ────────────────────


class TestPDWithCommunication:
    """Prisoner's Dilemma with pre-action messaging."""

    def test_pd_with_pre_action_communication(self) -> None:
        """Players exchange messages before each round."""
        config = PDConfig(
            num_rounds=3,
            communication=True,
            seed=42,
        )
        game = PrisonersDilemma(config)
        assert game.channel is not None
        assert game.channel.mode == CommunicationMode.PRE_ACTION

        game.reset()

        for round_num in range(3):
            # Begin communication phase
            game.channel.begin_round(game.current_round)

            # Players exchange messages
            game.send_message("player_0", "I will cooperate")
            game.send_message("player_1", "Me too")

            # Check observations include messages
            obs_p0 = game.observe("player_0")
            obs_p1 = game.observe("player_1")

            # p0 sees p1's message only
            assert len(obs_p0.messages) == 1
            assert obs_p0.messages[0].sender == "player_1"
            assert obs_p0.messages[0].content == "Me too"

            # p1 sees p0's message only
            assert len(obs_p1.messages) == 1
            assert obs_p1.messages[0].sender == "player_0"

            # End communication, play actions
            game.channel.end_actions(game.current_round)

            result = game.step({"player_0": "cooperate", "player_1": "cooperate"})
            game.channel.end_round()

        assert result.is_terminal

    def test_pd_messages_in_prompt(self) -> None:
        """Messages appear in LLM-friendly prompt."""
        config = PDConfig(communication=True, seed=42)
        game = PrisonersDilemma(config)
        game.reset()

        game.channel.begin_round(0)
        game.send_message("player_0", "Let's cooperate!")

        obs = game.observe("player_1")
        prompt = obs.to_prompt()
        assert "Messages:" in prompt
        assert "[player_0]: Let's cooperate!" in prompt

    def test_pd_without_communication(self) -> None:
        """Default PD has no communication channel."""
        game = PrisonersDilemma()
        game.reset()
        assert game.channel is None
        obs = game.observe("player_0")
        assert obs.messages == []

    def test_pd_communication_free_mode(self) -> None:
        """PD with free communication mode."""
        config = PDConfig(
            communication_mode=CommunicationMode.FREE,
            seed=42,
        )
        game = PrisonersDilemma(config)
        game.reset()
        assert game.channel is not None

        # Can send anytime in free mode
        game.send_message("player_0", "hello before round")
        msgs = game.channel.get_messages(0)
        assert len(msgs) == 1


# ── Sealed-bid auction with hidden information ──────────


class TestAuctionWithInformationSet:
    """Sealed-bid auction with InformationSet filtering."""

    def test_auction_private_values_hidden(self) -> None:
        """Each player sees only their own private value."""
        from game_envs.games.auction import Auction, AuctionConfig

        config = AuctionConfig(num_players=3, seed=42)
        game = Auction(config)
        game.reset()

        # Create information sets for each player
        info_sets = {
            pid: InformationSet(
                player_id=pid,
                visible_players=[pid],
            )
            for pid in game.player_ids
        }

        # Each player's observation only shows their value
        for pid in game.player_ids:
            obs = game.observe(pid)
            game_state = obs.game_state
            # Player sees own private value
            assert "your_private_value" in game_state

            # Use InformationSet to filter state further
            info_set = info_sets[pid]
            filtered_history = game.history.for_player(
                pid,
                visible_players=info_set.visible_players,
            )
            # Initially no history
            assert filtered_history == []

    def test_auction_bids_hidden_during_round(self) -> None:
        """Bids are not visible to other players during round."""
        from game_envs.games.auction import Auction, AuctionConfig

        config = AuctionConfig(num_players=2, seed=42)
        game = Auction(config)
        game.reset()

        # Create info sets that hide other players' actions
        info_set_p0 = InformationSet(
            player_id="player_0",
            visible_players=["player_0"],
        )

        # Play a round
        game.step({"player_0": 30.0, "player_1": 50.0})

        # Filter history with info set
        filtered = game.history.for_player(
            "player_0",
            visible_players=info_set_p0.visible_players,
        )
        assert len(filtered) == 1
        # Only player_0's bid visible
        assert "player_0" in filtered[0].actions
        assert "player_1" not in filtered[0].actions

    def test_information_set_filters_game_state(self) -> None:
        """InformationSet can hide specific state keys."""
        from game_envs.games.auction import Auction, AuctionConfig

        config = AuctionConfig(num_players=2, seed=42)
        game = Auction(config)
        game.reset()

        obs = game.observe("player_0")

        # Create info set that hides other players' values
        info_set = InformationSet(
            player_id="player_0",
            hidden_state_keys=["num_bidders"],
        )
        filtered_state = info_set.filter_state(obs.game_state)
        assert "num_bidders" not in filtered_state
        assert "your_private_value" in filtered_state

    def test_auction_full_observability_history(self) -> None:
        """Without InformationSet, full history visible."""
        from game_envs.games.auction import Auction, AuctionConfig

        config = AuctionConfig(num_players=2, seed=42)
        game = Auction(config)
        game.reset()
        game.step({"player_0": 30.0, "player_1": 50.0})

        # Full observability (no visible_players filter)
        full_history = game.history.for_player("player_0")
        assert len(full_history) == 1
        # Both players' bids visible
        assert "player_0" in full_history[0].actions
        assert "player_1" in full_history[0].actions


# ── Message timestamp ───────────────────────────────────


class TestMessageTimestamp:
    def test_timestamp_auto_set(self) -> None:
        msg = Message(sender="p0", content="hi", round_number=0)
        assert msg.timestamp > 0

    def test_timestamp_serialized(self) -> None:
        msg = Message(
            sender="p0",
            content="hi",
            round_number=0,
            timestamp=1234.5,
        )
        data = msg.to_dict()
        assert data["timestamp"] == 1234.5

    def test_timestamp_deserialized(self) -> None:
        data = {
            "sender": "p0",
            "content": "hi",
            "round_number": 0,
            "timestamp": 9999.0,
        }
        msg = Message.from_dict(data)
        assert msg.timestamp == 9999.0

    def test_timestamp_defaults_on_missing(self) -> None:
        data = {"sender": "p0", "content": "hi", "round_number": 0}
        msg = Message.from_dict(data)
        assert msg.timestamp == 0.0


# ── InformationSet.filter_history ─────────────────────


class TestInformationSetFilterHistory:
    """Tests for InformationSet.filter_history method."""

    def test_full_observability_returns_all(self) -> None:
        """No visible_players = full history."""
        from game_envs.core.state import RoundResult

        info_set = InformationSet(player_id="p0")
        history = [
            RoundResult(
                round_number=0,
                actions={"p0": "C", "p1": "D"},
                payoffs={"p0": 0.0, "p1": 5.0},
            ),
        ]
        result = info_set.filter_history(history)
        assert len(result) == 1
        assert "p0" in result[0].actions
        assert "p1" in result[0].actions

    def test_visible_players_filters_actions(self) -> None:
        """Only visible players' actions/payoffs shown."""
        from game_envs.core.state import RoundResult

        info_set = InformationSet(
            player_id="p0",
            visible_players=["p0"],
        )
        history = [
            RoundResult(
                round_number=0,
                actions={"p0": "C", "p1": "D"},
                payoffs={"p0": 0.0, "p1": 5.0},
            ),
        ]
        result = info_set.filter_history(history)
        assert len(result) == 1
        assert "p0" in result[0].actions
        assert "p1" not in result[0].actions
        assert "p0" in result[0].payoffs
        assert "p1" not in result[0].payoffs

    def test_filter_history_preserves_messages(self) -> None:
        """Messages from visible players are kept."""
        from game_envs.core.state import RoundResult

        info_set = InformationSet(
            player_id="p0",
            visible_players=["p0"],
        )
        history = [
            RoundResult(
                round_number=0,
                actions={"p0": "C", "p1": "D"},
                payoffs={"p0": 0.0, "p1": 5.0},
                messages=[
                    Message(
                        sender="p0",
                        content="my msg",
                        round_number=0,
                    ),
                    Message(
                        sender="p1",
                        content="hidden",
                        round_number=0,
                    ),
                ],
            ),
        ]
        result = info_set.filter_history(history)
        # p0's own messages kept, p1's filtered out
        assert len(result[0].messages) == 1
        assert result[0].messages[0].sender == "p0"

    def test_filter_history_empty(self) -> None:
        """Empty history returns empty list."""
        info_set = InformationSet(
            player_id="p0",
            visible_players=["p0"],
        )
        assert info_set.filter_history([]) == []

    def test_filter_history_multi_round(self) -> None:
        """Filtering works across multiple rounds."""
        from game_envs.core.state import RoundResult

        info_set = InformationSet(
            player_id="p0",
            visible_players=["p0"],
        )
        history = [
            RoundResult(
                round_number=i,
                actions={"p0": "C", "p1": "D", "p2": "C"},
                payoffs={"p0": 1.0, "p1": 2.0, "p2": 3.0},
            )
            for i in range(3)
        ]
        result = info_set.filter_history(history)
        assert len(result) == 3
        for rr in result:
            assert set(rr.actions.keys()) == {"p0"}
            assert set(rr.payoffs.keys()) == {"p0"}


# ── Game.get_information_set ──────────────────────────


class TestGameGetInformationSet:
    """Tests for Game.get_information_set method."""

    def test_default_returns_none(self) -> None:
        """Default Game has no information set (full obs)."""
        from tests.conftest import StubGame

        game = StubGame()
        assert game.get_information_set("player_0") is None

    def test_auction_returns_info_set(self) -> None:
        """Auction provides partial observability."""
        from game_envs.games.auction import Auction, AuctionConfig

        game = Auction(AuctionConfig(seed=42))
        game.reset()
        info_set = game.get_information_set("player_0")
        assert info_set is not None
        assert info_set.player_id == "player_0"
        assert info_set.visible_players == ["player_0"]

    def test_auction_observe_uses_info_set(self) -> None:
        """Auction.observe() filters history via info set."""
        from game_envs.games.auction import Auction, AuctionConfig

        config = AuctionConfig(num_players=2, seed=42)
        game = Auction(config)
        game.reset()

        # Play a round
        game.step({"player_0": 30.0, "player_1": 50.0})

        # Observation has filtered history
        obs = game.observe("player_0")
        assert len(obs.history) == 1
        # Only own bid visible in observation
        assert "player_0" in obs.history[0].actions
        assert "player_1" not in obs.history[0].actions


# ── Reset clears communication channel ────────────────


class TestResetClearsChannel:
    """Verify that game reset clears communication state."""

    def test_pd_reset_clears_channel(self) -> None:
        """PD reset() clears communication messages."""
        config = PDConfig(
            num_rounds=3,
            communication=True,
            seed=42,
        )
        game = PrisonersDilemma(config)
        game.reset()

        # Send some messages
        game.channel.begin_round(0)
        game.send_message("player_0", "hello")
        assert len(game.channel.get_all_messages()) == 1

        # Reset should clear messages
        game.reset()
        assert game.channel.get_all_messages() == []

    def test_auction_reset_clears_channel(self) -> None:
        """Auction reset() clears channel if configured."""
        from game_envs.games.auction import Auction, AuctionConfig

        config = AuctionConfig(
            communication_mode=CommunicationMode.FREE,
            seed=42,
        )
        game = Auction(config)
        game.reset()

        game.send_message("player_0", "msg")
        assert len(game.channel.get_all_messages()) == 1

        game.reset()
        assert game.channel.get_all_messages() == []

    def test_stub_game_reset_clears_channel(self) -> None:
        """StubGame._reset_base() clears channel."""
        from tests.conftest import StubGame

        cfg = GameConfig(
            communication_mode=CommunicationMode.FREE,
            num_rounds=2,
        )
        game = StubGame(config=cfg)
        game.reset()
        game.send_message("player_0", "msg")
        assert len(game.channel.get_all_messages()) == 1

        game.reset()
        assert game.channel.get_all_messages() == []

    def test_reset_no_channel_no_error(self) -> None:
        """Reset without channel doesn't raise."""
        game = PrisonersDilemma(PDConfig(seed=42))
        game.reset()
        assert game.channel is None
        # Second reset should be fine
        game.reset()


# ── Post-action communication in PD ──────────────────


class TestPDPostActionCommunication:
    """PD with post-action communication mode."""

    def test_post_action_messages(self) -> None:
        """Messages exchanged after actions are revealed."""
        config = PDConfig(
            num_rounds=2,
            communication_mode=CommunicationMode.POST_ACTION,
            seed=42,
        )
        game = PrisonersDilemma(config)
        game.reset()

        # Round 0: play first, then communicate
        game.channel.begin_round(0)
        # Cannot send before actions in post_action mode
        assert not game.channel.can_send()

        game.step({"player_0": "cooperate", "player_1": "defect"})

        # Now allow post-action messages
        game.channel.end_actions(0)
        assert game.channel.can_send()
        game.send_message("player_0", "Why did you defect?")

        obs = game.observe("player_1")
        assert len(obs.messages) == 1
        assert "defect" in obs.messages[0].content

        game.channel.end_round()

    def test_message_observation_serialization(self) -> None:
        """Observation with messages serializes correctly."""
        config = PDConfig(
            communication=True,
            seed=42,
        )
        game = PrisonersDilemma(config)
        game.reset()

        game.channel.begin_round(0)
        game.send_message("player_0", "test msg")

        obs = game.observe("player_1")
        data = obs.to_dict()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["sender"] == "player_0"
        assert data["messages"][0]["content"] == "test msg"

        from game_envs.core.state import Observation

        restored = Observation.from_dict(data)
        assert len(restored.messages) == 1
        assert restored.messages[0].sender == "player_0"
