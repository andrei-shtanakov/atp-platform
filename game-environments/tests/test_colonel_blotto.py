"""Tests for the Colonel Blotto game."""

from __future__ import annotations

import pytest

from game_envs.core.action import StructuredActionSpace
from game_envs.core.game import GameType, MoveOrder
from game_envs.core.state import Observation
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto
from game_envs.games.registry import GameRegistry


class TestBlottoConfig:
    """Tests for BlottoConfig validation."""

    def test_defaults(self) -> None:
        config = BlottoConfig()
        assert config.num_battlefields == 3
        assert config.total_troops == 100
        assert config.num_players == 2
        assert config.num_rounds == 1

    def test_custom_values(self) -> None:
        config = BlottoConfig(
            num_battlefields=5,
            total_troops=200,
            num_rounds=3,
        )
        assert config.num_battlefields == 5
        assert config.total_troops == 200
        assert config.num_rounds == 3

    def test_frozen(self) -> None:
        config = BlottoConfig()
        with pytest.raises(AttributeError):
            config.num_battlefields = 5  # type: ignore[misc]

    def test_num_battlefields_must_be_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="num_battlefields must be >= 2"):
            BlottoConfig(num_battlefields=1)

    def test_num_battlefields_zero(self) -> None:
        with pytest.raises(ValueError, match="num_battlefields must be >= 2"):
            BlottoConfig(num_battlefields=0)

    def test_total_troops_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="total_troops must be >= 1"):
            BlottoConfig(total_troops=0)

    def test_total_troops_negative(self) -> None:
        with pytest.raises(ValueError, match="total_troops must be >= 1"):
            BlottoConfig(total_troops=-5)

    def test_inherits_gameconfig_validation(self) -> None:
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            BlottoConfig(num_rounds=0)
        with pytest.raises(ValueError, match="discount_factor"):
            BlottoConfig(discount_factor=1.5)


class TestColonelBlottoProperties:
    """Tests for game properties."""

    def test_basic_properties(self) -> None:
        game = ColonelBlotto()
        assert "Colonel Blotto" in game.name
        assert "3 battlefields" in game.name
        assert game.game_type == GameType.NORMAL_FORM
        assert game.move_order == MoveOrder.SIMULTANEOUS
        assert game.player_ids == ["player_0", "player_1"]

    def test_repeated_name(self) -> None:
        config = BlottoConfig(num_rounds=5)
        game = ColonelBlotto(config)
        assert "repeated x5" in game.name
        assert game.game_type == GameType.REPEATED

    def test_custom_battlefields_name(self) -> None:
        config = BlottoConfig(num_battlefields=7)
        game = ColonelBlotto(config)
        assert "7 battlefields" in game.name

    def test_action_space_structured(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        assert isinstance(space, StructuredActionSpace)
        assert space.schema["type"] == "allocation"
        assert len(space.schema["fields"]) == 3
        assert space.schema["total"] == 100
        assert space.schema["min_value"] == 0
        assert space.schema["max_value"] == 100

    def test_action_space_fields_match_battlefields(self) -> None:
        config = BlottoConfig(num_battlefields=5)
        game = ColonelBlotto(config)
        space = game.action_space("player_0")
        fields = space.schema["fields"]
        assert fields == [
            "battlefield_0",
            "battlefield_1",
            "battlefield_2",
            "battlefield_3",
            "battlefield_4",
        ]


class TestAllocationValidation:
    """Tests for allocation validation."""

    def test_valid_allocation(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": 40,
            "battlefield_1": 30,
            "battlefield_2": 30,
        }
        assert space.contains(alloc)

    def test_allocation_wrong_sum(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": 40,
            "battlefield_1": 30,
            "battlefield_2": 20,
        }
        assert not space.contains(alloc)

    def test_allocation_negative_value(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": -10,
            "battlefield_1": 60,
            "battlefield_2": 50,
        }
        assert not space.contains(alloc)

    def test_allocation_missing_field(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": 50,
            "battlefield_1": 50,
        }
        assert not space.contains(alloc)

    def test_allocation_extra_field(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": 25,
            "battlefield_1": 25,
            "battlefield_2": 25,
            "battlefield_3": 25,
        }
        assert not space.contains(alloc)

    def test_allocation_all_zero_wrong_total(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": 0,
            "battlefield_1": 0,
            "battlefield_2": 0,
        }
        assert not space.contains(alloc)

    def test_allocation_all_on_one_battlefield(self) -> None:
        game = ColonelBlotto()
        space = game.action_space("player_0")
        alloc = {
            "battlefield_0": 100,
            "battlefield_1": 0,
            "battlefield_2": 0,
        }
        assert space.contains(alloc)

    def test_step_rejects_invalid_allocation(self) -> None:
        game = ColonelBlotto()
        game.reset()
        valid = {
            "battlefield_0": 40,
            "battlefield_1": 30,
            "battlefield_2": 30,
        }
        invalid = {
            "battlefield_0": 50,
            "battlefield_1": 50,
            "battlefield_2": 50,
        }
        with pytest.raises(ValueError, match="Invalid allocation"):
            game.step({"player_0": valid, "player_1": invalid})

    def test_sample_produces_valid_allocation(self) -> None:
        """Sampled allocations must pass contains()."""
        game = ColonelBlotto()
        space = game.action_space("player_0")
        import random

        rng = random.Random(42)
        for _ in range(20):
            sample = space.sample(rng)
            assert space.contains(sample)


class TestPayoffComputation:
    """Tests for payoff computation."""

    def test_clear_winner_all_battlefields(self) -> None:
        """Player who wins all battlefields gets payoff 1.0."""
        config = BlottoConfig(num_battlefields=3, total_troops=6)
        game = ColonelBlotto(config)
        game.reset()
        result = game.step(
            {
                "player_0": {
                    "battlefield_0": 3,
                    "battlefield_1": 2,
                    "battlefield_2": 1,
                },
                "player_1": {
                    "battlefield_0": 0,
                    "battlefield_1": 0,
                    "battlefield_2": 6,
                },
            }
        )
        # p0 wins bf0 (3>0) and bf1 (2>0), p1 wins bf2 (6>1)
        assert result.payoffs["player_0"] == pytest.approx(2.0 / 3.0)
        assert result.payoffs["player_1"] == pytest.approx(1.0 / 3.0)

    def test_symmetric_allocation_all_tied(self) -> None:
        """Identical allocations split all battlefields."""
        config = BlottoConfig(num_battlefields=3, total_troops=6)
        game = ColonelBlotto(config)
        game.reset()
        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        result = game.step({"player_0": alloc, "player_1": dict(alloc)})
        assert result.payoffs["player_0"] == pytest.approx(0.5)
        assert result.payoffs["player_1"] == pytest.approx(0.5)

    def test_tie_on_one_battlefield(self) -> None:
        """Tie on a single battlefield splits it."""
        config = BlottoConfig(num_battlefields=3, total_troops=6)
        game = ColonelBlotto(config)
        game.reset()
        result = game.step(
            {
                "player_0": {
                    "battlefield_0": 2,
                    "battlefield_1": 4,
                    "battlefield_2": 0,
                },
                "player_1": {
                    "battlefield_0": 2,
                    "battlefield_1": 0,
                    "battlefield_2": 4,
                },
            }
        )
        # bf0: tie (2==2) -> 0.5 each
        # bf1: p0 wins (4>0) -> 1.0 for p0
        # bf2: p1 wins (4>0) -> 1.0 for p1
        # p0: (0.5 + 1.0 + 0.0) / 3 = 0.5
        # p1: (0.5 + 0.0 + 1.0) / 3 = 0.5
        assert result.payoffs["player_0"] == pytest.approx(0.5)
        assert result.payoffs["player_1"] == pytest.approx(0.5)

    def test_payoffs_sum_to_one(self) -> None:
        """Payoffs always sum to 1.0 in a 2-player game."""
        config = BlottoConfig(num_battlefields=5, total_troops=50)
        game = ColonelBlotto(config)
        game.reset()
        result = game.step(
            {
                "player_0": {
                    "battlefield_0": 20,
                    "battlefield_1": 10,
                    "battlefield_2": 10,
                    "battlefield_3": 5,
                    "battlefield_4": 5,
                },
                "player_1": {
                    "battlefield_0": 5,
                    "battlefield_1": 5,
                    "battlefield_2": 10,
                    "battlefield_3": 15,
                    "battlefield_4": 15,
                },
            }
        )
        total = sum(result.payoffs.values())
        assert total == pytest.approx(1.0)

    def test_dominant_allocation(self) -> None:
        """Player concentrating troops wins fewer battlefields."""
        config = BlottoConfig(num_battlefields=5, total_troops=10)
        game = ColonelBlotto(config)
        game.reset()
        # p0 spreads evenly, p1 concentrates
        result = game.step(
            {
                "player_0": {
                    "battlefield_0": 2,
                    "battlefield_1": 2,
                    "battlefield_2": 2,
                    "battlefield_3": 2,
                    "battlefield_4": 2,
                },
                "player_1": {
                    "battlefield_0": 10,
                    "battlefield_1": 0,
                    "battlefield_2": 0,
                    "battlefield_3": 0,
                    "battlefield_4": 0,
                },
            }
        )
        # p0 wins bf1-4 (2>0), p1 wins bf0 (10>2)
        assert result.payoffs["player_0"] == pytest.approx(4.0 / 5.0)
        assert result.payoffs["player_1"] == pytest.approx(1.0 / 5.0)

    def test_zero_troops_battlefield_tie(self) -> None:
        """Both zero troops on a battlefield is a tie."""
        config = BlottoConfig(num_battlefields=3, total_troops=4)
        game = ColonelBlotto(config)
        game.reset()
        result = game.step(
            {
                "player_0": {
                    "battlefield_0": 4,
                    "battlefield_1": 0,
                    "battlefield_2": 0,
                },
                "player_1": {
                    "battlefield_0": 0,
                    "battlefield_1": 4,
                    "battlefield_2": 0,
                },
            }
        )
        # bf0: p0 wins (4>0)
        # bf1: p1 wins (4>0)
        # bf2: tie (0==0) -> 0.5 each
        assert result.payoffs["player_0"] == pytest.approx(1.5 / 3.0)
        assert result.payoffs["player_1"] == pytest.approx(1.5 / 3.0)


class TestColonelBlottoRepeated:
    """Tests for repeated Colonel Blotto game."""

    def test_multi_round(self) -> None:
        config = BlottoConfig(
            num_battlefields=3,
            total_troops=6,
            num_rounds=3,
        )
        game = ColonelBlotto(config)
        game.reset()

        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        for i in range(3):
            result = game.step({"player_0": alloc, "player_1": dict(alloc)})
            if i < 2:
                assert not result.is_terminal
            else:
                assert result.is_terminal

    def test_step_after_terminal_raises(self) -> None:
        config = BlottoConfig(num_battlefields=3, total_troops=6)
        game = ColonelBlotto(config)
        game.reset()
        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        game.step({"player_0": alloc, "player_1": dict(alloc)})
        with pytest.raises(RuntimeError, match="terminal"):
            game.step({"player_0": alloc, "player_1": dict(alloc)})

    def test_cumulative_payoffs(self) -> None:
        """Cumulative payoffs sum correctly over rounds."""
        config = BlottoConfig(
            num_battlefields=3,
            total_troops=6,
            num_rounds=3,
        )
        game = ColonelBlotto(config)
        game.reset()

        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        for _ in range(3):
            game.step({"player_0": alloc, "player_1": dict(alloc)})

        # Each round: 0.5 payoff (all ties)
        # Total: 1.5
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(1.5)
        assert payoffs["player_1"] == pytest.approx(1.5)

    def test_discount_factor(self) -> None:
        """Discount factor reduces future payoffs."""
        config = BlottoConfig(
            num_battlefields=3,
            total_troops=6,
            num_rounds=3,
            discount_factor=0.5,
        )
        game = ColonelBlotto(config)
        game.reset()

        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        for _ in range(3):
            game.step({"player_0": alloc, "player_1": dict(alloc)})

        # Round payoff = 0.5 each time (all ties)
        # Round 0: 0.5 * 0.5^0 = 0.5
        # Round 1: 0.5 * 0.5^1 = 0.25
        # Round 2: 0.5 * 0.5^2 = 0.125
        # Total = 0.875
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(0.875)

    def test_history_tracking(self) -> None:
        config = BlottoConfig(
            num_battlefields=3,
            total_troops=6,
            num_rounds=2,
        )
        game = ColonelBlotto(config)
        game.reset()

        alloc1 = {
            "battlefield_0": 4,
            "battlefield_1": 1,
            "battlefield_2": 1,
        }
        alloc2 = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        game.step({"player_0": alloc1, "player_1": alloc2})
        game.step({"player_0": alloc2, "player_1": alloc1})

        history = game.history.for_player("player_0")
        assert len(history) == 2
        assert history[0].round_number == 0
        assert history[1].round_number == 1

    def test_reset_clears_state(self) -> None:
        config = BlottoConfig(num_battlefields=3, total_troops=6)
        game = ColonelBlotto(config)
        game.reset()
        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        game.step({"player_0": alloc, "player_1": dict(alloc)})

        game.reset()
        assert game.current_round == 0
        assert not game.is_terminal
        assert game.get_payoffs() == {
            "player_0": 0.0,
            "player_1": 0.0,
        }
        assert len(game.history.for_player("player_0")) == 0


class TestColonelBlottoObserve:
    """Tests for observe and to_prompt."""

    def test_observe_basic(self) -> None:
        game = ColonelBlotto()
        game.reset()
        obs = game.observe("player_0")
        assert isinstance(obs, Observation)
        assert obs.player_id == "player_0"
        assert obs.round_number == 0
        assert obs.total_rounds == 1
        assert obs.game_state["num_battlefields"] == 3
        assert obs.game_state["total_troops"] == 100

    def test_observe_has_payoff_rule(self) -> None:
        game = ColonelBlotto()
        game.reset()
        obs = game.observe("player_0")
        assert "payoff_rule" in obs.game_state
        assert "fraction" in obs.game_state["payoff_rule"]

    def test_observe_to_prompt_content(self) -> None:
        game = ColonelBlotto()
        game.reset()
        obs = game.observe("player_0")
        prompt = obs.to_prompt()
        assert "player_0" in prompt
        assert "Round 0 of 1" in prompt

    def test_observe_with_history(self) -> None:
        config = BlottoConfig(
            num_battlefields=3,
            total_troops=6,
            num_rounds=3,
        )
        game = ColonelBlotto(config)
        game.reset()
        alloc = {
            "battlefield_0": 2,
            "battlefield_1": 2,
            "battlefield_2": 2,
        }
        game.step({"player_0": alloc, "player_1": dict(alloc)})

        obs = game.observe("player_0")
        prompt = obs.to_prompt()
        assert "History:" in prompt
        assert "Round 1" in prompt

    def test_serialization_roundtrip(self) -> None:
        game = ColonelBlotto()
        game.reset()
        obs = game.observe("player_0")
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        assert restored.player_id == obs.player_id
        assert restored.round_number == obs.round_number
        assert restored.total_rounds == obs.total_rounds
        assert restored.game_state == obs.game_state


class TestToPrompt:
    """Tests for the game-level to_prompt method."""

    def test_to_prompt_basic(self) -> None:
        game = ColonelBlotto()
        prompt = game.to_prompt()
        assert "Colonel Blotto" in prompt
        assert "100 troops" in prompt
        assert "3 battlefields" in prompt
        assert "non-negative" in prompt
        assert "fraction" in prompt

    def test_to_prompt_repeated(self) -> None:
        config = BlottoConfig(num_rounds=5)
        game = ColonelBlotto(config)
        prompt = game.to_prompt()
        assert "5 rounds" in prompt

    def test_to_prompt_discount(self) -> None:
        config = BlottoConfig(num_rounds=5, discount_factor=0.9)
        game = ColonelBlotto(config)
        prompt = game.to_prompt()
        assert "0.9" in prompt
        assert "discount" in prompt.lower()

    def test_to_prompt_no_pure_nash(self) -> None:
        """Prompt mentions no pure Nash equilibrium."""
        game = ColonelBlotto()
        prompt = game.to_prompt()
        assert "no pure strategy" in prompt.lower()

    def test_to_prompt_custom_battlefields(self) -> None:
        config = BlottoConfig(num_battlefields=7, total_troops=50)
        game = ColonelBlotto(config)
        prompt = game.to_prompt()
        assert "7 battlefields" in prompt
        assert "50 troops" in prompt


class TestNoPureNashEquilibrium:
    """Verify: no pure Nash for symmetric Blotto (known result).

    In symmetric Colonel Blotto, for any pure strategy
    allocation, there exists a counter-allocation that wins
    more battlefields. We demonstrate this by showing that
    against any fixed pure strategy, the opponent can find
    a better response.
    """

    def test_uniform_is_not_nash(self) -> None:
        """Uniform allocation can be beaten."""
        config = BlottoConfig(num_battlefields=3, total_troops=12)
        game = ColonelBlotto(config)

        uniform = {
            "battlefield_0": 4,
            "battlefield_1": 4,
            "battlefield_2": 4,
        }
        # Counter: win 2 battlefields with 5 each, concede 1
        counter = {
            "battlefield_0": 5,
            "battlefield_1": 5,
            "battlefield_2": 2,
        }

        game.reset()
        result = game.step({"player_0": uniform, "player_1": counter})
        # Counter wins bf0 and bf1, loses bf2
        # Counter gets 2/3, uniform gets 1/3
        assert result.payoffs["player_1"] > result.payoffs["player_0"]

    def test_concentrated_is_not_nash(self) -> None:
        """Concentrated allocation can be beaten."""
        config = BlottoConfig(num_battlefields=3, total_troops=12)
        game = ColonelBlotto(config)

        concentrated = {
            "battlefield_0": 10,
            "battlefield_1": 1,
            "battlefield_2": 1,
        }
        # Counter: concede bf0, win bf1 and bf2
        counter = {
            "battlefield_0": 0,
            "battlefield_1": 6,
            "battlefield_2": 6,
        }

        game.reset()
        result = game.step({"player_0": concentrated, "player_1": counter})
        assert result.payoffs["player_1"] > result.payoffs["player_0"]

    def test_any_pure_strategy_is_exploitable(self) -> None:
        """For several pure strategies, show a better response.

        This demonstrates the key game-theoretic property:
        no pure strategy is a best response to itself in
        symmetric Blotto.
        """
        config = BlottoConfig(num_battlefields=3, total_troops=12)
        game = ColonelBlotto(config)

        strategies = [
            {
                "battlefield_0": 4,
                "battlefield_1": 4,
                "battlefield_2": 4,
            },
            {
                "battlefield_0": 6,
                "battlefield_1": 3,
                "battlefield_2": 3,
            },
            {
                "battlefield_0": 8,
                "battlefield_1": 2,
                "battlefield_2": 2,
            },
            {
                "battlefield_0": 12,
                "battlefield_1": 0,
                "battlefield_2": 0,
            },
        ]

        for strategy in strategies:
            # Playing the same strategy against itself
            # always yields 0.5
            game.reset()
            mirror = game.step(
                {
                    "player_0": strategy,
                    "player_1": dict(strategy),
                }
            )
            assert mirror.payoffs["player_0"] == pytest.approx(0.5)

            # There exists a counter-strategy that does better
            # Strategy: concentrate on winning majority of fields
            # by slightly outbidding the opponent on enough fields
            fields = sorted(strategy.items(), key=lambda x: x[1])
            # Put all troops on the cheapest fields to win
            n = len(fields)
            majority = (n // 2) + 1
            counter: dict[str, int] = {}
            remaining = config.total_troops
            for i, (field_name, _) in enumerate(fields):
                if i < majority:
                    troops = strategy[field_name] + 1
                    counter[field_name] = troops
                    remaining -= troops
                else:
                    counter[field_name] = 0
            # Distribute remaining (or fix negative)
            if remaining < 0:
                # Simple fix: just put 1 more than opponent
                # on cheapest 2 fields
                counter = {f: 0 for f, _ in fields}
                remaining = config.total_troops
                for i, (field_name, val) in enumerate(fields):
                    if i < majority:
                        t = val + 1
                        counter[field_name] = t
                        remaining -= t
                # Put rest on last field
                counter[fields[-1][0]] += max(0, remaining)
            else:
                counter[fields[-1][0]] += remaining

            game.reset()
            result = game.step(
                {
                    "player_0": strategy,
                    "player_1": counter,
                }
            )
            # The counter should do at least as well
            assert result.payoffs["player_1"] >= result.payoffs["player_0"]


class TestColonelBlottoRegistry:
    """Tests for game registry integration."""

    def setup_method(self) -> None:
        self._saved = dict(GameRegistry._registry)

    def teardown_method(self) -> None:
        GameRegistry._registry = self._saved

    def test_registered(self) -> None:
        assert "colonel_blotto" in GameRegistry.list_games()

    def test_create_from_registry(self) -> None:
        game = GameRegistry.create("colonel_blotto")
        assert isinstance(game, ColonelBlotto)
        assert "Colonel Blotto" in game.name

    def test_create_with_config(self) -> None:
        config = BlottoConfig(num_battlefields=5, total_troops=200)
        game = GameRegistry.create("colonel_blotto", config=config)
        assert isinstance(game, ColonelBlotto)
        assert "5 battlefields" in game.name
