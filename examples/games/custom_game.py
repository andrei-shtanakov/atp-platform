"""Custom game implementation example.

Shows how to create a new game from scratch by implementing the
Game ABC, and how to register it for use in YAML suites.

Usage:
    cd game-environments
    uv run python ../examples/games/custom_game.py
"""

from __future__ import annotations

from game_envs import (
    DiscreteActionSpace,
    Game,
    GameConfig,
    GameState,
    GameType,
    MoveOrder,
    Observation,
    RoundResult,
    StepResult,
    Strategy,
)

# --- Step 1: Implement the Game ABC ---


class RockPaperScissors(Game):
    """Rock-Paper-Scissors: a zero-sum simultaneous game.

    Payoffs:
        Win  = +1
        Lose = -1
        Tie  =  0

    Known Nash equilibrium: uniform random (1/3, 1/3, 1/3).
    """

    ACTIONS = ["rock", "paper", "scissors"]
    BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

    @property
    def name(self) -> str:
        return "Rock-Paper-Scissors"

    @property
    def game_type(self) -> GameType:
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return ["player_0", "player_1"]

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        return DiscreteActionSpace(self.ACTIONS)

    def reset(self) -> StepResult:
        self._current_round = 0
        self._history.clear()
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        return StepResult(
            state=state,
            observations={pid: self.observe(pid) for pid in self.player_ids},
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, str]) -> StepResult:
        a0 = actions["player_0"]
        a1 = actions["player_1"]

        # Compute payoffs
        if a0 == a1:
            p0, p1 = 0.0, 0.0
        elif self.BEATS[a0] == a1:
            p0, p1 = 1.0, -1.0
        else:
            p0, p1 = -1.0, 1.0

        self._current_round += 1
        self._cumulative["player_0"] += p0
        self._cumulative["player_1"] += p1

        self._history.add_round(
            RoundResult(
                round_number=self._current_round,
                actions=actions,
                payoffs={"player_0": p0, "player_1": p1},
            )
        )

        terminal = self._current_round >= self.config.num_rounds
        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"last_actions": actions},
            is_terminal=terminal,
        )
        return StepResult(
            state=state,
            observations={pid: self.observe(pid) for pid in self.player_ids},
            payoffs={"player_0": p0, "player_1": p1},
            is_terminal=terminal,
        )

    def get_payoffs(self) -> dict[str, float]:
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._current_round >= self.config.num_rounds


# --- Step 2: Create custom strategies ---


class AlwaysRock(Strategy):
    """Always plays rock."""

    @property
    def name(self) -> str:
        return "always_rock"

    def choose_action(self, observation: Observation) -> str:
        return "rock"

    def reset(self) -> None:
        pass


class CounterStrategy(Strategy):
    """Plays the counter to opponent's last action."""

    COUNTER = {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }

    @property
    def name(self) -> str:
        return "counter"

    def choose_action(self, observation: Observation) -> str:
        if not observation.history:
            return "rock"  # Default first move
        last_round = observation.history[-1]
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                return self.COUNTER[action]
        return "rock"

    def reset(self) -> None:
        pass


class FrequencyCounter(Strategy):
    """Plays the counter to opponent's most frequent action."""

    COUNTER = {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }

    @property
    def name(self) -> str:
        return "frequency_counter"

    def choose_action(self, observation: Observation) -> str:
        if not observation.history:
            return "rock"

        # Count opponent's action frequencies
        counts: dict[str, int] = {"rock": 0, "paper": 0, "scissors": 0}
        for rr in observation.history:
            for pid, action in rr.actions.items():
                if pid != observation.player_id:
                    counts[action] = counts.get(action, 0) + 1

        # Counter the most frequent
        most_common = max(counts, key=lambda k: counts[k])
        return self.COUNTER[most_common]

    def reset(self) -> None:
        pass


# --- Step 3: Run the game ---


def main() -> None:
    """Demonstrate the custom game."""
    print("=" * 60)
    print("CUSTOM GAME: Rock-Paper-Scissors")
    print("=" * 60)

    # One-shot game
    game = RockPaperScissors(GameConfig(num_rounds=1))
    result = game.reset()

    print("\nPlayer observation:")
    print(result.observations["player_0"].to_prompt())

    result = game.step({"player_0": "rock", "player_1": "scissors"})
    print(f"Rock vs Scissors -> {result.payoffs}")

    # Repeated game: AlwaysRock vs Counter
    print("\n" + "-" * 40)
    print("AlwaysRock vs CounterStrategy (10 rounds)")
    print("-" * 40)

    game = RockPaperScissors(GameConfig(num_rounds=10))
    rock = AlwaysRock()
    counter = CounterStrategy()

    result = game.reset()
    while not result.is_terminal:
        obs_0 = result.observations["player_0"]
        obs_1 = result.observations["player_1"]
        result = game.step(
            {
                "player_0": rock.choose_action(obs_0),
                "player_1": counter.choose_action(obs_1),
            }
        )

    print(f"Final payoffs: {game.get_payoffs()}")
    print("Counter should win 9 of 10 (loses round 1, then adapts)")

    # FrequencyCounter vs AlwaysRock
    print("\n" + "-" * 40)
    print("FrequencyCounter vs AlwaysRock (20 rounds)")
    print("-" * 40)

    game = RockPaperScissors(GameConfig(num_rounds=20))
    freq = FrequencyCounter()
    rock = AlwaysRock()

    result = game.reset()
    while not result.is_terminal:
        obs_0 = result.observations["player_0"]
        obs_1 = result.observations["player_1"]
        result = game.step(
            {
                "player_0": freq.choose_action(obs_0),
                "player_1": rock.choose_action(obs_1),
            }
        )

    print(f"Final payoffs: {game.get_payoffs()}")

    # Show history
    print("\nGame history (last 5 rounds):")
    for rr in game.history.rounds[-5:]:
        print(f"  Round {rr.round_number}: {rr.actions} -> {rr.payoffs}")


if __name__ == "__main__":
    main()
