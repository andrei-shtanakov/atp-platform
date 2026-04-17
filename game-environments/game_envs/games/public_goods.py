"""Public Goods Game implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from game_envs.core.action import ContinuousActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import (
    GameState,
    Observation,
    RoundResult,
    StepResult,
)
from game_envs.games.registry import register_game


class PGStage(StrEnum):
    """Stages in the public goods game."""

    CONTRIBUTE = "contribute"
    PUNISH = "punish"


@dataclass(frozen=True)
class PGConfig(GameConfig):
    """Public Goods Game configuration.

    Attributes:
        endowment: Initial endowment each player receives
            per round.
        multiplier: Factor applied to the public pool before
            dividing equally. Must be > 1 and < num_players
            for the social dilemma to exist.
        punishment_cost: Cost to the punisher per unit of
            punishment. Set to 0 to disable punishment stage.
        punishment_effect: Reduction in target's payoff per
            unit of punishment received.
    """

    endowment: float = 20.0
    multiplier: float = 1.6
    punishment_cost: float = 0.0
    punishment_effect: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.endowment <= 0:
            raise ValueError(f"endowment must be positive, got {self.endowment}")
        if self.multiplier <= 1.0:
            raise ValueError(f"multiplier must be > 1, got {self.multiplier}")
        if self.multiplier >= self.num_players:
            raise ValueError(
                f"multiplier ({self.multiplier}) must be < "
                f"num_players ({self.num_players}) for a "
                f"social dilemma"
            )
        if self.punishment_cost < 0:
            raise ValueError(
                f"punishment_cost must be >= 0, got {self.punishment_cost}"
            )
        if self.punishment_effect < 0:
            raise ValueError(
                f"punishment_effect must be >= 0, got {self.punishment_effect}"
            )


@register_game("public_goods", PGConfig)
class PublicGoodsGame(Game):
    """N-player Public Goods Game (one-shot or repeated).

    Each player receives an endowment and simultaneously
    chooses how much to contribute to a public pool.
    The pool is multiplied and divided equally among all
    players.

    Payoff formula:
        payoff_i = endowment - contribution_i
                   + multiplier * sum(contributions) / n

    With punishment variant (2-stage):
        Stage 1: Players choose contributions.
        Stage 2: Players choose how much to punish others.
        Final payoff includes punishment costs and effects.

    Features:
        - Supports 2 to 20 players
        - Continuous contribution in [0, endowment]
        - Optional punishment stage
        - Discount factor for repeated games
    """

    def __init__(self, config: PGConfig | None = None) -> None:
        super().__init__(config or PGConfig())
        n = self.config.num_players
        if n < 2 or n > 20:
            raise ValueError(f"num_players must be between 2 and 20, got {n}")
        self._terminal = False
        self._cumulative: dict[str, float] = {pid: 0.0 for pid in self.player_ids}
        self._stage = PGStage.CONTRIBUTE
        self._round_contributions: dict[str, float] = {}

    @property
    def _pg_config(self) -> PGConfig:
        """Typed access to the PG-specific config."""
        return self.config  # type: ignore[return-value]

    @property
    def _has_punishment(self) -> bool:
        """Whether punishment stage is enabled."""
        c = self._pg_config
        return c.punishment_cost > 0 and c.punishment_effect > 0

    @property
    def name(self) -> str:
        base = "Public Goods Game"
        if self._has_punishment:
            base += " (with punishment)"
        if self.config.num_rounds > 1:
            base += f" (repeated x{self.config.num_rounds})"
        return base

    @property
    def game_type(self) -> GameType:
        if self.config.num_rounds > 1:
            return GameType.REPEATED
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self.config.num_players)]

    def action_space(self, player_id: str) -> ContinuousActionSpace:
        """Contribution or punishment action space."""
        c = self._pg_config
        if self._stage == PGStage.PUNISH:
            return ContinuousActionSpace(
                low=0.0,
                high=c.endowment,
                description=(
                    "Choose how much to spend on punishing "
                    "other players (total across all targets)"
                ),
            )
        return ContinuousActionSpace(
            low=0.0,
            high=c.endowment,
            description=(
                f"Choose how much to contribute to the public pool (0 to {c.endowment})"
            ),
        )

    def reset(self) -> StepResult:
        """Reset game to initial state."""
        self._reset_base()
        self._terminal = False
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        self._stage = PGStage.CONTRIBUTE
        self._round_contributions = {}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        """Process one step of actions.

        In the basic variant, each step is a full round.
        In the punishment variant, the contribution stage
        is followed by a punishment stage before the round
        completes.
        """
        if self._terminal:
            raise RuntimeError("Game is already terminal")

        if self._has_punishment and self._stage == PGStage.CONTRIBUTE:
            return self._step_contribute(actions)
        if self._has_punishment and self._stage == PGStage.PUNISH:
            return self._step_punish(actions)
        return self._step_basic(actions)

    def _step_basic(self, actions: dict[str, Any]) -> StepResult:
        """Process a basic (no punishment) round."""
        c = self._pg_config
        n = c.num_players
        contributions = {pid: float(actions[pid]) for pid in self.player_ids}

        total_pool = sum(contributions.values())
        share = c.multiplier * total_pool / n

        payoffs: dict[str, float] = {}
        for pid in self.player_ids:
            payoffs[pid] = c.endowment - contributions[pid] + share

        return self._finalize_round(contributions, payoffs)

    def _step_contribute(self, actions: dict[str, Any]) -> StepResult:
        """Process the contribution stage (punishment variant)."""
        self._round_contributions = {
            pid: float(actions[pid]) for pid in self.player_ids
        }
        self._stage = PGStage.PUNISH

        # Return intermediate result (not terminal, no payoffs)
        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={
                "game": self.name,
                "stage": PGStage.PUNISH,
                "contributions": dict(self._round_contributions),
            },
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
            info={"stage": PGStage.PUNISH},
        )

    def _step_punish(self, actions: dict[str, Any]) -> StepResult:
        """Process the punishment stage."""
        c = self._pg_config
        n = c.num_players
        contributions = self._round_contributions

        total_pool = sum(contributions.values())
        share = c.multiplier * total_pool / n

        # Parse punishment actions: each value is total spent
        punishment_spent = {pid: float(actions[pid]) for pid in self.player_ids}

        # Distribute punishment equally among other players
        payoffs: dict[str, float] = {}
        punishment_received: dict[str, float] = {pid: 0.0 for pid in self.player_ids}

        for pid in self.player_ids:
            spent = punishment_spent[pid]
            if spent > 0 and n > 1:
                per_target = spent / (n - 1)
                for other in self.player_ids:
                    if other != pid:
                        punishment_received[other] += per_target

        for pid in self.player_ids:
            base = c.endowment - contributions[pid] + share
            cost = punishment_spent[pid] * c.punishment_cost
            damage = punishment_received[pid] * c.punishment_effect
            payoffs[pid] = base - cost - damage

        # Merge contribution and punishment actions for history
        combined_actions: dict[str, Any] = {}
        for pid in self.player_ids:
            combined_actions[pid] = {
                "contribution": contributions[pid],
                "punishment_spent": punishment_spent[pid],
            }

        self._stage = PGStage.CONTRIBUTE
        self._round_contributions = {}
        return self._finalize_round(combined_actions, payoffs)

    def _finalize_round(
        self,
        actions: dict[str, Any],
        payoffs: dict[str, float],
    ) -> StepResult:
        """Apply discount, record round, check termination."""
        discount = self._pg_config.discount_factor**self._current_round
        discounted = {pid: p * discount for pid, p in payoffs.items()}
        for pid in self.player_ids:
            self._cumulative[pid] += discounted[pid]

        current_round_number = self._current_round
        self._current_round += 1

        rr = RoundResult(
            round_number=current_round_number,
            actions=actions,
            payoffs=payoffs,
        )
        self._history.add_round(rr)

        if self._current_round >= self.config.num_rounds:
            self._terminal = True

        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"game": self.name},
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,
            is_terminal=self._terminal,
        )

    def observe(self, player_id: str) -> Observation:
        """Get observation with game-specific state info."""
        c = self._pg_config
        game_state: dict[str, Any] = {
            "game": self.name,
            "your_role": player_id,
            "num_players": c.num_players,
            "endowment": c.endowment,
            "multiplier": c.multiplier,
            "payoff_formula": (
                "payoff = endowment - contribution"
                " + multiplier * total_contributions"
                f" / {c.num_players}"
            ),
        }
        if self._has_punishment:
            game_state["punishment_cost"] = c.punishment_cost
            game_state["punishment_effect"] = c.punishment_effect
            game_state["stage"] = str(self._stage)
        if self._stage == PGStage.PUNISH and self._round_contributions:
            game_state["contributions"] = dict(self._round_contributions)

        return Observation(
            player_id=player_id,
            game_state=game_state,
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self.config.num_rounds,
            messages=self._get_pending_messages(player_id),
        )

    def get_payoffs(self) -> dict[str, float]:
        """Get cumulative (discounted) payoffs."""
        return dict(self._cumulative)

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def to_prompt(self) -> str:
        """Describe the public goods scenario for LLM agents."""
        c = self._pg_config
        n = c.num_players
        lines = [
            f"This is a {n}-player Public Goods Game.",
            "",
            "Rules:",
            f"- Each player receives an endowment of {c.endowment}.",
            "- You simultaneously choose how much to "
            "contribute to a shared public pool "
            f"(between 0 and {c.endowment}).",
            f"- The total contributions are multiplied by "
            f"{c.multiplier} and divided equally among "
            f"all {n} players.",
            "",
            "Payoff formula:",
            f"  payoff = {c.endowment} - your_contribution"
            f" + {c.multiplier} * total_contributions"
            f" / {n}",
        ]
        if self._has_punishment:
            lines.extend(
                [
                    "",
                    "Punishment stage:",
                    "- After contributions are revealed, you "
                    "may spend part of your endowment to "
                    "punish other players.",
                    f"- Each unit spent on punishment costs "
                    f"you {c.punishment_cost} and reduces "
                    f"each target's payoff by "
                    f"{c.punishment_effect}.",
                    "- Punishment spending is distributed "
                    "equally among all other players.",
                ]
            )
        lines.extend(
            [
                "",
                "Strategy note: In a one-shot game, the "
                "dominant strategy is to contribute nothing "
                "(free ride), but the social optimum is "
                "achieved when everyone contributes fully.",
            ]
        )
        if c.num_rounds > 1:
            lines.append(f"\nThis game is repeated for {c.num_rounds} rounds.")
            if c.discount_factor < 1.0:
                lines.append(
                    f"Future payoffs are discounted by {c.discount_factor} per round."
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tournament-facing protocol methods
    #
    # These are the strict, server-side entry points used by the ATP
    # tournament engine; they sit alongside the simulation API above
    # (``step``, ``observe``, ``get_payoffs``) and are deliberately
    # non-stateful — each is a pure function of its inputs and the
    # frozen config. The tournament always runs the basic (no-punishment)
    # variant, so these methods assume ``PGStage.CONTRIBUTE`` semantics
    # and ignore ``_stage`` / ``_round_contributions``.
    # ------------------------------------------------------------------

    def validate_action(self, raw: Any) -> dict[str, float]:
        """Validate a client-submitted contribution and return canonical form.

        Strict path — used by ``TournamentService.submit_action``. Accepts
        only ``{"contribution": <number>}`` with the value in
        ``[0, endowment]``. Numeric values are normalized to ``float``.
        """
        from game_envs.core.errors import (
            ValidationError,
        )

        if not isinstance(raw, dict):
            raise ValidationError(f"action must be a dict, got {type(raw).__name__}")
        value = raw.get("contribution")
        if value is None:
            raise ValidationError("action must have field 'contribution'")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValidationError(
                f"contribution must be a number, got {type(value).__name__}"
            )
        endowment = self._pg_config.endowment
        contribution = float(value)
        if contribution < 0 or contribution > endowment:
            raise ValidationError(
                f"contribution must be in [0, {endowment}], got {contribution}"
            )
        return {"contribution": contribution}

    def default_action_on_timeout(self) -> dict[str, float]:
        """Free-ride on timeout — contribute zero.

        Contributing nothing is individually rational and matches the
        unique Nash equilibrium of the one-shot game, so it's the safest
        default when a player fails to submit in time.
        """
        return {"contribution": 0.0}

    def format_state_for_player(
        self,
        round_number: int,
        total_rounds: int,
        participant_idx: int,
        action_history: list[dict[str, Any]],
        cumulative_scores: list[float],
    ) -> dict[str, Any]:
        """Per-player round state for the MCP ``get_current_state`` tool.

        Public Goods is fully observable after each round — the server
        publishes every player's contribution. We surface both your own
        history (``your_history``) and the full per-round vector
        (``all_contributions_by_round``) so bots can react to specific
        free-riders.

        Does NOT include ``pending_submission`` — that's a service-level
        concern injected by ``TournamentService.get_state_for``.
        """
        c = self._pg_config
        your_history: list[float] = []
        all_contributions_by_round: list[list[float]] = []
        n = c.num_players
        for row in action_history:
            round_actions = row.get("actions", {})
            per_player: list[float] = []
            for idx in range(n):
                raw = round_actions.get(idx, {}) or {}
                per_player.append(float(raw.get("contribution", 0.0)))
            all_contributions_by_round.append(per_player)
            your_history.append(per_player[participant_idx])

        return {
            "tournament_id": -1,
            "game_type": "public_goods",
            "round_number": round_number,
            "total_rounds": total_rounds,
            "your_history": your_history,
            "all_contributions_by_round": all_contributions_by_round,
            "your_cumulative_score": cumulative_scores[participant_idx],
            "all_scores": list(cumulative_scores),
            "your_participant_idx": participant_idx,
            "num_players": n,
            "endowment": c.endowment,
            "multiplier": c.multiplier,
            "action_schema": {
                "type": "float",
                "value_range": [0.0, c.endowment],
            },
            "extra": {},
        }

    def compute_round_payoffs(self, actions: dict[int, dict[str, Any]]) -> list[float]:
        """Per-round payoff in participant-idx order (no discounting).

        Mirrors ``_step_basic`` but expressed over integer indices that
        the tournament service uses. Clamps missing / malformed entries
        to 0 contribution so timeout-default actions flow through
        cleanly.

        Args:
            actions: ``participant_idx -> {"contribution": float}``.

        Returns:
            List of per-round payoffs indexed by participant_idx.
        """
        c = self._pg_config
        n = c.num_players
        contributions: list[float] = []
        for idx in range(n):
            raw = actions.get(idx, {}) or {}
            contributions.append(float(raw.get("contribution", 0.0)))
        total_pool = sum(contributions)
        share = c.multiplier * total_pool / n
        return [c.endowment - contributions[i] + share for i in range(n)]
