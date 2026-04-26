"""Pydantic request/response schemas for the Tournament API."""

import os
from typing import Annotated, Any, Literal

from game_envs.games.el_farol import MAX_SLOTS_PER_DAY
from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_INTERVALS_PER_DAY = 2

_REASONING_MAX = int(os.environ.get("ATP_TOURNAMENT_REASONING_MAX_CHARS", "8000"))


class ActionTelemetry(BaseModel):
    """Optional agent-self-reported LLM telemetry attached to a move.

    Agents that want their submission to show up in the dashboard's
    DEBUG · OBSERVABILITY panel can populate these fields from their
    LLM client's usage data (e.g. ``response.usage`` on OpenAI /
    Anthropic SDKs). All fields are optional; unspecified ones render
    as "—" on the drawer. ``extra="forbid"`` keeps the payload tight so
    typos fail fast instead of silently dropping.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = Field(default=None, max_length=255)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    # Wall-clock milliseconds spent inside the agent's decide loop. When
    # omitted, ``submit_action`` falls back to ``(now - round.started_at)``
    # so the dashboard always has *something* — see service.submit_action.
    decide_ms: int | None = Field(default=None, ge=0)


class TournamentResponse(BaseModel):
    """Schema for returning a tournament."""

    id: int
    game_type: str
    status: str
    starts_at: str | None
    ends_at: str | None


class JoinRequest(BaseModel):
    """Schema for joining a tournament."""

    agent_name: str = Field(..., min_length=1)


class ActionRequest(BaseModel):
    """Schema for submitting an action in a round."""

    action_data: dict[str, Any]


class RoundResponse(BaseModel):
    """Schema for returning a tournament round."""

    round_number: int
    state: dict[str, Any]
    status: str
    deadline: str | None


class PDAction(BaseModel):
    """PD submit action. ``game_type`` is server-injected; clients may
    omit it on the wire (see spec §4)."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["prisoners_dilemma"]
    choice: Literal["cooperate", "defect"]
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)
    telemetry: ActionTelemetry | None = None


class ElFarolAction(BaseModel):
    """El Farol submit action. ``game_type`` is server-injected.

    Visits are expressed as ``intervals`` — up to ``_MAX_INTERVALS_PER_DAY``
    inclusive ``[start, end]`` pairs covering at most ``MAX_SLOTS_PER_DAY``
    slots in total. The empty list ``[]`` is the canonical "stay home"
    action. Pair shape matches the preferred form accepted by
    ``game_envs.games.el_farol.ElFarolActionSpace``.
    """

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["el_farol"]
    intervals: list[list[int]] = Field(
        ...,
        max_length=_MAX_INTERVALS_PER_DAY,
        description="List of inclusive [start, end] slot ranges.",
    )
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)
    telemetry: ActionTelemetry | None = None

    @field_validator("intervals")
    @classmethod
    def _validate_intervals(cls, pairs: list[list[int]]) -> list[list[int]]:
        total = 0
        for p in pairs:
            if len(p) != 2:
                raise ValueError(
                    f"each interval must be a [start, end] pair, got {p!r}"
                )
            start, end = p
            if start < 0 or end < start:
                raise ValueError(
                    f"interval [{start}, {end}] must satisfy 0 <= start <= end"
                )
            total += end - start + 1
        if total > MAX_SLOTS_PER_DAY:
            raise ValueError(
                f"intervals cover {total} slots; max is {MAX_SLOTS_PER_DAY}"
            )
        # Non-overlap + non-adjacency (needs >= 1 empty slot between runs).
        ordered = sorted(pairs, key=lambda p: p[0])
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt[0] <= prev[1] + 1:
                raise ValueError(
                    f"intervals {prev} and {nxt} overlap or are adjacent"
                )
        return pairs

    def to_slots(self) -> list[int]:
        """Expand ``intervals`` into a sorted flat slot list.

        The game-environments strict path (``ElFarolBar.validate_action``)
        accepts ``{"slots": list[int]}`` as its canonical input shape —
        the interval form is the wire-layer convenience. The tournament
        service uses this helper to bridge the two without touching the
        game-env contract.
        """
        return sorted({s for start, end in self.intervals for s in range(start, end + 1)})


class SHAction(BaseModel):
    """Stag Hunt submit action. ``game_type`` is server-injected."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["stag_hunt"]
    choice: Literal["stag", "hare"]
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)
    telemetry: ActionTelemetry | None = None


class BoSAction(BaseModel):
    """Battle of the Sexes submit action. ``game_type`` is server-injected; clients may
    omit it on the wire (see spec §4)."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["battle_of_sexes"]
    choice: Literal["A", "B"]
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)
    telemetry: ActionTelemetry | None = None


class PGAction(BaseModel):
    """Public Goods submit action. ``game_type`` is server-injected; clients may
    omit it on the wire (see spec §4)."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["public_goods"]
    contribution: float = Field(..., ge=0.0)
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)
    telemetry: ActionTelemetry | None = None


TournamentAction = Annotated[
    PDAction | SHAction | BoSAction | ElFarolAction | PGAction,
    Field(discriminator="game_type"),
]


class PDRoundState(BaseModel):
    """Wire schema for PD round state (spec §3.4).

    ``extra="forbid"`` — service strips internal-only keys before sending.
    """

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["prisoners_dilemma"]
    tournament_id: int
    your_history: list[str]
    opponent_history: list[str]
    your_cumulative_score: float
    opponent_cumulative_score: float
    round_number: int
    total_rounds: int
    your_turn: bool
    action_schema: dict

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ElFarolRoundState(BaseModel):
    """Wire schema for El Farol round state (spec §3.4).

    ``extra="forbid"`` — service strips internal-only keys before sending.
    """

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["el_farol"]
    tournament_id: int
    your_history: list[list[int]]
    attendance_by_round: list[list[int]]
    capacity_threshold: int
    your_cumulative_score: float
    all_scores: list[float]
    your_participant_idx: int
    num_slots: int
    round_number: int
    total_rounds: int
    pending_submission: bool
    action_schema: dict

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SHRoundState(BaseModel):
    """Wire schema for Stag Hunt round state.

    Structurally identical to PDRoundState — Stag Hunt is a 2-player
    simultaneous discrete-choice game — but uses a distinct
    ``game_type`` discriminator for routing clarity.

    ``extra="forbid"`` — service strips internal-only keys before sending.
    """

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["stag_hunt"]
    tournament_id: int
    your_history: list[str]
    opponent_history: list[str]
    your_cumulative_score: float
    opponent_cumulative_score: float
    round_number: int
    total_rounds: int
    your_turn: bool
    action_schema: dict

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class BoSRoundState(BaseModel):
    """Wire schema for Battle of the Sexes round state.

    Asymmetric cousin of ``SHRoundState``: adds ``your_preferred``
    (``"A"`` for participant 0, ``"B"`` for participant 1) so a client
    can reason about its own focal point without tracking participant
    indices itself.

    ``extra="forbid"`` — service strips internal-only keys before sending.
    """

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["battle_of_sexes"]
    tournament_id: int
    your_history: list[str]
    opponent_history: list[str]
    your_cumulative_score: float
    opponent_cumulative_score: float
    your_preferred: Literal["A", "B"]
    round_number: int
    total_rounds: int
    your_turn: bool
    action_schema: dict

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PGRoundState(BaseModel):
    """Wire schema for Public Goods round state.

    N-player simultaneous game; like ``ElFarolRoundState`` it exposes
    ``pending_submission`` instead of ``your_turn`` and publishes the
    full per-round contribution vector for public observability. The
    social-dilemma parameters (``endowment``, ``multiplier``) are
    included so bots can reason about the break-even threshold.

    ``extra="forbid"`` — service strips internal-only keys before sending.
    """

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["public_goods"]
    tournament_id: int
    your_history: list[float]
    all_contributions_by_round: list[list[float]]
    your_cumulative_score: float
    all_scores: list[float]
    your_participant_idx: int
    num_players: int
    endowment: float
    multiplier: float
    round_number: int
    total_rounds: int
    pending_submission: bool
    action_schema: dict

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


RoundState = Annotated[
    PDRoundState | SHRoundState | BoSRoundState | ElFarolRoundState | PGRoundState,
    Field(discriminator="game_type"),
]


__all__ = [
    "BoSAction",
    "BoSRoundState",
    "ElFarolAction",
    "ElFarolRoundState",
    "PDAction",
    "PDRoundState",
    "PGAction",
    "PGRoundState",
    "RoundState",
    "SHAction",
    "SHRoundState",
    "TournamentAction",
    "TournamentResponse",
]
