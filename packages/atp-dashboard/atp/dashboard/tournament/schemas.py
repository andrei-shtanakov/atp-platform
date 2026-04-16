"""Pydantic request/response schemas for the Tournament API."""

import os
from typing import Annotated, Any, Literal

from game_envs.games.el_farol import MAX_SLOTS_PER_DAY
from pydantic import BaseModel, ConfigDict, Field

_REASONING_MAX = int(os.environ.get("ATP_TOURNAMENT_REASONING_MAX_CHARS", "8000"))


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


class ElFarolAction(BaseModel):
    """El Farol submit action. ``game_type`` is server-injected."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["el_farol"]
    slots: list[int] = Field(..., max_length=MAX_SLOTS_PER_DAY)
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)


class SHAction(BaseModel):
    """Stag Hunt submit action. ``game_type`` is server-injected."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["stag_hunt"]
    choice: Literal["stag", "hare"]
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)


class BoSAction(BaseModel):
    """Battle of the Sexes submit action. ``game_type`` is server-injected; clients may
    omit it on the wire (see spec §4)."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["battle_of_sexes"]
    choice: Literal["A", "B"]
    reasoning: str | None = Field(default=None, max_length=_REASONING_MAX)


TournamentAction = Annotated[
    PDAction | SHAction | BoSAction | ElFarolAction,
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


RoundState = Annotated[
    PDRoundState | SHRoundState | BoSRoundState | ElFarolRoundState,
    Field(discriminator="game_type"),
]
