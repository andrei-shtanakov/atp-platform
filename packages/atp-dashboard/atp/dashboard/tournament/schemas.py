"""Pydantic request/response schemas for the Tournament API."""

from typing import Annotated, Any, Literal

from game_envs.games.el_farol import MAX_SLOTS_PER_DAY
from pydantic import BaseModel, ConfigDict, Field


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


class ElFarolAction(BaseModel):
    """El Farol submit action. ``game_type`` is server-injected."""

    model_config = ConfigDict(extra="forbid")

    game_type: Literal["el_farol"]
    slots: list[int] = Field(..., max_length=MAX_SLOTS_PER_DAY)


TournamentAction = Annotated[
    PDAction | ElFarolAction,
    Field(discriminator="game_type"),
]
