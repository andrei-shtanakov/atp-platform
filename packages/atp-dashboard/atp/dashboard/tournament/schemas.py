"""Pydantic request/response schemas for the Tournament API."""

from typing import Any

from pydantic import BaseModel, Field


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
