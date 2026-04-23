"""LABS-TSA PR-4 — builtins listing endpoint.

Powers the "Builtin sparring partners" widget on
``/ui/tournaments/new``.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from atp.dashboard.tournament.builtins import list_builtins_for_game

router = APIRouter(prefix="/v1/games", tags=["games", "builtins"])


class BuiltinEntry(BaseModel):
    name: str
    description: str


class BuiltinsResponse(BaseModel):
    game_type: str
    builtins: list[BuiltinEntry]


@router.get("/{game_type}/builtins", response_model=BuiltinsResponse)
async def list_builtins(game_type: str) -> BuiltinsResponse:
    """Return namespaced builtin strategies for ``game_type``.

    Unknown games return an empty list (HTTP 200 with
    ``{"game_type": ..., "builtins": []}``) so the UI does not have
    to special-case 404.
    """
    return BuiltinsResponse(
        game_type=game_type,
        builtins=[BuiltinEntry(**asdict(b)) for b in list_builtins_for_game(game_type)],
    )
