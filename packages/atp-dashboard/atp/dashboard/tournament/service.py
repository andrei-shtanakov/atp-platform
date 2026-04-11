"""TournamentService — protocol-agnostic core for tournament gameplay.

This module knows about SQLAlchemy and game-environments. It does NOT
know about FastAPI, FastMCP, or HTTP. Unit-tested via direct calls
with an in-memory session and a test event bus.

This is the v1 vertical slice version: only the methods needed for a
2-player 3-round PD e2e test. Plan 2 expands the surface (deadline
worker, leave/get_history/list, AD-9/AD-10 enforcement, etc.).
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.errors import ValidationError
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Tournament,
    TournamentStatus,
)

logger = logging.getLogger("atp.dashboard.tournament.service")

_SUPPORTED_GAMES = frozenset({"prisoners_dilemma"})


class TournamentService:
    def __init__(self, session: AsyncSession, bus: TournamentEventBus) -> None:
        self._session = session
        self._bus = bus

    async def create_tournament(
        self,
        admin: User,
        *,
        name: str,
        game_type: str,
        num_players: int,
        total_rounds: int,
        round_deadline_s: int,
    ) -> Tournament:
        """Create a new tournament in PENDING (accepting-joins) state.

        Caller is responsible for verifying admin authorization at the
        transport layer; this method trusts that admin.is_admin == True.
        """
        if game_type not in _SUPPORTED_GAMES:
            raise ValidationError(
                f"unsupported game_type {game_type!r}; "
                f"v1 slice supports: {sorted(_SUPPORTED_GAMES)}"
            )
        if num_players < 2:
            raise ValidationError("num_players must be >= 2")
        if total_rounds < 1:
            raise ValidationError("total_rounds must be >= 1")
        if round_deadline_s < 1:
            raise ValidationError("round_deadline_s must be >= 1")

        tournament = Tournament(
            game_type=game_type,
            status=TournamentStatus.PENDING,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            created_by=admin.id,
            config={"name": name},
        )
        self._session.add(tournament)
        await self._session.flush()
        await self._session.refresh(tournament)
        return tournament
