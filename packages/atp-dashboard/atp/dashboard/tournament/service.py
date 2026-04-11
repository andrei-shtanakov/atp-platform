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
from datetime import datetime, timedelta
from typing import Any

from game_envs.games.prisoners_dilemma import PrisonersDilemma
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import User
from atp.dashboard.tournament.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Participant,
    Round,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.state import RoundState

logger = logging.getLogger("atp.dashboard.tournament.service")

_SUPPORTED_GAMES = frozenset({"prisoners_dilemma"})

_GAME_INSTANCES: dict[str, Any] = {
    "prisoners_dilemma": PrisonersDilemma(),
}


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

    async def join(
        self,
        tournament_id: int,
        user: User,
        agent_name: str,
    ) -> Participant:
        """Join an open tournament.

        v1 slice: open-join only, no join_token, no
        MAX_ACTIVE_TOURNAMENTS_PER_USER (those land in Plan 2 per
        AD-10). When the join brings participant count to num_players,
        the tournament starts immediately and round 1 is created.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")
        if tournament.status != TournamentStatus.PENDING:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status}, "
                "not accepting joins"
            )

        participant = Participant(
            tournament_id=tournament_id,
            user_id=user.id,
            agent_name=agent_name,
        )
        self._session.add(participant)
        await self._session.flush()
        await self._session.refresh(participant)

        count = await self._session.scalar(
            select(func.count(Participant.id)).where(
                Participant.tournament_id == tournament_id
            )
        )
        if count == tournament.num_players:
            await self._start_tournament(tournament)

        return participant

    async def _start_tournament(self, tournament: Tournament) -> None:
        """Transition a PENDING tournament to ACTIVE and create round 1."""
        now = datetime.now()
        tournament.status = TournamentStatus.ACTIVE
        tournament.starts_at = now
        round_1 = Round(
            tournament_id=tournament.id,
            round_number=1,
            status="waiting_for_actions",
            started_at=now,
            deadline=now + timedelta(seconds=tournament.round_deadline_s),
            state={},
        )
        self._session.add(round_1)
        await self._session.flush()

    async def get_state_for(
        self,
        tournament_id: int,
        user: User,
    ) -> RoundState:
        """Build a player-private RoundState for the current round.

        v1 slice raises NotFoundError if the user is not a participant
        of the tournament (enumeration-guard: indistinguishable from
        'tournament does not exist').
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")

        participants = (
            (
                await self._session.execute(
                    select(Participant)
                    .where(Participant.tournament_id == tournament_id)
                    .order_by(Participant.id)
                )
            )
            .scalars()
            .all()
        )
        my_idx = next(
            (i for i, p in enumerate(participants) if p.user_id == user.id),
            None,
        )
        if my_idx is None:
            raise NotFoundError(f"tournament {tournament_id} not found")

        rounds = (
            (
                await self._session.execute(
                    select(Round)
                    .where(Round.tournament_id == tournament_id)
                    .order_by(Round.round_number)
                    .options(selectinload(Round.actions))
                )
            )
            .scalars()
            .all()
        )

        action_history: list[list[str]] = []
        cumulative_scores: list[float] = [0.0] * len(participants)
        current_round_number = 1
        found_active = False
        for r in rounds:
            if r.status == "completed":
                row: list[str] = [""] * len(participants)
                for action in r.actions:
                    p_idx = next(
                        i
                        for i, p in enumerate(participants)
                        if p.id == action.participant_id
                    )
                    row[p_idx] = action.action_data.get("choice", "")
                    cumulative_scores[p_idx] += action.payoff or 0.0
                action_history.append(row)
            else:
                current_round_number = r.round_number
                found_active = True
                break
        if not found_active and rounds:
            current_round_number = len(rounds)

        game = _GAME_INSTANCES[tournament.game_type]
        formatted = game.format_state_for_player(
            round_number=current_round_number,
            total_rounds=tournament.total_rounds,
            participant_idx=my_idx,
            action_history=action_history,
            cumulative_scores=cumulative_scores,
        )
        return RoundState(
            tournament_id=tournament_id,
            round_number=formatted["round_number"],
            game_type=formatted["game_type"],
            your_history=formatted["your_history"],
            opponent_history=formatted["opponent_history"],
            your_cumulative_score=formatted["your_cumulative_score"],
            opponent_cumulative_score=formatted["opponent_cumulative_score"],
            action_schema=formatted["action_schema"],
            your_turn=formatted["your_turn"],
            total_rounds=formatted["total_rounds"],
            extra=formatted.get("extra", {}),
        )
