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
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import User
from atp.dashboard.tournament.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from atp.dashboard.tournament.events import (
    TournamentCancelEvent,
    TournamentEvent,
    TournamentEventBus,
)
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
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
        required_players = _GAME_INSTANCES[game_type].config.num_players
        if num_players != required_players:
            _p = "player" if required_players == 1 else "players"
            raise ValidationError(
                f"{game_type} requires exactly {required_players} {_p}, "
                f"got {num_players}"
            )
        if total_rounds < 1:
            raise ValidationError("total_rounds must be >= 1")
        if round_deadline_s < 1:
            raise ValidationError("round_deadline_s must be >= 1")

        # TODO(Task 12): replace with computed deadline from pending_timeout_seconds
        tournament = Tournament(
            game_type=game_type,
            status=TournamentStatus.PENDING,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            created_by=admin.id,
            config={"name": name},
            pending_deadline=datetime.now(),
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
        join_token: str | None = None,
    ) -> tuple[Participant, bool]:
        """Idempotent join with race-aware IntegrityError handling.

        Returns (participant, is_new).

        Race semantics on INSERT IntegrityError:
        - uq_participant_tournament_user violation: a concurrent idempotent
          re-join from the same user won the insert race. Re-read the
          existing row and return (existing, False).
        - uq_participant_user_active violation: user has active participation
          in a different tournament. Raise ConflictError(409).
        - Any other IntegrityError: re-raise unchanged.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")
        if tournament.status != TournamentStatus.PENDING:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status!r}, "
                "not accepting joins"
            )

        # Private tournament token check (AD-10)
        if tournament.join_token is not None:
            import hmac

            if join_token is None or not hmac.compare_digest(
                tournament.join_token, join_token
            ):
                raise ConflictError(
                    f"tournament {tournament_id} requires a valid join_token"
                )

        # Capture scalar values before any potential rollback expiry
        user_id = user.id

        # Idempotent pre-check
        existing = await self._session.scalar(
            select(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.user_id == user_id)
        )
        if existing is not None:
            if existing.released_at is not None:
                # Leave is terminal — cannot rejoin
                raise ConflictError(
                    f"user {user_id} already left tournament "
                    f"{tournament_id}; rejoin not permitted"
                )
            return existing, False

        # INSERT path
        participant = Participant(
            tournament_id=tournament_id,
            user_id=user_id,
            agent_name=agent_name,
            released_at=None,
        )
        self._session.add(participant)
        try:
            await self._session.flush()
            count = await self._session.scalar(
                select(func.count(Participant.id)).where(
                    Participant.tournament_id == tournament_id
                )
            )
            if count == tournament.num_players:
                await self._start_tournament(tournament)
        except IntegrityError as exc:
            constraint_name = self._extract_constraint_name(exc)
            await self._session.rollback()
            if constraint_name in (
                "uq_participant_tournament_user",
                "uq_participant_user_active",
            ):
                # Re-read to determine actual conflict type.
                # Both constraints fire when the same user is already active in any
                # tournament. Check if the existing row is for THIS tournament
                # (idempotent re-join) or a DIFFERENT one (true conflict).
                existing = await self._session.scalar(
                    select(Participant)
                    .where(Participant.tournament_id == tournament_id)
                    .where(Participant.user_id == user_id)
                    .where(Participant.released_at.is_(None))
                )
                if existing is not None:
                    # Same tournament — idempotent re-join
                    return existing, False
                # Different tournament active — true conflict
                raise ConflictError("user already has an active tournament") from exc
            raise

        return participant, True

    @staticmethod
    def _extract_constraint_name(exc: Exception) -> str:
        """Best-effort constraint name extraction from IntegrityError.

        SQLite reports: 'UNIQUE constraint failed: table.col1, table.col2'
        or 'UNIQUE constraint failed: table.col' for single-column indices.
        Named constraints may appear as 'constraint failed: <name>'.
        PostgreSQL: exc.orig.diag.constraint_name is available.

        Disambiguation for tournament_participants:
        - uq_participant_user_active: single-column partial index on user_id
          → message contains only 'tournament_participants.user_id'
        - uq_participant_tournament_user: two-column index on
          (tournament_id, user_id) → message contains 'tournament_id'
        """
        # Extract only the first line of the error — the constraint description.
        # SQLite format: '(sqlite3.IntegrityError) UNIQUE constraint failed: table.col'
        # The full message includes the SQL statement which can contaminate searches.
        first_line = str(exc).split("\n")[0].lower()
        # Named index/constraint — check for explicit name first
        for name in (
            "uq_participant_user_active",
            "uq_participant_tournament_user",
            "uq_action_round_participant",
            "uq_round_tournament_number",
        ):
            if name in first_line:
                return name
        # SQLite column-based disambiguation for tournament_participants.
        # The partial unique index on user_id (uq_participant_user_active) shows
        # only 'tournament_participants.user_id' in the first line (single column).
        # The composite (tournament_id, user_id) unique constraint shows both
        # 'tournament_participants.tournament_id' and 'tournament_participants.user_id'.
        if "tournament_participants" in first_line and "user_id" in first_line:
            if "tournament_id" in first_line:
                # composite (tournament_id, user_id) → uq_participant_tournament_user
                return "uq_participant_tournament_user"
            else:
                # single user_id partial index → uq_participant_user_active
                return "uq_participant_user_active"
        # PostgreSQL path
        orig = getattr(exc, "orig", None)
        diag = getattr(orig, "diag", None)
        if diag is not None:
            return getattr(diag, "constraint_name", "") or ""
        return ""

    async def _start_tournament(self, tournament: Tournament) -> None:
        """Transition a PENDING tournament to ACTIVE and create round 1."""
        now = datetime.now()
        tournament.status = TournamentStatus.ACTIVE
        tournament.starts_at = now
        round_1 = Round(
            tournament_id=tournament.id,
            round_number=1,
            status=RoundStatus.WAITING_FOR_ACTIONS,
            started_at=now,
            deadline=now + timedelta(seconds=tournament.round_deadline_s),
            state={},
        )
        self._session.add(round_1)
        await self._session.flush()
        await self._bus.publish(
            TournamentEvent(
                event_type="round_started",
                tournament_id=tournament.id,
                round_number=1,
                data={"total_rounds": tournament.total_rounds},
                timestamp=datetime.now(),
            )
        )

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
            if r.status == RoundStatus.COMPLETED:
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

    async def submit_action(
        self,
        tournament_id: int,
        user: User,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit one player's action for the current round.

        Returns:
            {"status": "waiting", "round_number": N} if more actions
            are still expected this round.

            {"status": "round_resolved", ...} if this was the last
            action and the round resolved synchronously.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id} not found")
        if tournament.status != TournamentStatus.ACTIVE:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status}, not active"
            )

        my_participant = (
            await self._session.execute(
                select(Participant).where(
                    Participant.tournament_id == tournament_id,
                    Participant.user_id == user.id,
                )
            )
        ).scalar_one_or_none()
        if my_participant is None:
            raise NotFoundError(f"tournament {tournament_id} not found")

        current_round = (
            await self._session.execute(
                select(Round)
                .where(
                    Round.tournament_id == tournament_id,
                    Round.status == RoundStatus.WAITING_FOR_ACTIONS,
                )
                .order_by(Round.round_number.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if current_round is None:
            raise ConflictError(
                f"tournament {tournament_id} has no round accepting actions"
            )

        game = _GAME_INSTANCES[tournament.game_type]
        schema_probe = game.format_state_for_player(
            round_number=1,
            total_rounds=1,
            participant_idx=0,
            action_history=[],
            cumulative_scores=[0.0, 0.0],
        )
        if action.get("choice") not in schema_probe["action_schema"]["options"]:
            raise ValidationError(
                f"invalid action {action!r} for game {tournament.game_type}"
            )

        existing = (
            await self._session.execute(
                select(Action).where(
                    Action.round_id == current_round.id,
                    Action.participant_id == my_participant.id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ConflictError(
                f"participant {my_participant.id} already submitted in round "
                f"{current_round.round_number}"
            )

        new_action = Action(
            round_id=current_round.id,
            participant_id=my_participant.id,
            action_data={"choice": action["choice"]},
        )
        self._session.add(new_action)
        await self._session.flush()

        action_count = (
            await self._session.scalar(
                select(func.count(Action.id)).where(Action.round_id == current_round.id)
            )
            or 0
        )

        if action_count < tournament.num_players:
            return {
                "status": "waiting",
                "round_number": current_round.round_number,
            }

        return await self._resolve_round(current_round, tournament)

    async def _resolve_round(
        self, round_obj: Round, tournament: Tournament
    ) -> dict[str, Any]:
        """Resolve a round: compute payoffs, write Action.payoff, mark
        round completed, and either create the next round or finish the
        tournament if this was the last.
        """
        round_obj.status = "resolving"
        await self._session.flush()

        actions = (
            (
                await self._session.execute(
                    select(Action)
                    .where(Action.round_id == round_obj.id)
                    .order_by(Action.participant_id)
                )
            )
            .scalars()
            .all()
        )

        participants = (
            (
                await self._session.execute(
                    select(Participant)
                    .where(Participant.tournament_id == tournament.id)
                    .order_by(Participant.id)
                )
            )
            .scalars()
            .all()
        )
        idx_by_pid = {p.id: i for i, p in enumerate(participants)}

        action_vec: list[str] = [""] * len(participants)
        actions_by_idx: dict[int, Action] = {}
        for a in actions:
            i = idx_by_pid[a.participant_id]
            action_vec[i] = a.action_data["choice"]
            actions_by_idx[i] = a

        # PD payoff matrix (slice-local; in Plan 2 this comes from the
        # game-environments PD class).
        # CC = 3,3 ; CD = 0,5 ; DC = 5,0 ; DD = 1,1
        a0, a1 = action_vec[0], action_vec[1]
        if a0 == "cooperate" and a1 == "cooperate":
            payoffs = [3.0, 3.0]
        elif a0 == "cooperate" and a1 == "defect":
            payoffs = [0.0, 5.0]
        elif a0 == "defect" and a1 == "cooperate":
            payoffs = [5.0, 0.0]
        else:
            payoffs = [1.0, 1.0]

        for i, action in actions_by_idx.items():
            action.payoff = payoffs[i]

        round_obj.status = RoundStatus.COMPLETED
        await self._session.flush()

        if round_obj.round_number >= tournament.total_rounds:
            await self._complete_tournament(tournament)
            await self._session.flush()
            return {
                "status": "round_resolved",
                "round_number": round_obj.round_number,
                "tournament_completed": True,
                "payoffs": payoffs,
            }

        now = datetime.now()
        next_round = Round(
            tournament_id=tournament.id,
            round_number=round_obj.round_number + 1,
            status=RoundStatus.WAITING_FOR_ACTIONS,
            started_at=now,
            deadline=now + timedelta(seconds=tournament.round_deadline_s),
            state={},
        )
        self._session.add(next_round)
        await self._session.flush()
        await self._bus.publish(
            TournamentEvent(
                event_type="round_started",
                tournament_id=tournament.id,
                round_number=next_round.round_number,
                data={"total_rounds": tournament.total_rounds},
                timestamp=datetime.now(),
            )
        )

        return {
            "status": "round_resolved",
            "round_number": round_obj.round_number,
            "tournament_completed": False,
            "payoffs": payoffs,
            "next_round_number": next_round.round_number,
        }

    async def _complete_tournament(self, tournament: Tournament) -> None:
        """Mark tournament COMPLETED and write final per-participant scores."""
        tournament.status = TournamentStatus.COMPLETED
        tournament.ends_at = datetime.now()

        participants = (
            (
                await self._session.execute(
                    select(Participant).where(
                        Participant.tournament_id == tournament.id
                    )
                )
            )
            .scalars()
            .all()
        )
        for p in participants:
            total = await self._session.scalar(
                select(func.coalesce(func.sum(Action.payoff), 0.0)).where(
                    Action.participant_id == p.id
                )
            )
            p.total_score = float(total or 0.0)
        await self._session.flush()
        await self._bus.publish(
            TournamentEvent(
                event_type="tournament_completed",
                tournament_id=tournament.id,
                round_number=None,
                data={
                    "final_scores": {p.user_id: p.total_score for p in participants},
                },
                timestamp=datetime.now(),
            )
        )

    async def leave(self, tournament_id: int, user: User) -> None:
        """Mark the caller's Participant as released.

        If the caller is the last active participant of an ACTIVE
        tournament, cascade to _cancel_impl with reason=ABANDONED inside
        the same transaction. Caller owns the transaction boundary.
        """
        participant = await self._session.scalar(
            select(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.user_id == user.id)
            .where(Participant.released_at.is_(None))
        )
        if participant is None:
            raise NotFoundError(
                f"user {user.id} is not active in tournament {tournament_id}"
            )

        participant.released_at = datetime.now()
        await self._session.flush()

        remaining = await self._session.scalar(
            select(func.count())
            .select_from(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.released_at.is_(None))
        )
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if remaining == 0 and tournament.status == TournamentStatus.ACTIVE:
            logger.info(
                "tournament.leave.abandoned_cascade",
                extra={
                    "tournament_id": tournament_id,
                    "leaving_user_id": user.id,
                },
            )
            cancel_event = await self._cancel_impl(
                tournament_id,
                reason=CancelReason.ABANDONED,
                cancelled_by=None,
                reason_detail=None,
            )
            if cancel_event is not None:
                try:
                    await self._bus.publish(cancel_event)
                except Exception:
                    logger.warning(
                        "tournament.cancel.publish_failed",
                        extra={"tournament_id": tournament_id},
                        exc_info=True,
                    )

    async def cancel_tournament(
        self,
        user: User,
        tournament_id: int,
        reason_detail: str | None = None,
    ) -> None:
        """User-facing cancel entry point.

        Called by REST POST /api/v1/tournaments/{id}/cancel and the MCP
        `cancel_tournament` tool. Authorization runs against an unlocked
        SELECT before the write transaction.
        """
        await self._load_for_auth(tournament_id, user)

        async with self._session.begin():
            event = await self._cancel_impl(
                tournament_id,
                reason=CancelReason.ADMIN_ACTION,
                cancelled_by=user.id,
                reason_detail=reason_detail,
            )

        if event is not None:
            try:
                await self._bus.publish(event)
            except Exception:
                logger.warning(
                    "tournament.cancel.publish_failed",
                    extra={"tournament_id": tournament_id},
                    exc_info=True,
                )

    async def cancel_tournament_system(
        self,
        tournament_id: int,
        reason: CancelReason,
        reason_detail: str | None = None,
    ) -> None:
        """System-initiated cancel. Called ONLY by the deadline worker
        (pending_timeout path) and `leave()` (abandoned cascade).

        Code-review invariant: no handler file imports this method.
        Enforced by tests/unit/dashboard/tournament/test_static_guards.py.
        """
        async with self._session.begin():
            event = await self._cancel_impl(
                tournament_id,
                reason=reason,
                cancelled_by=None,
                reason_detail=reason_detail,
            )

        if event is not None:
            try:
                await self._bus.publish(event)
            except Exception:
                logger.warning(
                    "tournament.cancel.publish_failed",
                    extra={"tournament_id": tournament_id},
                    exc_info=True,
                )

    async def _load_for_auth(
        self,
        tournament_id: int,
        user: User,
    ) -> Tournament:
        """Load tournament and verify that `user` is authorized to act on it.

        Authorization rule:
        - Admins (user.is_admin): always allowed.
        - Owners (tournament.created_by == user.id): allowed.
        - Legacy with no owner (tournament.created_by IS NULL): admin only.
        - Everyone else: denied.

        All denial cases raise NotFoundError — same exception as
        "tournament doesn't exist". Preserves the enumeration-guard
        invariant: unauthorized callers cannot distinguish between
        "doesn't exist" and "not allowed".
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if user.is_admin:
            return tournament

        if tournament.created_by is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if tournament.created_by != user.id:
            raise NotFoundError(f"tournament {tournament_id}")

        return tournament

    async def _cancel_impl(
        self,
        tournament_id: int,
        reason: CancelReason,
        cancelled_by: int | None,
        reason_detail: str | None,
    ) -> TournamentCancelEvent | None:
        """Shared cancellation logic. Single source of truth.

        Mutates DB state but does NOT commit — caller owns the transaction.
        Does NOT publish to bus — returns the event for the caller to
        publish after its commit succeeds.

        Returns None if the tournament was already in a terminal state
        (idempotent no-op); returns a TournamentCancelEvent if the call
        caused a state transition.
        """
        # Step 1: Lock + load tournament
        tournament = await self._session.get(
            Tournament, tournament_id, with_for_update=True
        )
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        # Step 2: Idempotent guard
        if tournament.status in (
            TournamentStatus.CANCELLED,
            TournamentStatus.COMPLETED,
        ):
            return None

        # Step 3: Snapshot final_rounds_played BEFORE step 6 bulk UPDATE.
        # Otherwise the count would include in-flight rounds that step 6
        # transitions to CANCELLED.
        final_rounds_played = (
            await self._session.scalar(
                select(func.count())
                .select_from(Round)
                .where(Round.tournament_id == tournament_id)
                .where(Round.status == RoundStatus.COMPLETED)
            )
            or 0
        )

        # Step 4: Write tournament audit fields
        now = datetime.utcnow()
        tournament.status = TournamentStatus.CANCELLED
        tournament.cancelled_at = now
        tournament.cancelled_by = cancelled_by
        tournament.cancelled_reason = reason
        tournament.cancelled_reason_detail = reason_detail

        # Step 5: Release all unreleased participants (bulk UPDATE)
        await self._session.execute(
            update(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.released_at.is_(None))
            .values(released_at=now)
        )

        # Step 6: Cancel all in-flight rounds (bulk UPDATE)
        await self._session.execute(
            update(Round)
            .where(Round.tournament_id == tournament_id)
            .where(
                Round.status.in_(
                    [
                        RoundStatus.WAITING_FOR_ACTIONS,
                        RoundStatus.IN_PROGRESS,
                    ]
                )
            )
            .values(status=RoundStatus.CANCELLED)
        )

        # Step 7: Build event (caller publishes post-commit)
        return TournamentCancelEvent(
            tournament_id=tournament_id,
            cancelled_at=now,
            cancelled_by=cancelled_by,
            cancelled_reason=reason,
            cancelled_reason_detail=reason_detail,
            final_rounds_played=final_rounds_played,
            final_status=TournamentStatus.CANCELLED,
        )
