"""TournamentService — protocol-agnostic core for tournament gameplay.

This module knows about SQLAlchemy and game-environments. It does NOT
know about FastAPI, FastMCP, or HTTP. Unit-tested via direct calls
with an in-memory session and a test event bus.

This is the v1 vertical slice version: only the methods needed for a
2-player 3-round PD e2e test. Plan 2 expands the surface (deadline
worker, leave/get_history/list, AD-9/AD-10 enforcement, etc.).
"""

from __future__ import annotations

import functools
import logging
import os
import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from game_envs.games.battle_of_sexes import BattleOfSexes
from game_envs.games.el_farol import MAX_SLOTS_PER_DAY, ElFarolBar, ElFarolConfig
from game_envs.games.prisoners_dilemma import PrisonersDilemma
from game_envs.games.public_goods import PGConfig, PublicGoodsGame
from game_envs.games.stag_hunt import StagHunt
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, GameResult, User
from atp.dashboard.tournament.builtins import (
    BuiltinNotFoundError,
    resolve_builtin,
)
from atp.dashboard.tournament.errors import (
    ConcurrentPrivateCapExceededError,
    ConflictError,
    NotFoundError,
    RosterValidationError,
    ValidationError,
)
from atp.dashboard.tournament.events import (
    TournamentCancelEvent,
    TournamentEvent,
    TournamentEventBus,
)
from atp.dashboard.tournament.models import (
    Action,
    ActionSource,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.schemas import (
    BoSRoundState,
    ElFarolAction,
    ElFarolRoundState,
    PDRoundState,
    PGRoundState,
    SHRoundState,
    TournamentAction,
)
from atp.dashboard.tournament.schemas import (
    RoundState as _RS_UNION,
)

logger = logging.getLogger("atp.dashboard.tournament.service")


def _utc_now() -> datetime:
    """Current UTC time as naive datetime (SQLite-compatible).

    Uses ``datetime.now(UTC)`` (no deprecation warning), then strips
    tzinfo so comparisons with existing naive DB values work correctly.
    """
    return datetime.now(UTC).replace(tzinfo=None)


TOURNAMENT_PENDING_MAX_WAIT_S: int = int(
    os.environ.get("ATP_TOURNAMENT_PENDING_MAX_WAIT_S", "300")
)

SUPPORTED_GAMES: frozenset[str] = frozenset(
    {
        "prisoners_dilemma",
        "el_farol",
        "stag_hunt",
        "battle_of_sexes",
        "public_goods",
    }
)
# Back-compat alias for private usage within this module; external
# callers should import ``SUPPORTED_GAMES`` instead.
_SUPPORTED_GAMES = SUPPORTED_GAMES

_PD_SINGLETON: PrisonersDilemma = PrisonersDilemma()
_SH_SINGLETON: StagHunt = StagHunt()
_BOS_SINGLETON: BattleOfSexes = BattleOfSexes()

# Hardcoded El Farol V1 preset (spec §3.2).
_EL_FAROL_V1_NUM_SLOTS = 16
_EL_FAROL_V1_THRESHOLD_RATIO = 0.6
_EL_FAROL_V1_MIN_TOTAL_HOURS = 0

# Startup assert — spec §12 silent-min_total_hours mitigation.
assert _EL_FAROL_V1_MIN_TOTAL_HOURS == 0, (
    "Raising _EL_FAROL_V1_MIN_TOTAL_HOURS without a Phase C "
    "finalize_scores hook would silently ignore DQ. See spec §12."
)


@functools.lru_cache(maxsize=64)
def _el_farol_for(num_players: int) -> ElFarolBar:
    cap = max(1, int(_EL_FAROL_V1_THRESHOLD_RATIO * num_players))
    cfg = ElFarolConfig(
        num_players=num_players,
        num_slots=_EL_FAROL_V1_NUM_SLOTS,
        capacity_threshold=cap,
        min_total_hours=_EL_FAROL_V1_MIN_TOTAL_HOURS,
    )
    return ElFarolBar(cfg)


@functools.lru_cache(maxsize=64)
def _pg_for(num_players: int) -> PublicGoodsGame:
    """Cached Public Goods engine for a given player count.

    Deliberately uses ``PGConfig`` defaults — endowment=20, multiplier=1.6,
    punishment_cost=0 and punishment_effect=0 — so the tournament runs
    the simple single-step variant (``_step_basic``). Per-tournament
    config (custom endowment/multiplier) is a future extension; matches
    the El Farol pattern above where threshold is hardcoded v1-style.
    """
    return PublicGoodsGame(PGConfig(num_players=num_players))


def _game_for(tournament: Any) -> Any:
    """Return the Game instance for a tournament.

    PD/SH/BoS use module-level singletons; N-player games (El Farol,
    Public Goods) use per-num_players cached factories. Unknown
    game_type raises ValidationError.
    """
    gt = tournament.game_type
    if gt == "prisoners_dilemma":
        return _PD_SINGLETON
    if gt == "stag_hunt":
        return _SH_SINGLETON
    if gt == "battle_of_sexes":
        return _BOS_SINGLETON
    if gt == "el_farol":
        return _el_farol_for(tournament.num_players)
    if gt == "public_goods":
        return _pg_for(tournament.num_players)
    raise ValidationError(f"unsupported game_type {gt!r}")


_ACTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(TournamentAction)
_ROUND_STATE_ADAPTER: TypeAdapter[
    PDRoundState | SHRoundState | BoSRoundState | ElFarolRoundState | PGRoundState
] = TypeAdapter(_RS_UNION)


def _action_hint_for(game_type: str) -> str:
    """Human-readable expected-shape hint for error messages (spec §4)."""
    if game_type == "prisoners_dilemma":
        return "{choice: 'cooperate' | 'defect'}"
    if game_type == "stag_hunt":
        return "{choice: 'stag' | 'hare'}"
    if game_type == "battle_of_sexes":
        return "{choice: 'A' | 'B'}"
    if game_type == "el_farol":
        return (
            "{intervals: list[[start, end]], up to 2 non-adjacent inclusive "
            f"pairs with 0 <= start <= end < num_slots, max {MAX_SLOTS_PER_DAY} "
            "slots total; [] = stay home}"
        )
    if game_type == "public_goods":
        return "{contribution: float in [0, endowment]}"
    return "{} (unknown game_type)"


class TournamentService:
    def __init__(self, session: AsyncSession, bus: TournamentEventBus) -> None:
        self._session = session
        self._bus = bus

    async def create_tournament(
        self,
        creator: User,
        *,
        name: str,
        game_type: str,
        num_players: int,
        total_rounds: int,
        round_deadline_s: int,
        private: bool = False,
        roster: list[str] | None = None,
    ) -> tuple[Tournament, str | None]:
        """Create a tournament. AD-9 duration cap validation, pending_deadline,
        optional join_token.

        Does NOT auto-join the creator.

        ``roster`` (LABS-TSA PR-4) is an optional list of namespaced
        builtin strategy names (e.g. ``["el_farol/traditionalist"]``)
        that will be inserted as Participant rows with
        ``builtin_strategy=...`` so private test tournaments can
        spin up against deterministic sparring partners.

        Private tournaments enforce:
        * Every roster entry is a known builtin for this game_type.
        * Roster size does not exceed ``num_players``.
        * Creator has at least one tournament-purpose agent OR the
          roster fills every slot (so the tournament actually has
          enough participants to resolve).
        * Creator is under the concurrent-private cap
          (``ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER``) —
          raises ``ConcurrentPrivateCapExceededError``.
        """
        roster = list(roster or [])
        if game_type not in _SUPPORTED_GAMES:
            raise ValidationError(
                f"unsupported game_type {game_type!r}; "
                f"supports: {sorted(_SUPPORTED_GAMES)}"
            )
        if game_type == "prisoners_dilemma":
            if num_players != 2:
                raise ValidationError(
                    f"prisoners_dilemma requires exactly 2 players, got {num_players}"
                )
        elif game_type == "stag_hunt":
            if num_players != 2:
                raise ValidationError(
                    f"stag_hunt requires exactly 2 players, got {num_players}"
                )
        elif game_type == "battle_of_sexes":
            if num_players != 2:
                raise ValidationError(
                    f"battle_of_sexes requires exactly 2 players, got {num_players}"
                )
        elif game_type == "el_farol":
            if not (2 <= num_players <= 20):
                raise ValidationError(
                    f"el_farol requires 2 <= num_players <= 20 (phase B bound), "
                    f"got {num_players}"
                )
        elif game_type == "public_goods":
            # Engine itself enforces 2..20 in PGConfig.__post_init__, but
            # fail here first for a nicer error message and to avoid a
            # half-constructed cache entry.
            if not (2 <= num_players <= 20):
                raise ValidationError(
                    f"public_goods requires 2 <= num_players <= 20, got {num_players}"
                )
        if total_rounds < 1:
            raise ValidationError("total_rounds must be >= 1")
        if round_deadline_s < 1:
            raise ValidationError("round_deadline_s must be >= 1")

        # AD-9 duration cap
        token_expire_minutes = int(os.environ.get("ATP_TOKEN_EXPIRE_MINUTES", "60"))
        max_wall_clock = TOURNAMENT_PENDING_MAX_WAIT_S + total_rounds * round_deadline_s
        budget = (token_expire_minutes - 10) * 60
        if max_wall_clock > budget:
            raise ValidationError(
                f"max duration {budget}s (pending {TOURNAMENT_PENDING_MAX_WAIT_S}s "
                f"+ {total_rounds} rounds × {round_deadline_s}s = "
                f"{max_wall_clock}s) exceeds "
                f"(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60 = {budget}s cap. "
                f"Reduce total_rounds or round_deadline_s."
            )

        # --- LABS-TSA PR-4: roster validation (size, namespacing, lookup) ---
        if len(roster) > num_players:
            raise RosterValidationError(
                f"builtin roster ({len(roster)}) larger than "
                f"num_players ({num_players})"
            )
        for entry in roster:
            if "/" not in entry:
                raise RosterValidationError(
                    f"unknown builtin strategy {entry!r} "
                    "(must be namespaced as 'game/name')"
                )
            prefix, _ = entry.split("/", 1)
            if prefix != game_type:
                raise RosterValidationError(
                    f"unknown builtin strategy {entry!r} "
                    f"for game {game_type!r} (namespace mismatch)"
                )
            try:
                # Probe resolution with dummy ids — the instance is
                # discarded; only the class lookup matters.
                resolve_builtin(entry, tournament_id=0, participant_id=0)
            except BuiltinNotFoundError as exc:
                raise RosterValidationError(
                    f"unknown builtin strategy {entry!r}: {exc}"
                ) from exc

        # --- LABS-TSA PR-4: creator-commit check for private tournaments ---
        if private:
            creator_tournament_agents = await self._session.scalar(
                select(func.count(Agent.id)).where(
                    Agent.owner_id == creator.id,
                    Agent.purpose == "tournament",
                    Agent.deleted_at.is_(None),
                )
            )
            agent_count = int(creator_tournament_agents or 0)
            has_agent = agent_count > 0
            roster_fills_all = len(roster) >= num_players
            if not has_agent and not roster_fills_all:
                missing = num_players - len(roster)
                raise RosterValidationError(
                    f"private tournament needs {num_players} participants; "
                    f"you have {agent_count} tournament agents and "
                    f"{len(roster)} builtin(s), leaving {missing} slot(s) "
                    "unfilled. Register a tournament-purpose agent via "
                    "/ui/agents, or extend the builtin roster until "
                    f"len(roster) == num_players ({num_players})."
                )

        # --- LABS-TSA PR-4: concurrent-private cap ---
        if private:
            from atp.dashboard.v2.config import get_config

            cfg = get_config()
            active = await self._session.scalar(
                select(func.count(Tournament.id)).where(
                    Tournament.created_by == creator.id,
                    Tournament.join_token.is_not(None),
                    Tournament.status.in_(
                        [TournamentStatus.PENDING, TournamentStatus.ACTIVE]
                    ),
                    # Exclude expired-pending so deadlines.py auto-cancel
                    # doesn't race the cap.
                    (Tournament.status != TournamentStatus.PENDING)
                    | (Tournament.pending_deadline > _utc_now()),
                )
            )
            current = int(active or 0)
            cap = cfg.max_concurrent_private_tournaments_per_user
            if current >= cap:
                raise ConcurrentPrivateCapExceededError(current, cap)

        now = _utc_now()
        pending_deadline = now + timedelta(seconds=TOURNAMENT_PENDING_MAX_WAIT_S)

        join_token_plaintext: str | None = None
        if private:
            join_token_plaintext = secrets.token_urlsafe(32)

        tournament = Tournament(
            game_type=game_type,
            status=TournamentStatus.PENDING,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            created_by=creator.id,
            config={"name": name},
            pending_deadline=pending_deadline,
            join_token=join_token_plaintext,
        )
        # Caller owns the transaction boundary (ambient autobegin from FastAPI
        # DB dependency). Do NOT wrap in begin() here — that would nest
        # transactions and conflict with the ambient session.
        self._session.add(tournament)
        await self._session.flush()

        # --- LABS-TSA PR-4: insert builtin participants ---
        for entry in roster:
            self._session.add(
                Participant(
                    tournament_id=tournament.id,
                    user_id=None,
                    agent_id=None,
                    agent_name=entry,
                    builtin_strategy=entry,
                )
            )
        if roster:
            await self._session.flush()
            # If the roster already fills every slot, transition straight
            # to ACTIVE so downstream deadline/worker logic can pick up
            # immediately. Mirrors the count-based transition in ``join``.
            count = await self._session.scalar(
                select(func.count(Participant.id)).where(
                    Participant.tournament_id == tournament.id
                )
            )
            if count == num_players:
                await self._start_tournament(tournament)

        return tournament, join_token_plaintext

    async def join(
        self,
        tournament_id: int,
        user: User,
        agent_name: str,
        join_token: str | None = None,
        *,
        agent_id: int | None = None,
    ) -> tuple[Participant, bool]:
        """Idempotent join with race-aware IntegrityError handling.

        Returns (participant, is_new).

        LABS-TSA PR-6: uniqueness is agent-keyed, not user-keyed. A
        single user may register multiple tournament agents and play
        them all in the same tournament.

        LABS-TSA PR-6 (post-review): ``agent_id`` is the clean-path
        input from MCP agent-scoped tokens — the caller has already
        identified the exact Agent row and we skip name-based lookup
        entirely. When ``agent_id is None`` (CLI/admin callers), we
        fall back to the by-name lookup + auto-provision path, which
        now enforces ``ATP_MAX_TOURNAMENT_AGENTS_PER_USER`` and runs
        AFTER join validation so a rejected join cannot orphan an
        Agent row.

        Race semantics on INSERT IntegrityError:
        - uq_participant_tournament_agent violation: either a concurrent
          idempotent re-join of the same agent won the insert race, or
          the agent previously participated in this tournament and was
          released. Re-read; if an active row exists return it, else
          raise ConflictError("previously participated").
        - uq_participant_agent_active violation: the agent has active
          participation in a *different* tournament. Raise ConflictError(409).
        - Any other IntegrityError: re-raise unchanged.
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        # Capture scalar values before any potential rollback expiry
        user_id = user.id

        # --- Step 1: validate the join itself BEFORE any auto-provision ---
        # Previously the service auto-provisioned an Agent row before it
        # knew whether the join would succeed. A failed join
        # (wrong join_token, agent already active elsewhere) would leave
        # an orphan Agent consuming a quota slot. We now validate first.
        # The idempotent pre-check for agent_id-supplied callers happens
        # inside Step 2 below; for name-based callers the pre-check runs
        # after resolution (also in Step 2).

        resolved_agent_id: int | None = agent_id

        # --- Step 2: resolve agent_id ---
        if resolved_agent_id is None:
            # By-name resolution for CLI/admin callers. MCP clients
            # pass agent_id explicitly and skip this branch.
            agent_lookup_filter = (
                Agent.owner_id == user_id,
                Agent.tenant_id == DEFAULT_TENANT_ID,
                Agent.name == agent_name,
                Agent.purpose == "tournament",
                Agent.deleted_at.is_(None),
            )
            resolved_agent_id = await self._session.scalar(
                select(Agent.id).where(*agent_lookup_filter).limit(1)
            )
            needs_auto_provision = resolved_agent_id is None
        else:
            needs_auto_provision = False
            agent_lookup_filter = ()  # unused on the explicit-agent_id path

        # Idempotent pre-check (only meaningful once we already have a
        # concrete agent_id — either supplied directly or resolved from
        # an existing Agent row). The auto-provision branch inserts a
        # brand-new Agent so it cannot have any prior Participant rows.
        if resolved_agent_id is not None:
            existing = await self._session.scalar(
                select(Participant)
                .where(Participant.tournament_id == tournament_id)
                .where(Participant.agent_id == resolved_agent_id)
            )
            if existing is not None:
                if existing.released_at is not None:
                    # Leave is terminal — cannot rejoin.
                    raise ConflictError(
                        f"agent {agent_name!r} previously participated in "
                        f"tournament {tournament_id} "
                        f"(released={existing.released_at.isoformat()}); "
                        "rejoin not permitted"
                    )
                return existing, False

        # --- Step 3: validate tournament status and join_token ---
        if tournament.status != TournamentStatus.PENDING:
            raise ConflictError(
                f"tournament {tournament_id} is {tournament.status!r}, "
                "not accepting joins"
            )

        if tournament.join_token is not None:
            import hmac

            if join_token is None or not hmac.compare_digest(
                tournament.join_token, join_token
            ):
                raise ConflictError(
                    f"tournament {tournament_id} requires a valid join_token"
                )

        # --- Step 4: auto-provision (only for by-name callers, only
        # after validation succeeded) ---
        if needs_auto_provision:
            from atp.dashboard.v2.config import get_config

            cfg = get_config()
            cap = cfg.max_tournament_agents_per_user
            existing_count = await self._session.scalar(
                select(func.count(Agent.id)).where(
                    Agent.owner_id == user_id,
                    Agent.purpose == "tournament",
                    Agent.deleted_at.is_(None),
                )
            )
            current = int(existing_count or 0)
            if current >= cap:
                raise ConflictError(
                    f"tournament agent quota exceeded ({current}/{cap})"
                )

            # Race-tolerant auto-provision: another concurrent join may
            # win the insert race on the agent's UNIQUE
            # (tenant_id, owner_id, name, version) index. Use a
            # savepoint so an IntegrityError doesn't tear down the
            # outer transaction — we simply re-read the row and
            # continue.
            auto_agent = Agent(
                tenant_id=DEFAULT_TENANT_ID,
                name=agent_name,
                agent_type="mcp",
                owner_id=user_id,
                config={},
                purpose="tournament",
            )
            try:
                async with self._session.begin_nested():
                    self._session.add(auto_agent)
                    await self._session.flush()
                resolved_agent_id = auto_agent.id
            except IntegrityError:
                # Re-read with the same purpose-aware filter so we pick
                # up the concurrent insert, not a stale benchmark-
                # purpose agent that happens to share the name.
                resolved_agent_id = await self._session.scalar(
                    select(Agent.id).where(*agent_lookup_filter).limit(1)
                )
                if resolved_agent_id is None:
                    # Reread failed — propagate the race
                    raise

            # Re-check idempotency now that we have an agent_id —
            # unlikely but possible if two concurrent joins race past
            # the auto-provision block.
            existing = await self._session.scalar(
                select(Participant)
                .where(Participant.tournament_id == tournament_id)
                .where(Participant.agent_id == resolved_agent_id)
            )
            if existing is not None:
                if existing.released_at is not None:
                    raise ConflictError(
                        f"agent {agent_name!r} previously participated in "
                        f"tournament {tournament_id} "
                        f"(released={existing.released_at.isoformat()}); "
                        "rejoin not permitted"
                    )
                return existing, False

        # --- Step 5: INSERT path ---
        assert resolved_agent_id is not None
        participant = Participant(
            tournament_id=tournament_id,
            user_id=user_id,
            agent_name=agent_name,
            agent_id=resolved_agent_id,
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
            if constraint_name == "uq_participant_tournament_agent":
                # Composite (tournament_id, agent_id) uniqueness: fires
                # on either a concurrent idempotent re-join (active row
                # exists) OR a rejoin-after-leave (released row still
                # exists). Disambiguate via a lookup that does NOT
                # filter on released_at.
                existing = await self._session.scalar(
                    select(Participant)
                    .where(Participant.tournament_id == tournament_id)
                    .where(Participant.agent_id == resolved_agent_id)
                )
                if existing is not None and existing.released_at is None:
                    return existing, False
                if existing is not None and existing.released_at is not None:
                    raise ConflictError(
                        f"agent {agent_name!r} previously participated in "
                        f"tournament {tournament_id} "
                        f"(released={existing.released_at.isoformat()}); "
                        "rejoin not permitted"
                    ) from exc
                # Row vanished between INSERT and re-read — extremely
                # unlikely but defend with a generic conflict.
                raise ConflictError(
                    f"agent {agent_name!r} conflict in tournament {tournament_id}"
                ) from exc
            if constraint_name == "uq_participant_agent_active":
                # Agent is active *somewhere*. Distinguish same-
                # tournament (concurrent idempotent re-join) from
                # different-tournament (true conflict). Either row
                # satisfies the partial unique index; inspect the
                # existing one.
                other = await self._session.scalar(
                    select(Participant)
                    .where(Participant.agent_id == resolved_agent_id)
                    .where(Participant.released_at.is_(None))
                )
                if other is not None and other.tournament_id == tournament_id:
                    # Same tournament — idempotent re-join.
                    return other, False
                if other is not None:
                    raise ConflictError(
                        f"agent {other.agent_name!r} already active in "
                        f"tournament {other.tournament_id}"
                    ) from exc
                raise ConflictError(
                    f"agent {agent_name!r} already has an active tournament"
                ) from exc
            raise

        return participant, True

    @staticmethod
    def _extract_constraint_name(exc: Exception) -> str:
        """Best-effort constraint name extraction from IntegrityError.

        SQLite reports: 'UNIQUE constraint failed: table.col1, table.col2'
        or 'UNIQUE constraint failed: table.col' for single-column indices.
        Named constraints may appear as 'constraint failed: <name>'.
        PostgreSQL: exc.orig.diag.constraint_name is available.

        Disambiguation for tournament_participants (LABS-TSA PR-6):
        - uq_participant_agent_active: single-column partial index on agent_id
          → message contains only 'tournament_participants.agent_id'
        - uq_participant_tournament_agent: two-column partial index on
          (tournament_id, agent_id) → message contains 'tournament_id'
        """
        # Extract only the first line of the error — the constraint description.
        # SQLite format: '(sqlite3.IntegrityError) UNIQUE constraint failed: table.col'
        # The full message includes the SQL statement which can contaminate searches.
        first_line = str(exc).split("\n")[0].lower()
        # Named index/constraint — check for explicit name first
        for name in (
            "uq_participant_agent_active",
            "uq_participant_tournament_agent",
            "uq_action_round_participant",
            "uq_round_tournament_number",
        ):
            if name in first_line:
                return name
        # SQLite column-based disambiguation for tournament_participants.
        # The partial unique index on agent_id (uq_participant_agent_active)
        # shows only 'tournament_participants.agent_id' in the first line
        # (single column). The composite (tournament_id, agent_id) partial
        # unique index shows both 'tournament_participants.tournament_id'
        # and 'tournament_participants.agent_id'.
        if "tournament_participants" in first_line and "agent_id" in first_line:
            if "tournament_id" in first_line:
                # composite (tournament_id, agent_id) →
                # uq_participant_tournament_agent
                return "uq_participant_tournament_agent"
            else:
                # single agent_id partial index → uq_participant_agent_active
                return "uq_participant_agent_active"
        # PostgreSQL path
        orig = getattr(exc, "orig", None)
        diag = getattr(orig, "diag", None)
        if diag is not None:
            return getattr(diag, "constraint_name", "") or ""
        return ""

    async def _start_tournament(self, tournament: Tournament) -> None:
        """Transition a PENDING tournament to ACTIVE and create round 1."""
        now = _utc_now()
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
        # Commit (not just flush) BEFORE publishing round_started —
        # same invariant the subsequent-round branch of
        # ``_resolve_round`` enforces (LABS-74). The notification
        # forwarder opens a fresh DB session to build the per-player
        # state snapshot; without the commit that session reads
        # pre-commit rows and the snapshot's state trails the outer
        # event.round_number by one. The MCP ``join`` shim also
        # subscribes to the event bus *before* calling
        # ``service.join()``, so a pre-commit publish could deliver a
        # first-round state that hasn't yet been persisted.
        await self._session.commit()
        await self._bus.publish(
            TournamentEvent(
                event_type="round_started",
                tournament_id=tournament.id,
                round_number=1,
                data={"total_rounds": tournament.total_rounds},
                timestamp=_utc_now(),
            )
        )

    async def get_state_for(
        self,
        tournament_id: int,
        user: User,
        *,
        agent_id: int | None = None,
    ) -> PDRoundState | SHRoundState | BoSRoundState | ElFarolRoundState | PGRoundState:
        """Build a player-private RoundState for the current round.

        Returns a pydantic ``PDRoundState`` or ``ElFarolRoundState``
        (see ``atp.dashboard.tournament.schemas``), chosen by the
        tournament's game_type discriminator.

        LABS-TSA PR-6 (post-review): ``agent_id`` selects the caller's
        Participant when the user has multiple agents in the same
        tournament. The MCP path always passes it (extracted from
        ``scope.state['agent_id']`` by the caller). When None, the
        legacy single-agent path looks up the first Participant whose
        ``user_id`` matches — preserved for backwards compatibility
        with admin/CLI call sites that don't own an agent_id.

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
        if agent_id is not None:
            # Agent-keyed lookup — required for MCP multi-agent-per-user.
            my_idx = next(
                (i for i, p in enumerate(participants) if p.agent_id == agent_id),
                None,
            )
        else:
            # Legacy single-agent path: match by user_id. If the user has
            # multiple agents, the first-registered Participant wins. MCP
            # callers always pass agent_id to avoid this ambiguity.
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

        action_history: list[dict[str, Any]] = []
        cumulative_scores: list[float] = [0.0] * len(participants)
        current_round_number = 1
        found_active = False
        for r in rounds:
            if r.status == RoundStatus.COMPLETED:
                actions_by_idx: dict[int, dict[str, Any]] = {}
                for action in r.actions:
                    p_idx = next(
                        i
                        for i, p in enumerate(participants)
                        if p.id == action.participant_id
                    )
                    actions_by_idx[p_idx] = action.action_data
                    cumulative_scores[p_idx] += action.payoff or 0.0
                action_history.append(
                    {"round": r.round_number, "actions": actions_by_idx}
                )
            else:
                current_round_number = r.round_number
                found_active = True
                break
        if not found_active and rounds:
            current_round_number = len(rounds)

        game = _game_for(tournament)
        formatted = game.format_state_for_player(
            round_number=current_round_number,
            total_rounds=tournament.total_rounds,
            participant_idx=my_idx,
            action_history=action_history,
            cumulative_scores=cumulative_scores,
        )

        # Compute submission state for the active round by inspecting
        # the already-loaded rounds+actions (no extra DB round-trip).
        my_participant_id = participants[my_idx].id
        if found_active:
            has_submitted = False
            for r in rounds:
                if r.status == RoundStatus.WAITING_FOR_ACTIONS:
                    for a in r.actions:
                        if a.participant_id == my_participant_id:
                            has_submitted = True
                            break
                    break
        else:
            # No active round (tournament completed, pending, or just
            # created). Nothing to submit — flip flags to False by
            # treating the player as already submitted.
            has_submitted = True

        # Inject game-specific submission-state field (spec §3.2 step 6).
        if tournament.game_type in (
            "prisoners_dilemma",
            "stag_hunt",
            "battle_of_sexes",
        ):
            # Both are 2-player simultaneous discrete-choice games; they
            # share the "your_turn" flag semantics.
            formatted["your_turn"] = not has_submitted
        elif tournament.game_type in ("el_farol", "public_goods"):
            # Both are N-player simultaneous games; client polls via
            # ``pending_submission`` rather than the 2-player
            # ``your_turn`` flag.
            formatted["pending_submission"] = not has_submitted
        else:
            raise ValidationError(f"unsupported game_type {tournament.game_type!r}")

        # Strip internal-only keys that the wire schemas forbid
        # (extra="forbid"), then set authoritative server-side fields.
        formatted.pop("extra", None)
        formatted["tournament_id"] = tournament_id
        formatted["game_type"] = tournament.game_type

        return _ROUND_STATE_ADAPTER.validate_python(formatted)

    async def submit_action(
        self,
        tournament_id: int,
        user: User,
        action: dict[str, Any],
        *,
        agent_id: int | None = None,
    ) -> dict[str, Any]:
        """Submit one player's action for the current round.

        LABS-TSA PR-6 (post-review): ``agent_id`` selects the caller's
        Participant when the user has multiple agents in the same
        tournament. When None, the legacy single-agent path matches by
        user_id with ``.limit(1)`` — admin/CLI callers that never used
        agent-scoped tokens stay working, but concurrent multi-agent
        setups must pass agent_id to avoid the ambiguity.

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

        # Server-side game_type enforcement (spec §4).
        # Reject cross-game payloads early; inject the server-authoritative
        # game_type so clients may omit it on the wire.
        incoming_gt = action.get("game_type") if isinstance(action, dict) else None
        if incoming_gt is not None and incoming_gt != tournament.game_type:
            logger.info(
                "action_rejected",
                extra={
                    "event": "action_rejected",
                    "game_type": tournament.game_type,
                    "tournament_id": tournament_id,
                    "validation_error_path": "game_type_mismatch",
                },
            )
            raise ValidationError(
                f"action game_type {incoming_gt!r} does not match "
                f"tournament {tournament_id} game_type {tournament.game_type!r}"
            )
        if isinstance(action, dict):
            action_with_type = {**action, "game_type": tournament.game_type}
        else:
            action_with_type = action

        try:
            typed = _ACTION_ADAPTER.validate_python(action_with_type)
        except PydanticValidationError as e:
            expected = _action_hint_for(tournament.game_type)
            errors = e.errors()
            first_err = errors[0] if errors else {"msg": "unknown"}
            logger.info(
                "action_rejected",
                extra={
                    "event": "action_rejected",
                    "game_type": tournament.game_type,
                    "tournament_id": tournament_id,
                    "validation_error_path": "client_submission",
                },
            )
            raise ValidationError(
                f"invalid action for tournament {tournament_id} "
                f"(game_type={tournament.game_type!r}); "
                f"expected: {expected}; "
                f"pydantic: {first_err.get('loc')}: {first_err.get('msg')}"
            ) from e

        if agent_id is not None:
            # Agent-keyed lookup (MCP multi-agent-per-user path).
            my_participant = (
                await self._session.execute(
                    select(Participant).where(
                        Participant.tournament_id == tournament_id,
                        Participant.agent_id == agent_id,
                    )
                )
            ).scalar_one_or_none()
        else:
            # Legacy single-agent path. ``.limit(1)`` guards against the
            # ambiguity window where a user has >1 Participant row in the
            # tournament — callers on this path are CLI/admin tooling
            # that has never supported multiple agents. MCP clients
            # always pass agent_id.
            my_participant = (
                await self._session.execute(
                    select(Participant)
                    .where(
                        Participant.tournament_id == tournament_id,
                        Participant.user_id == user.id,
                    )
                    .order_by(Participant.id)
                    .limit(1)
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

        raw_reasoning = getattr(typed, "reasoning", None)
        reasoning = raw_reasoning.strip() if raw_reasoning else None
        reasoning = reasoning or None  # "" / whitespace-only → None

        game = _game_for(tournament)
        # El Farol wire format is intervals; the game-env strict path
        # (validate_action) expects flat slots. Bridge here so the
        # game-env contract stays unchanged.
        if isinstance(typed, ElFarolAction):
            game_payload: dict[str, Any] = {"slots": typed.to_slots()}
        else:
            game_payload = typed.model_dump(
                exclude={"game_type", "reasoning", "telemetry"}
            )
        canonical = game.validate_action(game_payload)

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
            action_data=canonical,
            reasoning=reasoning,
        )
        self._session.add(new_action)
        await self._session.flush()

        # LABS-TSA PR-4: fill in any builtin-participant moves so a
        # mixed roster (agents + builtins) does not stall when the
        # last human submits. _resolve_round is idempotent on this
        # insert.
        await self._ensure_builtin_actions(current_round, tournament)

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

    async def _build_observation_for_builtin(
        self,
        *,
        tournament: Tournament,
        round_obj: Round,
        participant: Participant,
        participants: list[Participant],
    ) -> Any:
        """Build a game-specific ``Observation`` for a builtin participant.

        Replays the per-round state from the database (prior rounds'
        resolved actions → ``history`` / attendance / contributions)
        and constructs the Observation dataclass the strategy's
        ``choose_action`` expects.

        Per-game shape:
          * ``el_farol``: ``game_state['attendance_history']`` (list of
            per-slot counts per round) + ``num_slots``.
          * ``prisoners_dilemma`` / ``stag_hunt`` / ``battle_of_sexes``:
            only ``history`` (list of RoundResult) + ``player_id``.
          * ``public_goods``: ``game_state['endowment']`` + ``history``.
        """
        from game_envs.core.state import Observation, RoundResult

        prior_rounds_stmt = (
            select(Round)
            .where(
                Round.tournament_id == tournament.id,
                Round.round_number < round_obj.round_number,
                Round.status == RoundStatus.COMPLETED,
            )
            .order_by(Round.round_number)
            .options(selectinload(Round.actions))
        )
        prior_rounds = (await self._session.execute(prior_rounds_stmt)).scalars().all()
        idx_by_pid = {p.id: i for i, p in enumerate(participants)}
        my_idx = idx_by_pid[participant.id]
        player_id = str(my_idx)

        # Build RoundResult history keyed by "0", "1", ... player_ids.
        history: list[RoundResult] = []
        for r in prior_rounds:
            actions_by_pid: dict[str, Any] = {}
            payoffs_by_pid: dict[str, float] = {}
            for a in r.actions:
                idx = idx_by_pid.get(a.participant_id)
                if idx is None:
                    continue
                actions_by_pid[str(idx)] = a.action_data
                payoffs_by_pid[str(idx)] = float(a.payoff or 0.0)
            history.append(
                RoundResult(
                    round_number=r.round_number,
                    actions=actions_by_pid,
                    payoffs=payoffs_by_pid,
                )
            )

        game = _game_for(tournament)
        available_actions = list(getattr(game, "available_actions", lambda: [])())
        game_state: dict[str, Any] = {}

        if tournament.game_type == "el_farol":
            num_slots = _EL_FAROL_V1_NUM_SLOTS
            attendance_history: list[list[int]] = []
            for r in prior_rounds:
                counts = [0] * num_slots
                for a in r.actions:
                    for s in (a.action_data or {}).get("slots", []):
                        if 0 <= s < num_slots:
                            counts[s] += 1
                attendance_history.append(counts)
            game_state = {
                "attendance_history": attendance_history,
                "num_slots": num_slots,
            }
        elif tournament.game_type == "public_goods":
            pg_game = _pg_for(tournament.num_players)
            pg_cfg: PGConfig = pg_game._pg_config  # typed config view
            game_state = {
                "endowment": pg_cfg.endowment,
                "stage": "contribute",
            }

        return Observation(
            player_id=player_id,
            game_state=game_state,
            available_actions=available_actions,
            history=history,
            round_number=round_obj.round_number,
            total_rounds=tournament.total_rounds,
        )

    def _coerce_builtin_action(self, *, game_type: str, raw: Any) -> dict[str, Any]:
        """Map a Strategy's ``choose_action`` return to canonical action_data.

        Each game has a slightly different action payload shape:
        * PD / SH / BoS → ``{"choice": <str>}``
        * El Farol → ``{"slots": [int, ...]}``
        * Public Goods → ``{"contribution": <float>}``

        Strategies return the bare value (e.g. ``"cooperate"``,
        ``[0,1,2]``, ``5.0``) so we wrap it here.
        """
        if isinstance(raw, dict):
            return raw
        if game_type in ("prisoners_dilemma", "stag_hunt", "battle_of_sexes"):
            return {"choice": raw}
        if game_type == "el_farol":
            slots = list(raw) if raw is not None else []
            return {"slots": [int(s) for s in slots]}
        if game_type == "public_goods":
            return {"contribution": float(raw)}
        # Best-effort fallback
        return {"value": raw}

    async def _ensure_builtin_actions(
        self, round_obj: Round, tournament: Tournament
    ) -> int:
        """Synthesise Action rows for every builtin participant in ``round_obj``.

        Idempotent — existing Action rows are left alone. Returns the
        number of Action rows inserted in this call.

        Must be called whenever we are about to count submitted
        actions (submit_action, force_resolve_round, _resolve_round)
        so builtins never stall progress.
        """
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
        builtins = [p for p in participants if p.builtin_strategy is not None]
        if not builtins:
            return 0

        existing_ids_stmt = select(Action.participant_id).where(
            Action.round_id == round_obj.id
        )
        existing_ids = {
            row[0] for row in await self._session.execute(existing_ids_stmt)
        }

        inserted = 0
        game = _game_for(tournament)
        for p in builtins:
            if p.id in existing_ids:
                continue
            obs = await self._build_observation_for_builtin(
                tournament=tournament,
                round_obj=round_obj,
                participant=p,
                participants=list(participants),
            )
            strategy = resolve_builtin(
                p.builtin_strategy or "",
                tournament_id=tournament.id,
                participant_id=p.id,
            )
            raw = strategy.choose_action(obs)
            action_data = self._coerce_builtin_action(
                game_type=tournament.game_type, raw=raw
            )
            # Validate against the game's action schema; fall back to
            # default_action_on_timeout() if the builtin returns
            # malformed data so a broken strategy can't deadlock a
            # round.
            try:
                canonical = game.validate_action(action_data)
            except Exception:
                canonical = game.default_action_on_timeout()
            self._session.add(
                Action(
                    round_id=round_obj.id,
                    participant_id=p.id,
                    action_data=canonical,
                    source=ActionSource.BUILTIN,
                )
            )
            inserted += 1

        if inserted:
            await self._session.flush()
        return inserted

    async def _resolve_round(
        self, round_obj: Round, tournament: Tournament
    ) -> dict[str, Any]:
        """Resolve a round: compute payoffs, write Action.payoff, mark
        round completed, and either create the next round or finish the
        tournament if this was the last.
        """
        # LABS-TSA PR-4: synthesise builtin moves before counting /
        # resolving so a mixed roster (agents + builtins) never stalls.
        await self._ensure_builtin_actions(round_obj, tournament)
        start = time.perf_counter()
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

        action_by_idx: dict[int, Action] = {}
        action_data_by_idx: dict[int, dict[str, Any]] = {}
        for a in actions:
            i = idx_by_pid[a.participant_id]
            action_by_idx[i] = a
            action_data_by_idx[i] = a.action_data

        game = _game_for(tournament)
        payoffs = game.compute_round_payoffs(action_data_by_idx)

        for i, action in action_by_idx.items():
            action.payoff = payoffs[i]

        round_obj.status = RoundStatus.COMPLETED
        await self._session.flush()

        if round_obj.round_number >= tournament.total_rounds:
            await self._complete_tournament(tournament)
            await self._session.flush()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "round_resolved",
                extra={
                    "event": "round_resolved",
                    "game_type": tournament.game_type,
                    "tournament_id": tournament.id,
                    "round_number": round_obj.round_number,
                    "round_resolution_ms": elapsed_ms,
                },
            )
            return {
                "status": "round_resolved",
                "round_number": round_obj.round_number,
                "tournament_completed": True,
                "payoffs": payoffs,
            }

        now = _utc_now()
        next_round = Round(
            tournament_id=tournament.id,
            round_number=round_obj.round_number + 1,
            status=RoundStatus.WAITING_FOR_ACTIONS,
            started_at=now,
            deadline=now + timedelta(seconds=tournament.round_deadline_s),
            state={},
        )
        self._session.add(next_round)
        # Commit (not just flush) BEFORE publishing round_started. The
        # notification forwarder opens a fresh DB session to build the
        # per-player state snapshot; if we only flushed, that fresh
        # session reads pre-commit and the snapshot's state.round_number
        # trails the outer event.round_number by one. Committing first
        # guarantees subscribers see the new round. (LABS-74.)
        await self._session.commit()
        await self._bus.publish(
            TournamentEvent(
                event_type="round_started",
                tournament_id=tournament.id,
                round_number=next_round.round_number,
                data={"total_rounds": tournament.total_rounds},
                timestamp=_utc_now(),
            )
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "round_resolved",
            extra={
                "event": "round_resolved",
                "game_type": tournament.game_type,
                "tournament_id": tournament.id,
                "round_number": round_obj.round_number,
                "round_resolution_ms": elapsed_ms,
            },
        )
        return {
            "status": "round_resolved",
            "round_number": round_obj.round_number,
            "tournament_completed": False,
            "payoffs": payoffs,
            "next_round_number": next_round.round_number,
        }

    async def _complete_tournament(self, tournament: Tournament) -> None:
        """Mark tournament COMPLETED and write final per-participant scores.

        Also releases every participant (``released_at = now``) so their
        ``agent_id`` is no longer matched by
        ``uq_participant_agent_active`` and they are free to join another
        tournament. Mirrors the release step in ``_cancel_impl``; without
        it, natural completion would leave agents "stuck" active forever.

        LABS-TSA PR-5: writes a companion ``GameResult`` row with a fresh
        UUID ``match_id`` and ``tournament_id = tournament.id`` so
        ``/ui/matches`` can surface tournament outcomes. The UNIQUE
        partial index on ``game_results.tournament_id`` plus a SAVEPOINT
        around the insert make the dual-write idempotent — a second
        completion attempt hits IntegrityError and is absorbed silently.
        The JSON blobs (``actions_json`` etc.) are left as empty lists
        in this PR; a follow-up ticket will port the full Phase-7
        reshape from ``atp/cli/commands/game.py::_build_game_result_kwargs``.
        """
        now = _utc_now()
        tournament.status = TournamentStatus.COMPLETED
        tournament.ends_at = now

        # Dual-write a GameResult row linked to this tournament. Must run
        # before the per-participant release loop so an IntegrityError
        # from a duplicate completion only rolls back the SAVEPOINT, not
        # the outer transaction flipping tournament.status.
        await self._write_game_result_for_tournament(tournament)

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
            if p.released_at is None:
                p.released_at = now
        await self._session.flush()
        # Builtin-strategy Participant rows have ``user_id IS NULL``
        # (LABS-TSA PR-4). Skip them when keying the final_scores dict
        # by user_id — without the filter, multiple builtins collapse
        # into a single ``null`` key and clients expecting int keys
        # (see tests/e2e/test_mcp_pd_tournament.py) get a malformed
        # event. Builtin scores are still available via the tournament
        # detail API for callers that need them.
        await self._bus.publish(
            TournamentEvent(
                event_type="tournament_completed",
                tournament_id=tournament.id,
                round_number=None,
                data={
                    "final_scores": {
                        p.user_id: p.total_score
                        for p in participants
                        if p.user_id is not None
                    },
                },
                timestamp=_utc_now(),
            )
        )

    async def _write_game_result_for_tournament(self, tournament: Tournament) -> None:
        """Dual-write a ``GameResult`` row linked to this tournament.

        LABS-TSA PR-5. Populates the scalar fields only; the four
        Phase-7 JSON columns stay NULL because LABS-106 reshapes the
        dashboard payload at read time from ``Round``/``Action`` ORM
        rows (see ``el_farol_from_tournament._reshape_from_tournament``).
        ``Round``/``Action`` are the single source of truth for
        tournament-produced data — duplicating into JSON blobs would
        only invite consistency drift.

        ``match_id`` is a fresh UUID — **never** the tournament's
        ``join_token`` (that's a secret the creator must share with
        agents out-of-band).

        Wrapped in ``session.begin_nested()`` so the UNIQUE partial
        index ``uq_game_results_tournament_id`` converts a duplicate
        completion into a silent no-op instead of crashing the outer
        transaction that just flipped ``tournament.status``.
        """
        match_id = str(uuid.uuid4())
        variant = (tournament.config or {}).get("variant", "tournament")
        row = GameResult(
            match_id=match_id,
            game_name=tournament.game_type,
            game_type=variant,
            num_players=tournament.num_players,
            num_rounds=tournament.total_rounds,
            num_episodes=1,
            status="completed",
            completed_at=_utc_now(),
            tournament_id=tournament.id,
            # Phase-7 JSON columns stay NULL on purpose — LABS-106
            # reshapes the dashboard payload at read time from
            # Round/Action ORM rows. The /ui/matches listing accepts
            # these rows via the (tournament_id IS NOT NULL AND
            # game_name == "el_farol") branch in renderable_filters
            # (LABS-104, ui.py::ui_matches).
            actions_json=None,
            day_aggregates_json=None,
            round_payoffs_json=None,
            agents_json=None,
        )
        try:
            async with self._session.begin_nested():
                self._session.add(row)
                await self._session.flush()
        except IntegrityError:
            # Idempotency path — a previous completion already wrote the
            # GameResult for this tournament_id (UNIQUE partial index
            # ``uq_game_results_tournament_id``). Fall through silently.
            logger.info(
                "tournament._write_game_result.duplicate",
                extra={
                    "tournament_id": tournament.id,
                    "event": "tournament_completion_dual_write_duplicate",
                },
            )

    async def leave(
        self,
        tournament_id: int,
        user: User,
        *,
        agent_id: int | None = None,
    ) -> None:
        """Mark the caller's Participant as released.

        LABS-TSA PR-6 (post-review): ``agent_id`` selects the caller's
        Participant when the user has multiple agents in the same
        tournament. When None, the legacy single-agent path matches by
        user_id. MCP agent-scoped callers always pass agent_id.

        If the caller is the last active participant of an ACTIVE
        tournament, cascade to _cancel_impl with reason=ABANDONED inside
        the same transaction. Caller owns the transaction boundary.
        """
        participant_stmt = (
            select(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.released_at.is_(None))
        )
        if agent_id is not None:
            participant_stmt = participant_stmt.where(Participant.agent_id == agent_id)
        else:
            participant_stmt = participant_stmt.where(Participant.user_id == user.id)
        participant = await self._session.scalar(participant_stmt)
        if participant is None:
            raise NotFoundError(
                f"user {user.id} is not active in tournament {tournament_id}"
            )

        participant.released_at = _utc_now()
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

        # Caller owns the transaction boundary (ambient autobegin from FastAPI
        # DB dependency or MCP session). Do NOT wrap in begin() here.
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
        # Caller owns the transaction boundary (ambient autobegin from the
        # session_factory() context manager in the deadline worker or test).
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

    async def list_tournaments(
        self,
        user: User,
        status: TournamentStatus | None = None,
        game_type: str | None = None,
    ) -> list[Tournament]:
        """Return tournaments visible to `user`, optionally filtered by
        status and/or game_type.

        Visibility rule:
        - user.is_admin: all tournaments.
        - Else: Tournament.join_token IS NULL OR created_by = user.id
                OR EXISTS participant row for (tournament, user).
        """
        from sqlalchemy import exists, or_

        stmt = select(Tournament)
        if status is not None:
            stmt = stmt.where(Tournament.status == status)
        if game_type is not None:
            stmt = stmt.where(Tournament.game_type == game_type)

        if not user.is_admin:
            stmt = stmt.where(
                or_(
                    Tournament.join_token.is_(None),
                    Tournament.created_by == user.id,
                    exists().where(
                        (Participant.tournament_id == Tournament.id)
                        & (Participant.user_id == user.id)
                    ),
                )
            )

        result = await self._session.scalars(stmt)
        return list(result)

    async def get_tournament(
        self,
        tournament_id: int,
        user: User,
    ) -> Tournament:
        """Return a single tournament if visible to `user`.

        Uses the same visibility rules as list_tournaments. Invisible
        tournaments raise NotFoundError (enumeration guard).
        """
        tournament = await self._session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if user.is_admin:
            return tournament
        if tournament.join_token is None:
            return tournament
        if tournament.created_by == user.id:
            return tournament

        # Check participant rows
        exists_row = await self._session.scalar(
            select(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.user_id == user.id)
        )
        if exists_row is not None:
            return tournament
        raise NotFoundError(f"tournament {tournament_id}")

    async def get_history(
        self,
        tournament_id: int,
        user: User,
        last_n: int | None = None,
    ) -> list[Round]:
        """Return rounds for a tournament visible to `user`.

        Plan 2a ships PD-only, so all participants see all actions
        post-resolution. last_n is hard-capped at 100.
        """
        # Visibility check reuses get_tournament's rule
        await self.get_tournament(tournament_id, user)

        cap = min(last_n or 100, 100)
        stmt = (
            select(Round)
            .where(Round.tournament_id == tournament_id)
            .order_by(Round.round_number.desc())
            .limit(cap)
        )

        result = await self._session.scalars(stmt)
        rounds = list(result)
        rounds.reverse()  # ascending by round_number for consumer convenience
        return rounds

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
        now = _utc_now()
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

    async def force_resolve_round(self, round_id: int) -> None:
        """Force-resolve an expired round by creating TIMEOUT_DEFAULT actions.

        Called by the deadline worker when a round's deadline has passed and
        not all participants have submitted. Creates synthetic Action rows with
        source=TIMEOUT_DEFAULT for every participant that has not yet acted,
        marks the round COMPLETED, then either starts the next round or
        completes the tournament.

        Task 21 (integration test) validates the full race-guard path.
        """
        from atp.dashboard.tournament.models import Action, ActionSource, Participant

        # Load the round with its tournament
        row = await self._session.execute(
            select(Round)
            .where(Round.id == round_id)
            .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
        )
        round_obj = row.scalar_one_or_none()
        if round_obj is None:
            # Already resolved or cancelled — idempotent no-op
            return

        tournament = await self._session.get(Tournament, round_obj.tournament_id)
        if tournament is None:
            return

        # Find participant IDs that have NOT submitted
        submitted_result = await self._session.execute(
            select(Action.participant_id).where(Action.round_id == round_id)
        )
        submitted_ids = {row[0] for row in submitted_result}

        participants_result = await self._session.execute(
            select(Participant.id).where(
                Participant.tournament_id == round_obj.tournament_id
            )
        )
        all_participant_ids = [row[0] for row in participants_result]

        game = _game_for(tournament)
        now = _utc_now()
        for participant_id in all_participant_ids:
            if participant_id not in submitted_ids:
                self._session.add(
                    Action(
                        round_id=round_id,
                        participant_id=participant_id,
                        action_data=game.default_action_on_timeout(),
                        submitted_at=now,
                        source=ActionSource.TIMEOUT_DEFAULT,
                    )
                )

        # Reuse the normal resolution path so timeout-forced rounds compute
        # and persist payoffs identically to submit-driven resolution.
        await self._session.flush()
        await self._resolve_round(round_obj, tournament)

    # ------------------------------------------------------------------
    # Admin: kick participant
    # ------------------------------------------------------------------

    async def kick_participant(self, tournament_id: int, participant_id: int) -> None:
        """Release a participant from a live tournament.

        Sets ``Participant.released_at = now``. If there is no Action
        yet for the current in-progress round, inserts a
        ``TIMEOUT_DEFAULT`` action with the game's
        ``default_action_on_timeout()`` payload so round resolution is
        not blocked by the freed slot.

        Kicking is only allowed on *live* tournaments (PENDING or
        ACTIVE). COMPLETED and CANCELLED tournaments are frozen
        post-mortem state and mutating ``released_at`` there would
        corrupt the audit trail.

        Raises:
            LookupError: participant does not exist in this tournament.
            ValueError: participant is already released, OR the
                tournament is not in a live status (pending/active).
        """
        stmt = (
            select(Participant)
            .where(
                Participant.tournament_id == tournament_id,
                Participant.id == participant_id,
            )
            .options(selectinload(Participant.tournament))
        )
        participant = (await self._session.execute(stmt)).scalars().first()
        if participant is None:
            raise LookupError(
                f"participant {participant_id} not found in tournament {tournament_id}"
            )
        tournament = participant.tournament
        if tournament.status not in (
            TournamentStatus.PENDING.value,
            TournamentStatus.ACTIVE.value,
        ):
            raise ValueError(
                f"cannot kick from tournament in status {tournament.status!r}; "
                f"kicking is only allowed on live (pending/active) tournaments"
            )
        if participant.released_at is not None:
            raise ValueError("participant already released")

        participant.released_at = _utc_now()

        if tournament.status == TournamentStatus.ACTIVE.value:
            round_stmt = (
                select(Round)
                .where(
                    Round.tournament_id == tournament_id,
                    Round.status.in_(
                        (
                            RoundStatus.WAITING_FOR_ACTIONS.value,
                            RoundStatus.IN_PROGRESS.value,
                        )
                    ),
                )
                .order_by(Round.round_number.desc())
            )
            current_round = (await self._session.execute(round_stmt)).scalars().first()
            if current_round is not None:
                existing_stmt = select(Action).where(
                    Action.round_id == current_round.id,
                    Action.participant_id == participant.id,
                )
                existing = (
                    (await self._session.execute(existing_stmt)).scalars().first()
                )
                if existing is None:
                    game = _game_for(tournament)
                    default_action = game.default_action_on_timeout()
                    self._session.add(
                        Action(
                            round_id=current_round.id,
                            participant_id=participant.id,
                            action_data=default_action,
                            submitted_at=_utc_now(),
                            source=ActionSource.TIMEOUT_DEFAULT,
                        )
                    )

        await self._session.commit()

    # ------------------------------------------------------------------
    # Admin activity snapshot
    # ------------------------------------------------------------------

    async def get_admin_activity(self, tournament_id: int) -> dict:
        """Return an admin-level activity snapshot for HTMX rendering.

        Loads the tournament, its participants, and every Round + Action
        in a single eagerly-loaded query, then aggregates into the shape
        documented in the admin-GUI spec. Raises LookupError if the
        tournament does not exist.
        """
        stmt = (
            select(Tournament)
            .where(Tournament.id == tournament_id)
            .options(
                selectinload(Tournament.participants),
                selectinload(Tournament.rounds).selectinload(Round.actions),
            )
        )
        tournament = (await self._session.execute(stmt)).scalars().first()
        if tournament is None:
            raise LookupError(f"tournament {tournament_id} not found")

        total_rounds = tournament.total_rounds
        rounds_by_number: dict[int, Round] = {
            r.round_number: r for r in tournament.rounds
        }

        # Current round = highest round_number among rounds not yet
        # COMPLETED; else the highest round_number overall; else 0.
        active_rounds = [
            r
            for r in tournament.rounds
            if r.status != RoundStatus.COMPLETED.value
            and r.status != RoundStatus.CANCELLED.value
        ]
        if active_rounds:
            current_round_number = max(r.round_number for r in active_rounds)
        elif tournament.rounds:
            current_round_number = max(r.round_number for r in tournament.rounds)
        else:
            current_round_number = 0

        # Deadline countdown (live rounds only).
        deadline_remaining_s: int | None = None
        current_round_obj = rounds_by_number.get(current_round_number)
        if (
            current_round_obj is not None
            and current_round_obj.deadline is not None
            and current_round_obj.status
            in (
                RoundStatus.WAITING_FOR_ACTIONS.value,
                RoundStatus.IN_PROGRESS.value,
            )
        ):
            now = _utc_now()  # naive UTC per module convention
            deadline = current_round_obj.deadline
            if deadline.tzinfo is not None:
                deadline = deadline.astimezone(UTC).replace(tzinfo=None)
            delta = deadline - now
            deadline_remaining_s = max(0, int(delta.total_seconds()))

        # Action lookup: (participant_id, round_number) -> Action.
        actions_by_pid_round: dict[tuple[int, int], Action] = {}
        for rnd in tournament.rounds:
            for act in rnd.actions:
                actions_by_pid_round[(act.participant_id, rnd.round_number)] = act

        def cell_for(action: Action | None, round_status: str | None) -> str:
            if action is None:
                # Rounds that have already completed or been cancelled
                # without an action from this participant count as
                # timeout (force_resolve_round normally fills completed
                # rounds; cancelled rounds may simply stop with missing
                # actions, which we still surface as a miss).
                if round_status in (
                    RoundStatus.COMPLETED.value,
                    RoundStatus.CANCELLED.value,
                ):
                    return "timeout"
                return "waiting"
            if action.source == ActionSource.TIMEOUT_DEFAULT.value:
                return "timeout"
            return "submitted"

        participants_out: list[dict] = []
        submitted_this_round = 0
        for p in tournament.participants:
            row_per_round: list[str] = []
            for r_num in range(1, total_rounds + 1):
                rnd = rounds_by_number.get(r_num)
                act = actions_by_pid_round.get((p.id, r_num))
                row_per_round.append(
                    cell_for(act, rnd.status if rnd is not None else None)
                )

            current_act = actions_by_pid_round.get((p.id, current_round_number))
            if p.released_at is not None:
                status_str = "released"
            elif current_round_number == 0:
                status_str = "waiting"
            elif current_act is None:
                status_str = "waiting"
            elif current_act.source == ActionSource.TIMEOUT_DEFAULT.value:
                status_str = "timeout"
            else:
                status_str = "submitted"
                submitted_this_round += 1

            participants_out.append(
                {
                    "id": p.id,
                    "agent_name": p.agent_name,
                    "released_at": p.released_at,
                    "total_score": p.total_score,
                    "current_round_status": status_str,
                    "current_round_submitted_at": (
                        current_act.submitted_at if current_act else None
                    ),
                    "row_per_round": row_per_round,
                }
            )

        return {
            "tournament_id": tournament.id,
            "status": tournament.status,
            "current_round": current_round_number or None,
            "total_rounds": total_rounds,
            "deadline_remaining_s": deadline_remaining_s,
            "participants": participants_out,
            "submitted_this_round": submitted_this_round,
            "total_this_round": len(tournament.participants),
        }
