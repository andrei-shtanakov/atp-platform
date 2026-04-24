"""SQLAlchemy models for the ATP Tournament platform."""

from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import DEFAULT_TENANT_ID, Base
from atp.dashboard.tournament.reasons import CancelReason


class TournamentStatus(StrEnum):
    """Status of a tournament."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RoundStatus(StrEnum):
    """Round lifecycle status.

    WAITING_FOR_ACTIONS, IN_PROGRESS, COMPLETED existed as bare string
    literals in vertical slice service.py. Plan 2a introduces this StrEnum
    for type safety and adds CANCELLED as a new value used by _cancel_impl
    to transition in-flight rounds when their tournament is cancelled.

    Stored as plain String(20) in the DB without a native enum type or
    CHECK constraint.
    """

    WAITING_FOR_ACTIONS = "waiting_for_actions"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ActionSource(StrEnum):
    """Origin of an Action row.

    SUBMITTED — player sent make_move via MCP tool before deadline.
    TIMEOUT_DEFAULT — deadline worker force_resolve_round created a
    default action for a participant who did not submit before the
    round deadline.
    BUILTIN — LABS-TSA PR-4: tournament runner synthesised this
    action by calling a builtin strategy's ``choose_action`` during
    round resolution. Distinct from SUBMITTED so UI/admin reporting
    can tell real player moves from sparring-partner moves.

    Stored as plain String(32) without a native enum type or CHECK
    constraint.
    """

    SUBMITTED = "submitted"
    TIMEOUT_DEFAULT = "timeout_default"
    BUILTIN = "builtin"


class Tournament(Base):
    """A tournament definition for game-theoretic evaluation."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=DEFAULT_TENANT_ID,
        index=True,
    )
    game_type: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TournamentStatus.PENDING,
    )
    starts_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rules: Mapped[dict] = mapped_column(JSON, default=dict)
    num_players: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="2"
    )
    total_rounds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    round_deadline_s: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="30"
    )
    created_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Plan 2a additive columns — AD-9 pending deadline
    # server_default=now() so create_all()-built test DBs accept inserts
    # that omit the column. Production code in TournamentService always
    # sets it explicitly to creator-controlled deadline, so the default
    # only matters for tests and for migrations that backfill stale rows.
    pending_deadline: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Plan 2a additive columns — AD-10 join token
    join_token: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Plan 2a additive columns — cancel audit
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancelled_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    cancelled_reason: Mapped[CancelReason | None] = mapped_column(
        sa.Enum(
            CancelReason,
            native_enum=False,
            length=32,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=True,
    )
    cancelled_reason_detail: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )

    # Relationships
    participants: Mapped[list["Participant"]] = relationship(
        "Participant",
        back_populates="tournament",
    )
    rounds: Mapped[list["Round"]] = relationship(
        "Round",
        back_populates="tournament",
    )

    __table_args__ = (
        Index("idx_tournaments_status", "status"),
        Index("idx_tournaments_tenant", "tenant_id"),
        Index(
            "idx_tournaments_status_pending_deadline",
            "status",
            "pending_deadline",
        ),
        # Cancel audit invariant: either all cancelled_* fields are NULL
        # (not cancelled), or admin_action has cancelled_by set, or system
        # reasons (pending_timeout/abandoned) have cancelled_by NULL but
        # cancelled_at set. This mirrors the DB CHECK constraint added by
        # the Plan 2a Alembic migration and is duplicated here so that
        # `Base.metadata.create_all()` produces a schema equivalent to
        # `alembic upgrade head`. Also enforced in application code by
        # TournamentCancelEvent.__post_init__ (defence in depth).
        CheckConstraint(
            "("
            "  (cancelled_reason IS NULL"
            "   AND cancelled_by IS NULL"
            "   AND cancelled_at IS NULL)"
            "  OR (cancelled_reason = 'admin_action'"
            "      AND cancelled_by IS NOT NULL"
            "      AND cancelled_at IS NOT NULL)"
            "  OR (cancelled_reason IN ('pending_timeout', 'abandoned')"
            "      AND cancelled_by IS NULL"
            "      AND cancelled_at IS NOT NULL)"
            ")",
            name="ck_tournament_cancel_consistency",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"Tournament(id={self.id}, game_type={self.game_type!r}, "
            f"status={self.status!r})"
        )


class Participant(Base):
    """A participant in a tournament."""

    __tablename__ = "tournament_participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournaments.id"),
        nullable=False,
    )
    # LABS-TSA PR-1: nullable to allow builtin-strategy participants
    # (which have no User). Enforced together with agent_id / builtin_strategy
    # via a CHECK constraint in __table_args__ below.
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    agent_name: Mapped[str] = mapped_column(String(200), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Plan 2a additive column — AD-10 slot release
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Scope #2 — link to Agent record (nullable for old participants)
    agent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=True
    )

    # LABS-TSA PR-1: builtin strategy name (namespaced as "{game}/{strategy}").
    # Exactly one of agent_id / builtin_strategy must be set; the CHECK
    # constraint in __table_args__ enforces this.
    builtin_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    tournament: Mapped["Tournament"] = relationship(
        "Tournament",
        back_populates="participants",
    )
    actions: Mapped[list["Action"]] = relationship(
        "Action",
        back_populates="participant",
    )

    __table_args__ = (
        Index("idx_participant_tournament", "tournament_id"),
        Index("idx_participant_user", "user_id"),
        # LABS-TSA PR-6: uniqueness is agent-keyed, not user-keyed. A
        # single user may register multiple tournament agents and play
        # them all in the same tournament. Builtins (agent_id IS NULL)
        # remain unconstrained so a tournament can seat several of the
        # same builtin strategy.
        Index(
            # uq_ prefix: semantically a unique constraint, implemented
            # as a partial unique index because UniqueConstraint does
            # not accept WHERE clauses and neither SQLite nor PostgreSQL
            # support partial UNIQUE in CREATE TABLE syntax.
            "uq_participant_tournament_agent",
            "tournament_id",
            "agent_id",
            unique=True,
            sqlite_where=text("agent_id IS NOT NULL"),
            postgresql_where=text("agent_id IS NOT NULL"),
        ),
        Index(
            "uq_participant_agent_active",
            "agent_id",
            unique=True,
            sqlite_where=text("agent_id IS NOT NULL AND released_at IS NULL"),
            postgresql_where=text("agent_id IS NOT NULL AND released_at IS NULL"),
        ),
        # LABS-TSA PR-1
        Index(
            "idx_participants_builtin",
            "tournament_id",
            "builtin_strategy",
            sqlite_where=text("builtin_strategy IS NOT NULL"),
            postgresql_where=text("builtin_strategy IS NOT NULL"),
        ),
        # LABS-TSA PR-4: agent-xor-builtin CHECK. Every Participant row
        # is either a real agent-backed entry (agent_id set, builtin
        # null) or a synthetic builtin (builtin_strategy set, agent_id
        # null) — never both, never neither. Duplicated here so
        # Base.metadata.create_all() produces a schema equivalent to
        # ``alembic upgrade head``. See the PR-4 migration
        # ``a9c4e81f3d2a_tournament_participant_agent_xor_builtin_check``.
        CheckConstraint(
            "(agent_id IS NOT NULL AND builtin_strategy IS NULL)"
            " OR (agent_id IS NULL AND builtin_strategy IS NOT NULL)",
            name="ck_participants_agent_xor_builtin",
        ),
    )

    def __repr__(self) -> str:
        return f"Participant(id={self.id}, agent_name={self.agent_name!r})"


class Round(Base):
    """A round within a tournament."""

    __tablename__ = "tournament_rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournaments.id"),
        nullable=False,
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    deadline: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    tournament: Mapped["Tournament"] = relationship(
        "Tournament",
        back_populates="rounds",
    )
    actions: Mapped[list["Action"]] = relationship(
        "Action",
        back_populates="round",
    )

    __table_args__ = (
        Index("idx_round_tournament", "tournament_id"),
        UniqueConstraint(
            "tournament_id",
            "round_number",
            name="uq_round_tournament_number",
        ),
        Index("idx_round_status_deadline", "status", "deadline"),
    )

    def __repr__(self) -> str:
        return (
            f"Round(id={self.id}, round_number={self.round_number}, "
            f"status={self.status!r})"
        )


class Action(Base):
    """An action submitted by a participant in a round."""

    __tablename__ = "tournament_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournament_rounds.id"),
        nullable=False,
    )
    participant_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tournament_participants.id"),
        nullable=False,
    )
    action_data: Mapped[dict] = mapped_column(JSON, default=dict)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    payoff: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Plan 2a additive column — audit trail for timeout-default vs
    # player-submitted actions
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="submitted",
    )

    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    round: Mapped["Round"] = relationship(
        "Round",
        back_populates="actions",
    )
    participant: Mapped["Participant"] = relationship(
        "Participant",
        back_populates="actions",
    )

    __table_args__ = (
        Index("idx_action_round", "round_id"),
        UniqueConstraint(
            "round_id",
            "participant_id",
            name="uq_action_round_participant",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"Action(id={self.id}, round_id={self.round_id}, "
            f"participant_id={self.participant_id})"
        )
