"""SQLAlchemy models for the ATP Tournament platform."""

from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
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

    Stored as plain String(32) without a native enum type or CHECK
    constraint.
    """

    SUBMITTED = "submitted"
    TIMEOUT_DEFAULT = "timeout_default"


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
    pending_deadline: Mapped[datetime] = mapped_column(DateTime, nullable=False)

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
        sa.Enum(CancelReason, native_enum=False, length=32),
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
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(200), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Plan 2a additive column — AD-10 slot release
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

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
        UniqueConstraint(
            "tournament_id",
            "user_id",
            name="uq_participant_tournament_user",
        ),
        Index(
            # uq_ prefix: semantically a unique constraint, implemented
            # as a partial unique index because UniqueConstraint does
            # not accept WHERE clauses and neither SQLite nor PostgreSQL
            # support partial UNIQUE in CREATE TABLE syntax.
            "uq_participant_user_active",
            "user_id",
            unique=True,
            sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
            postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
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
