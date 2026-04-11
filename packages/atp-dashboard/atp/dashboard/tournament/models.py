"""SQLAlchemy models for the ATP Tournament platform."""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atp.dashboard.models import DEFAULT_TENANT_ID, Base


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
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
    )
    agent_name: Mapped[str] = mapped_column(String(200), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)

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

    __table_args__ = (Index("idx_round_tournament", "tournament_id"),)

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

    # Relationships
    round: Mapped["Round"] = relationship(
        "Round",
        back_populates="actions",
    )
    participant: Mapped["Participant"] = relationship(
        "Participant",
        back_populates="actions",
    )

    __table_args__ = (Index("idx_action_round", "round_id"),)

    def __repr__(self) -> str:
        return (
            f"Action(id={self.id}, round_id={self.round_id}, "
            f"participant_id={self.participant_id})"
        )
