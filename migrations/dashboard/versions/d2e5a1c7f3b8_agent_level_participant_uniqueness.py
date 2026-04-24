"""Shift Participant uniqueness from user_id to agent_id (LABS-TSA PR-6).

LABS-TSA lets a single user register up to 5 tournament agents and play
them all in the same tournament. The legacy "one user = one player"
uniqueness on ``tournament_participants`` has to shift to agent-level
uniqueness to unblock that scenario.

Changes:
- Drop ``uq_participant_tournament_user`` UniqueConstraint on
  ``(tournament_id, user_id)``.
- Drop ``uq_participant_user_active`` partial unique index on
  ``user_id WHERE user_id IS NOT NULL AND released_at IS NULL``.
- Add ``uq_participant_tournament_agent`` partial unique index on
  ``(tournament_id, agent_id) WHERE agent_id IS NOT NULL``. Built-in
  participants (``agent_id IS NULL``) stay unconstrained.
- Add ``uq_participant_agent_active`` partial unique index on
  ``agent_id WHERE agent_id IS NOT NULL AND released_at IS NULL``.
  Same builtin exemption applies.

Revision ID: d2e5a1c7f3b8
Revises: a9c4e81f3d2a
Create Date: 2026-04-23 22:00:00.000000
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "d2e5a1c7f3b8"
down_revision: str | Sequence[str] | None = "a9c4e81f3d2a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop user-keyed uniqueness and add agent-keyed uniqueness."""
    # Drop the old partial unique index first (created outside batch in
    # its original migration, so dropped the same way here).
    op.drop_index(
        "uq_participant_user_active",
        table_name="tournament_participants",
    )

    # Drop the composite UniqueConstraint via batch (SQLite needs the
    # table rewrite path).
    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.drop_constraint("uq_participant_tournament_user", type_="unique")

    # Add new partial unique indexes. Both are created outside
    # batch_alter_table because op.create_index accepts the
    # sqlite_where/postgresql_where kwargs.
    op.create_index(
        "uq_participant_tournament_agent",
        "tournament_participants",
        ["tournament_id", "agent_id"],
        unique=True,
        sqlite_where=text("agent_id IS NOT NULL"),
        postgresql_where=text("agent_id IS NOT NULL"),
    )
    op.create_index(
        "uq_participant_agent_active",
        "tournament_participants",
        ["agent_id"],
        unique=True,
        sqlite_where=text("agent_id IS NOT NULL AND released_at IS NULL"),
        postgresql_where=text("agent_id IS NOT NULL AND released_at IS NULL"),
    )


def downgrade() -> None:
    """Restore user-keyed uniqueness; drop agent-keyed uniqueness."""
    op.drop_index(
        "uq_participant_agent_active",
        table_name="tournament_participants",
    )
    op.drop_index(
        "uq_participant_tournament_agent",
        table_name="tournament_participants",
    )

    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.create_unique_constraint(
            "uq_participant_tournament_user", ["tournament_id", "user_id"]
        )

    op.create_index(
        "uq_participant_user_active",
        "tournament_participants",
        ["user_id"],
        unique=True,
        sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
    )
