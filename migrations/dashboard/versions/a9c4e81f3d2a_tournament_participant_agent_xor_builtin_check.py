"""Add Participant agent-xor-builtin CHECK (LABS-TSA PR-4).

Deferred from PR-1 — PR-4 updates ``TournamentService.join`` to
populate ``agent_id`` on every real-agent Participant insert
(auto-provisioning an Agent row if the caller does not already
own one), so the XOR invariant is now safe to enforce at the DB
level.

Invariant:
    (agent_id IS NOT NULL AND builtin_strategy IS NULL)
    OR (agent_id IS NULL AND builtin_strategy IS NOT NULL)

Revision ID: a9c4e81f3d2a
Revises: 52987a83afb7
Create Date: 2026-04-23 21:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a9c4e81f3d2a"
down_revision: str | Sequence[str] | None = "52987a83afb7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ck_participants_agent_xor_builtin CHECK on
    ``tournament_participants``.
    """
    with op.batch_alter_table("tournament_participants", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "ck_participants_agent_xor_builtin",
            "(agent_id IS NOT NULL AND builtin_strategy IS NULL)"
            " OR (agent_id IS NULL AND builtin_strategy IS NOT NULL)",
        )


def downgrade() -> None:
    """Drop the agent-xor-builtin CHECK."""
    with op.batch_alter_table("tournament_participants", schema=None) as batch_op:
        batch_op.drop_constraint("ck_participants_agent_xor_builtin", type_="check")
