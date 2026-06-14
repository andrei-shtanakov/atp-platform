"""SP-1 eval dimensions + run aggregates

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TEST_COLS = [
    ("axis_level", sa.String(length=50)),
    ("capability", sa.String(length=50)),
    ("family", sa.String(length=120)),
    ("case_version", sa.Integer()),
    ("critical_pass", sa.Boolean()),
    ("malformed", sa.Boolean()),
    ("recall", sa.Float()),
    ("precision", sa.Float()),
    ("fp_count", sa.Integer()),
    ("rubric_score", sa.Float()),
    ("grader_version", sa.String(length=80)),
]
_SUITE_COLS = [
    ("task_type", sa.String(length=50)),
    ("run_uuid", sa.String(length=36)),
    ("critical_pass_rate", sa.Float()),
    ("malformed_rate", sa.Float()),
    ("mean_rubric", sa.Float()),
    ("breakpoint_axis_level", sa.String(length=50)),
]


def upgrade() -> None:
    for name, type_ in _TEST_COLS:
        op.add_column("test_executions", sa.Column(name, type_, nullable=True))
    for name, type_ in _SUITE_COLS:
        op.add_column("suite_executions", sa.Column(name, type_, nullable=True))
    # Name matches SQLAlchemy's auto-index from `index=True` on the model column,
    # so the migration path and the create_all path leave the same index name.
    op.create_index(
        "ix_suite_executions_run_uuid",
        "suite_executions",
        ["run_uuid"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_suite_executions_run_uuid", table_name="suite_executions")
    for name, _ in _SUITE_COLS:
        op.drop_column("suite_executions", name)
    for name, _ in _TEST_COLS:
        op.drop_column("test_executions", name)
