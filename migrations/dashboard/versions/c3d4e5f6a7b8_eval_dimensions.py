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


def _columns(insp: sa.Inspector, table: str) -> set[str]:
    return {c["name"] for c in insp.get_columns(table)}


def _indexes(insp: sa.Inspector, table: str) -> set[str]:
    return {i["name"] for i in insp.get_indexes(table)}


def upgrade() -> None:
    # Idempotent: `_add_missing_columns()` runs at app startup and may already
    # have added these columns on a live SQLite (boot-before-migrate). Skip any
    # that already exist so the revision still applies + stamps cleanly.
    insp = sa.inspect(op.get_bind())
    have = _columns(insp, "test_executions")
    for name, type_ in _TEST_COLS:
        if name not in have:
            op.add_column("test_executions", sa.Column(name, type_, nullable=True))
    have = _columns(insp, "suite_executions")
    for name, type_ in _SUITE_COLS:
        if name not in have:
            op.add_column("suite_executions", sa.Column(name, type_, nullable=True))
    # Name matches SQLAlchemy's auto-index from `index=True` on the model column,
    # so the migration path and the create_all path leave the same index name.
    if "ix_suite_executions_run_uuid" not in _indexes(insp, "suite_executions"):
        op.create_index(
            "ix_suite_executions_run_uuid",
            "suite_executions",
            ["run_uuid"],
            unique=False,
        )


def downgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if "ix_suite_executions_run_uuid" in _indexes(insp, "suite_executions"):
        op.drop_index("ix_suite_executions_run_uuid", table_name="suite_executions")
    have = _columns(insp, "suite_executions")
    for name, _ in _SUITE_COLS:
        if name in have:
            op.drop_column("suite_executions", name)
    have = _columns(insp, "test_executions")
    for name, _ in _TEST_COLS:
        if name in have:
            op.drop_column("test_executions", name)
