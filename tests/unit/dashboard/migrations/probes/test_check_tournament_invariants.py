"""Tests for the Plan 2a pre-migration probe module."""

import pytest
from sqlalchemy import create_engine, text

from atp.dashboard.migrations.probes.check_tournament_invariants import (
    check_tournament_schema_ready,
)


@pytest.fixture
def baseline_db(tmp_path):
    """SQLite DB matching the vertical-slice schema (pre-Plan-2a).

    Only creates the tables the probes touch: users, tournaments,
    tournament_participants, tournament_rounds, tournament_actions.
    FK enforcement OFF by default so tests can seed orphan rows for P2.
    """
    db_path = tmp_path / "probe_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                is_admin BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE tournaments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                game_type TEXT NOT NULL DEFAULT 'prisoners_dilemma',
                config TEXT DEFAULT '{}'
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE tournament_participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER NOT NULL,
                user_id INTEGER,
                agent_name TEXT NOT NULL DEFAULT ''
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE tournament_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER NOT NULL,
                round_number INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'waiting_for_actions'
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE tournament_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                participant_id INTEGER NOT NULL,
                action_data TEXT DEFAULT '{}'
            )
        """)
        )
        conn.execute(
            text("INSERT INTO users (id, username) VALUES (1, 'alice'), (2, 'bob')")
        )
    yield engine
    engine.dispose()


def test_clean_db_returns_empty(baseline_db):
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert violations == []


def test_p1_detects_null_user_id(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text("INSERT INTO tournaments (id, name) VALUES (1, 't')"))
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES (1, NULL, 'anon')"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P1:") for v in violations)


def test_p2_detects_fk_orphan(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text("INSERT INTO tournaments (id, name) VALUES (1, 't')"))
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES (1, 99999, 'ghost')"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P2:") for v in violations)


def test_p3_detects_duplicate_participant(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text("INSERT INTO tournaments (id, name) VALUES (1, 't')"))
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES (1, 1, 'a')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES (1, 1, 'b')"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P3:") for v in violations)


def test_p4_detects_duplicate_action(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text("INSERT INTO tournaments (id, name) VALUES (1, 't')"))
        conn.execute(
            text(
                "INSERT INTO tournament_rounds "
                "(id, tournament_id, round_number) VALUES (1, 1, 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(id, tournament_id, user_id, agent_name) "
                "VALUES (1, 1, 1, 'a')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO tournament_actions "
                "(round_id, participant_id) VALUES (1, 1), (1, 1)"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P4:") for v in violations)


def test_p5_detects_duplicate_round(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text("INSERT INTO tournaments (id, name) VALUES (1, 't')"))
        conn.execute(
            text(
                "INSERT INTO tournament_rounds "
                "(tournament_id, round_number) VALUES (1, 1), (1, 1)"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P5:") for v in violations)


def test_p6_detects_multi_active_user(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tournaments (id, name, status) VALUES "
                "(1, 't1', 'active'), (2, 't2', 'pending')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES "
                "(1, 1, 'a'), (2, 1, 'b')"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P6:") for v in violations)


def test_p6_ignores_completed_tournament_history(baseline_db):
    """Critical relaxed-probe edge case: a user with past participation
    in completed tournaments plus one active participation must pass."""
    with baseline_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tournaments (id, name, status) VALUES "
                "(1, 't1', 'completed'), "
                "(2, 't2', 'completed'), "
                "(3, 't3', 'completed'), "
                "(4, 't4', 'active')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES "
                "(1, 1, 'a'), (2, 1, 'b'), (3, 1, 'c'), (4, 1, 'd')"
            )
        )
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    # No P6 violation — historical completed participations are ignored
    assert not any(v.startswith("P6:") for v in violations)
