"""Tests for tournament SQLAlchemy models."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atp.dashboard.models import Base, User
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    Tournament,
    TournamentStatus,
)


def _make_user(session: Session, username: str = "u1") -> User:
    """Helper: create a User row so Participant.user_id (NOT NULL since
    Plan 2a IDOR fix) can be satisfied."""
    user = User(
        username=username,
        email=f"{username}@test.com",
        hashed_password="x",
        is_active=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    """Create a new database session for testing."""
    with Session(engine) as sess:
        yield sess


class TestTournamentStatus:
    """Tests for TournamentStatus enum."""

    def test_all_statuses_exist(self) -> None:
        assert TournamentStatus.PENDING == "pending"
        assert TournamentStatus.ACTIVE == "active"
        assert TournamentStatus.COMPLETED == "completed"
        assert TournamentStatus.CANCELLED == "cancelled"

    def test_status_is_str(self) -> None:
        assert isinstance(TournamentStatus.PENDING, str)


class TestTournament:
    """Tests for the Tournament model."""

    def test_create_tournament_defaults(self, session: Session) -> None:
        t = Tournament(game_type="prisoners_dilemma")
        session.add(t)
        session.commit()
        session.refresh(t)

        assert t.id is not None
        assert t.tenant_id == "default"
        assert t.game_type == "prisoners_dilemma"
        assert t.config == {}
        assert t.status == TournamentStatus.PENDING
        assert t.starts_at is None
        assert t.ends_at is None
        assert t.rules == {}
        assert t.created_by is None
        assert isinstance(t.created_at, datetime)

    def test_create_tournament_all_fields(self, session: Session) -> None:
        now = datetime.now()
        t = Tournament(
            tenant_id="acme",
            game_type="el_farol_bar",
            config={"threshold": 60},
            status=TournamentStatus.ACTIVE,
            starts_at=now,
            ends_at=now,
            rules={"max_players": 10},
            created_by=None,
        )
        session.add(t)
        session.commit()
        session.refresh(t)

        assert t.tenant_id == "acme"
        assert t.game_type == "el_farol_bar"
        assert t.config == {"threshold": 60}
        assert t.status == TournamentStatus.ACTIVE
        assert t.starts_at == now
        assert t.ends_at == now
        assert t.rules == {"max_players": 10}

    def test_tournament_table_name(self) -> None:
        assert Tournament.__tablename__ == "tournaments"


class TestParticipant:
    """Tests for the Participant model."""

    def test_participant_joins_tournament(self, session: Session) -> None:
        t = Tournament(game_type="prisoners_dilemma")
        session.add(t)
        session.commit()
        session.refresh(t)

        # Plan 2a IDOR fix made Participant.user_id NOT NULL.
        u = _make_user(session, "alice")

        p = Participant(
            tournament_id=t.id,
            user_id=u.id,
            agent_name="tit-for-tat",
        )
        session.add(p)
        session.commit()
        session.refresh(p)

        assert p.id is not None
        assert p.tournament_id == t.id
        assert p.agent_name == "tit-for-tat"
        assert p.user_id == u.id
        assert p.total_score is None
        assert isinstance(p.joined_at, datetime)

    def test_participant_with_score(self, session: Session) -> None:
        t = Tournament(game_type="hawks_doves")
        session.add(t)
        session.commit()
        session.refresh(t)

        u = _make_user(session, "bob")

        p = Participant(
            tournament_id=t.id,
            user_id=u.id,
            agent_name="always-cooperate",
            total_score=42.5,
        )
        session.add(p)
        session.commit()
        session.refresh(p)

        assert p.total_score == pytest.approx(42.5)

    def test_participant_table_name(self) -> None:
        assert Participant.__tablename__ == "tournament_participants"


class TestRound:
    """Tests for the Round model."""

    def test_create_round(self, session: Session) -> None:
        t = Tournament(game_type="prisoners_dilemma")
        session.add(t)
        session.commit()
        session.refresh(t)

        r = Round(
            tournament_id=t.id,
            round_number=1,
            state={"turn": 1, "history": []},
        )
        session.add(r)
        session.commit()
        session.refresh(r)

        assert r.id is not None
        assert r.tournament_id == t.id
        assert r.round_number == 1
        assert r.state == {"turn": 1, "history": []}
        assert r.status == "pending"
        assert r.deadline is None
        assert isinstance(r.started_at, datetime)

    def test_round_table_name(self) -> None:
        assert Round.__tablename__ == "tournament_rounds"


class TestAction:
    """Tests for the Action model."""

    def test_create_action(self, session: Session) -> None:
        t = Tournament(game_type="prisoners_dilemma")
        session.add(t)
        session.commit()
        session.refresh(t)

        u = _make_user(session, "carol")
        p = Participant(tournament_id=t.id, user_id=u.id, agent_name="test-agent")
        session.add(p)
        session.commit()
        session.refresh(p)

        r = Round(
            tournament_id=t.id,
            round_number=1,
        )
        session.add(r)
        session.commit()
        session.refresh(r)

        a = Action(
            round_id=r.id,
            participant_id=p.id,
            action_data={"choice": "cooperate"},
        )
        session.add(a)
        session.commit()
        session.refresh(a)

        assert a.id is not None
        assert a.round_id == r.id
        assert a.participant_id == p.id
        assert a.action_data == {"choice": "cooperate"}
        assert isinstance(a.submitted_at, datetime)

    def test_action_table_name(self) -> None:
        assert Action.__tablename__ == "tournament_actions"
