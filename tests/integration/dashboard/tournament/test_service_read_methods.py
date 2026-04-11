"""Tests for list_tournaments, get_tournament, get_history visibility."""

import pytest
from sqlalchemy import text

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.service import TournamentService


class _DummyBus:
    async def publish(self, event):
        pass


async def _seed(session):
    for uid, uname, is_admin in [(1, "alice", 0), (2, "bob", 0), (99, "admin", 1)]:
        await session.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (:id, 'default', :u, :e, 'x', 1, :admin, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": uid, "u": uname, "e": f"{uname}@test.com", "admin": is_admin},
        )
    # Three tournaments: public (owned by 1), private (owned by 2),
    # public (owned by 99/admin).
    await session.execute(
        text(
            "INSERT INTO tournaments "
            "(id, tenant_id, game_type, config, rules, status, num_players, "
            "total_rounds, round_deadline_s, pending_deadline, created_by, "
            "join_token, created_at) "
            "VALUES "
            "(1, 'default', 'prisoners_dilemma', '{\"name\": \"public_alice\"}', "
            "'{}', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1, NULL, "
            "CURRENT_TIMESTAMP),"
            "(2, 'default', 'prisoners_dilemma', '{\"name\": \"private_bob\"}', "
            "'{}', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 2, 'secret', "
            "CURRENT_TIMESTAMP),"
            "(3, 'default', 'prisoners_dilemma', '{\"name\": \"public_admin\"}', "
            "'{}', 'active', 2, 3, 30, CURRENT_TIMESTAMP, 99, NULL, "
            "CURRENT_TIMESTAMP)"
        )
    )


async def _load_user(session, user_id):
    from atp.dashboard.models import User

    return await session.get(User, user_id)


def _get_name(tournament) -> str:
    """Extract name from tournament config JSON."""
    return tournament.config.get("name", "")


@pytest.mark.anyio
async def test_list_tournaments_admin_sees_all(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        admin = await _load_user(s, 99)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=admin)
        assert len(result) == 3


@pytest.mark.anyio
async def test_list_tournaments_non_admin_hides_private(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=alice)
        names = {_get_name(t) for t in result}
        assert "public_alice" in names
        assert "public_admin" in names
        assert "private_bob" not in names  # not visible to alice


@pytest.mark.anyio
async def test_list_tournaments_shows_own_private(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        bob = await _load_user(s, 2)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=bob)
        names = {_get_name(t) for t in result}
        assert "private_bob" in names  # bob created it


@pytest.mark.anyio
async def test_get_tournament_invisible_raises_not_found(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        with pytest.raises(NotFoundError):
            await svc.get_tournament(tournament_id=2, user=alice)


@pytest.mark.anyio
async def test_get_tournament_admin_sees_private(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        admin = await _load_user(s, 99)
        svc = TournamentService(s, _DummyBus())
        t = await svc.get_tournament(tournament_id=2, user=admin)
        assert t.id == 2


@pytest.mark.anyio
async def test_list_tournaments_status_filter(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        admin = await _load_user(s, 99)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=admin, status=TournamentStatus.ACTIVE)
        assert len(result) == 1
        assert result[0].status == TournamentStatus.ACTIVE
