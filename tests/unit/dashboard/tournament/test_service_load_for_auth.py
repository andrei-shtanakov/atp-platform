"""Tests for TournamentService._load_for_auth enumeration guard."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.models import Tournament
from atp.dashboard.tournament.service import TournamentService


def _make_tournament(tournament_id: int, created_by: int | None) -> Tournament:
    t = MagicMock(spec=Tournament)
    t.id = tournament_id
    t.created_by = created_by
    return t


def _make_user(user_id: int, is_admin: bool = False) -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _make_service(tournament_result):
    session = MagicMock()
    session.get = AsyncMock(return_value=tournament_result)
    bus = MagicMock()
    return TournamentService(session=session, bus=bus)


@pytest.mark.anyio
async def test_admin_always_authorized():
    svc = _make_service(_make_tournament(1, created_by=99))
    admin = _make_user(42, is_admin=True)
    result = await svc._load_for_auth(tournament_id=1, user=admin)
    assert result.id == 1


@pytest.mark.anyio
async def test_admin_authorized_on_legacy_null_owner():
    svc = _make_service(_make_tournament(1, created_by=None))
    admin = _make_user(42, is_admin=True)
    result = await svc._load_for_auth(tournament_id=1, user=admin)
    assert result.id == 1


@pytest.mark.anyio
async def test_owner_authorized_on_match():
    svc = _make_service(_make_tournament(1, created_by=42))
    user = _make_user(42, is_admin=False)
    result = await svc._load_for_auth(tournament_id=1, user=user)
    assert result.id == 1


@pytest.mark.anyio
async def test_non_owner_raises_not_found():
    svc = _make_service(_make_tournament(1, created_by=99))
    user = _make_user(42, is_admin=False)
    with pytest.raises(NotFoundError):
        await svc._load_for_auth(tournament_id=1, user=user)


@pytest.mark.anyio
async def test_non_admin_legacy_null_raises_not_found():
    svc = _make_service(_make_tournament(1, created_by=None))
    user = _make_user(42, is_admin=False)
    with pytest.raises(NotFoundError):
        await svc._load_for_auth(tournament_id=1, user=user)


@pytest.mark.anyio
async def test_missing_tournament_raises_not_found():
    svc = _make_service(None)
    user = _make_user(42, is_admin=False)
    with pytest.raises(NotFoundError):
        await svc._load_for_auth(tournament_id=99999, user=user)
