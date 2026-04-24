"""Tests for TournamentService.create_tournament AD-9 validation,
pending_deadline computation, and join_token generation."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.errors import ValidationError
from atp.dashboard.tournament.service import (
    TOURNAMENT_PENDING_MAX_WAIT_S,
    TournamentService,
)


def _make_user(user_id=1, is_admin=False):
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _make_service(*, has_tournament_agent: bool = True):
    """Build a stubbed service.

    LABS-TSA PR-4: ``create_tournament`` now consults the DB to
    enforce the creator-commit invariant on private tournaments.
    The unit test stub returns ``has_tournament_agent`` as a positive
    scalar count so the default flow succeeds; pass
    ``has_tournament_agent=False`` to exercise the rejection branch.
    """
    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    # create_tournament awaits session.scalar(...) for the
    # creator-commit + concurrent-cap checks. Return an int so
    # len()/boolean comparisons work.
    session.scalar = AsyncMock(return_value=1 if has_tournament_agent else 0)
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    bus = MagicMock()
    return TournamentService(session=session, bus=bus)


@pytest.mark.anyio
async def test_create_within_cap_succeeds(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, token = await svc.create_tournament(
        creator=_make_user(),
        name="ok",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=90,
        round_deadline_s=30,
    )
    assert t is not None
    assert token is None  # public tournament


@pytest.mark.anyio
async def test_create_over_cap_raises_validation_error(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="max duration"):
        await svc.create_tournament(
            creator=_make_user(),
            name="too_long",
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=200,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_private_returns_join_token(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, token = await svc.create_tournament(
        creator=_make_user(),
        name="private",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
        private=True,
    )
    assert token is not None
    assert len(token) >= 32  # secrets.token_urlsafe(32) base64-ish


@pytest.mark.anyio
async def test_create_sets_pending_deadline(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, _ = await svc.create_tournament(
        creator=_make_user(),
        name="x",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    assert t.pending_deadline is not None
    # Should be roughly now + TOURNAMENT_PENDING_MAX_WAIT_S
    delta = t.pending_deadline - datetime.utcnow()
    assert timedelta(seconds=TOURNAMENT_PENDING_MAX_WAIT_S - 5) < delta
    assert delta < timedelta(seconds=TOURNAMENT_PENDING_MAX_WAIT_S + 5)


@pytest.mark.anyio
async def test_create_el_farol_n5_ok(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, _ = await svc.create_tournament(
        creator=_make_user(),
        name="smoke",
        game_type="el_farol",
        num_players=5,
        total_rounds=3,
        round_deadline_s=30,
    )
    assert t.game_type == "el_farol"
    assert t.num_players == 5


@pytest.mark.anyio
async def test_create_el_farol_n1_rejected(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="2 <= num_players <= 20"):
        await svc.create_tournament(
            creator=_make_user(),
            name="nope",
            game_type="el_farol",
            num_players=1,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_el_farol_n21_rejected(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="2 <= num_players <= 20"):
        await svc.create_tournament(
            creator=_make_user(),
            name="nope",
            game_type="el_farol",
            num_players=21,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_pd_still_requires_exactly_two(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="exactly 2"):
        await svc.create_tournament(
            creator=_make_user(),
            name="nope",
            game_type="prisoners_dilemma",
            num_players=3,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_public_goods_n4_ok(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, _ = await svc.create_tournament(
        creator=_make_user(),
        name="pg-smoke",
        game_type="public_goods",
        num_players=4,
        total_rounds=3,
        round_deadline_s=30,
    )
    assert t.game_type == "public_goods"
    assert t.num_players == 4


@pytest.mark.anyio
async def test_create_public_goods_n1_rejected(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="2 <= num_players <= 20"):
        await svc.create_tournament(
            creator=_make_user(),
            name="nope",
            game_type="public_goods",
            num_players=1,
            total_rounds=3,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_public_goods_n21_rejected(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="2 <= num_players <= 20"):
        await svc.create_tournament(
            creator=_make_user(),
            name="nope",
            game_type="public_goods",
            num_players=21,
            total_rounds=3,
            round_deadline_s=30,
        )
