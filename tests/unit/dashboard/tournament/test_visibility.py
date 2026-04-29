"""Tests for ``is_tournament_visible_to`` (visibility predicate).

Mirrors the SQL filter used in ``ui.py::ui_matches`` and gates both the
live tournament dashboard JSON endpoint and its SSE stream.

Rules:
  * ``tournament.join_token IS NULL`` -> visible to everyone (including anon).
  * Otherwise visible only to admins, the creator, or a Participant.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.visibility import is_tournament_visible_to


async def _make_tournament(
    session: AsyncSession,
    *,
    creator: User,
    join_token: str | None,
) -> Tournament:
    t = Tournament(
        game_type="el_farol",
        status=TournamentStatus.PENDING,
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
        created_by=creator.id,
        config={},
        rules={},
        join_token=join_token,
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t


# ---- Public tournaments: join_token IS NULL ----------------------------------


@pytest.mark.anyio
async def test_public_tournament_visible_to_anonymous(
    session: AsyncSession, admin_user: User
) -> None:
    # GIVEN a public tournament (no join token)
    t = await _make_tournament(session, creator=admin_user, join_token=None)
    # WHEN/THEN an anonymous viewer can see it
    assert await is_tournament_visible_to(session, t, None) is True


@pytest.mark.anyio
async def test_public_tournament_visible_to_random_user(
    session: AsyncSession,
    admin_user: User,
    alice: User,
) -> None:
    # GIVEN a public tournament owned by admin and a random non-participant
    t = await _make_tournament(session, creator=admin_user, join_token=None)
    # WHEN/THEN alice (no relationship) can still see it
    assert await is_tournament_visible_to(session, t, alice) is True


# ---- Private tournaments: join_token IS NOT NULL -----------------------------


@pytest.mark.anyio
async def test_private_tournament_hidden_from_anonymous(
    session: AsyncSession, admin_user: User
) -> None:
    # GIVEN a private tournament
    t = await _make_tournament(session, creator=admin_user, join_token="secret")
    # WHEN/THEN an anonymous viewer is denied
    assert await is_tournament_visible_to(session, t, None) is False


@pytest.mark.anyio
async def test_private_tournament_visible_to_admin(
    session: AsyncSession,
    admin_user: User,
    alice: User,
) -> None:
    # GIVEN a private tournament owned by alice (non-admin)
    t = await _make_tournament(session, creator=alice, join_token="secret")
    # WHEN/THEN admin can see anything
    assert await is_tournament_visible_to(session, t, admin_user) is True


@pytest.mark.anyio
async def test_private_tournament_visible_to_creator(
    session: AsyncSession, alice: User
) -> None:
    # GIVEN a private tournament created by alice
    t = await _make_tournament(session, creator=alice, join_token="secret")
    # WHEN/THEN alice (creator) can see it
    assert await is_tournament_visible_to(session, t, alice) is True


@pytest.mark.anyio
async def test_private_tournament_visible_to_participant(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
) -> None:
    # GIVEN a private tournament with bob as a participant (admin created it)
    t = await _make_tournament(session, creator=admin_user, join_token="secret")
    p = Participant(
        tournament_id=t.id,
        user_id=bob.id,
        agent_name="bob-agent",
        agent_id=None,
        builtin_strategy="el_farol/calibrated",
    )
    session.add(p)
    await session.commit()
    # WHEN/THEN bob can see it as a participant
    assert await is_tournament_visible_to(session, t, bob) is True
    # AND alice (no relationship) still cannot
    assert await is_tournament_visible_to(session, t, alice) is False


@pytest.mark.anyio
async def test_private_tournament_hidden_from_random_user(
    session: AsyncSession,
    admin_user: User,
    alice: User,
) -> None:
    # GIVEN a private tournament owned by admin
    t = await _make_tournament(session, creator=admin_user, join_token="secret")
    # WHEN/THEN alice (not admin, not creator, not participant) is denied
    assert await is_tournament_visible_to(session, t, alice) is False
