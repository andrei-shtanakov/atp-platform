"""Unit tests for the winners aggregation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import DEFAULT_TENANT_ID, Agent, User
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.services.winners import (
    _hall_of_fame_query,
    _winners_query,
)


async def _make_user(session: AsyncSession, username: str) -> User:
    u = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(u)
    await session.flush()
    return u


async def _make_agent(
    session: AsyncSession,
    *,
    owner: User,
    name: str,
    description: str | None = None,
    version: str = "1",
    deleted_at: datetime | None = None,
) -> Agent:
    a = Agent(
        tenant_id=DEFAULT_TENANT_ID,
        name=name,
        agent_type="tournament",
        owner_id=owner.id,
        description=description,
        version=version,
        deleted_at=deleted_at,
        purpose="tournament",
    )
    session.add(a)
    await session.flush()
    return a


async def _make_tournament(
    session: AsyncSession,
    *,
    game_type: str = "el_farol",
    status: TournamentStatus = TournamentStatus.COMPLETED,
    join_token: str | None = None,
    name: str = "T",
    num_players: int = 2,
    total_rounds: int = 5,
    starts_at: datetime | None = None,
    ends_at: datetime | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> Tournament:
    starts_at = starts_at or datetime(2026, 5, 1, 12, 0, 0)
    ends_at = ends_at or starts_at + timedelta(minutes=10)
    t = Tournament(
        tenant_id=tenant_id,
        game_type=game_type,
        config={"name": name},
        status=status,
        starts_at=starts_at,
        ends_at=ends_at,
        num_players=num_players,
        total_rounds=total_rounds,
        round_deadline_s=30,
        join_token=join_token,
        pending_deadline=starts_at,
    )
    session.add(t)
    await session.flush()
    return t


async def _make_participant(
    session: AsyncSession,
    *,
    tournament: Tournament,
    user: User | None = None,
    agent: Agent | None = None,
    builtin_strategy: str | None = None,
    agent_name: str = "agent",
    total_score: float | None = None,
) -> Participant:
    p = Participant(
        tournament_id=tournament.id,
        user_id=user.id if user else None,
        agent_id=agent.id if agent else None,
        builtin_strategy=builtin_strategy,
        agent_name=agent_name,
        total_score=total_score,
    )
    session.add(p)
    await session.flush()
    return p


@pytest.mark.anyio
async def test_winners_query_returns_empty_for_no_participants(
    session: AsyncSession,
):
    t = await _make_tournament(session)
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert rows == []


async def _make_round_with_action(
    session: AsyncSession,
    *,
    participant: Participant,
    round_number: int,
    payoff: float | None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
    model_id: str | None = None,
) -> Action:
    rnd = Round(
        tournament_id=participant.tournament_id,
        round_number=round_number,
        status=RoundStatus.COMPLETED,
        deadline=datetime(2026, 5, 1, 12, 0, 0),
    )
    session.add(rnd)
    await session.flush()
    act = Action(
        round_id=rnd.id,
        participant_id=participant.id,
        action_data={},
        payoff=payoff,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model_id=model_id,
    )
    session.add(act)
    await session.flush()
    return act


@pytest.mark.anyio
async def test_winners_query_single_participant_with_telemetry(
    session: AsyncSession,
):
    t = await _make_tournament(session)
    alice = await _make_user(session, "alice")
    agent = await _make_agent(
        session, owner=alice, name="alfa", description="greedy spammer"
    )
    p = await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=42.0,
    )
    await _make_round_with_action(
        session,
        participant=p,
        round_number=1,
        payoff=42.0,
        tokens_in=100,
        tokens_out=80,
        cost_usd=0.01,
        model_id="gpt-4o-mini",
    )
    await _make_round_with_action(
        session,
        participant=p,
        round_number=2,
        payoff=0.0,
        tokens_in=50,
        tokens_out=40,
        cost_usd=0.005,
        model_id="gpt-4o-mini",
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 1
    e = rows[0]
    assert e.rank == 1
    assert e.agent_name == "alfa"
    assert e.agent_description == "greedy spammer"
    assert e.owner_username == "alice"
    assert e.score == 42.0
    assert e.tokens_in == 150
    assert e.tokens_out == 120
    assert e.cost_usd == pytest.approx(0.015)
    assert e.model_id == "gpt-4o-mini"


@pytest.mark.anyio
async def test_winners_query_builtin_owner_is_system(session: AsyncSession):
    t = await _make_tournament(session)
    p = await _make_participant(
        session,
        tournament=t,
        user=None,
        agent=None,
        builtin_strategy="el_farol/random",
        agent_name="el_farol/random",
        total_score=10.0,
    )
    await _make_round_with_action(session, participant=p, round_number=1, payoff=10.0)
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 1
    assert rows[0].owner_username == "system"
    assert rows[0].agent_description == "built-in strategy"
    assert rows[0].model_id is None  # no telemetry recorded
    assert rows[0].tokens_in is None
    assert rows[0].cost_usd is None


@pytest.mark.anyio
async def test_winners_query_archived_agent_gets_suffix(session: AsyncSession):
    t = await _make_tournament(session)
    alice = await _make_user(session, "alice")
    agent = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        deleted_at=datetime(2026, 5, 1, 11, 0, 0),
    )
    await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=5.0,
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 1
    assert rows[0].agent_name == "alfa (archived)"


@pytest.mark.anyio
async def test_winners_query_mixed_model_id(session: AsyncSession):
    t = await _make_tournament(session)
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    p = await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=10.0,
    )
    await _make_round_with_action(
        session, participant=p, round_number=1, payoff=5.0, model_id="gpt-4o-mini"
    )
    await _make_round_with_action(
        session,
        participant=p,
        round_number=2,
        payoff=5.0,
        model_id="claude-haiku-4-5",
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert rows[0].model_id == "mixed"


@pytest.mark.anyio
async def test_winners_query_dense_rank_with_ties(session: AsyncSession):
    t = await _make_tournament(session, num_players=3)
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    carol = await _make_user(session, "carol")
    a = await _make_agent(session, owner=alice, name="a")
    b = await _make_agent(session, owner=bob, name="b")
    c = await _make_agent(session, owner=carol, name="c")
    await _make_participant(
        session, tournament=t, user=alice, agent=a, agent_name="a", total_score=100.0
    )
    await _make_participant(
        session, tournament=t, user=bob, agent=b, agent_name="b", total_score=100.0
    )
    await _make_participant(
        session, tournament=t, user=carol, agent=c, agent_name="c", total_score=90.0
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert [r.rank for r in rows] == [1, 1, 3]


@pytest.mark.anyio
async def test_winners_query_null_score_sorted_last(session: AsyncSession):
    t = await _make_tournament(session, num_players=2)
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    a = await _make_agent(session, owner=alice, name="a")
    b = await _make_agent(session, owner=bob, name="b")
    await _make_participant(
        session, tournament=t, user=alice, agent=a, agent_name="a", total_score=None
    )
    await _make_participant(
        session, tournament=t, user=bob, agent=b, agent_name="b", total_score=10.0
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert [r.agent_name for r in rows] == ["b", "a"]
    assert rows[1].score is None


@pytest.mark.anyio
async def test_winners_query_all_null_scores_get_rank_one(
    session: AsyncSession,
):
    """Regression: when every participant has total_score=None, ranks
    must still start at 1 (not 0). The competition-ranking sentinel
    must not collide with NULL scores."""
    t = await _make_tournament(session, num_players=2)
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    a = await _make_agent(session, owner=alice, name="a")
    b = await _make_agent(session, owner=bob, name="b")
    await _make_participant(
        session, tournament=t, user=alice, agent=a, agent_name="a", total_score=None
    )
    await _make_participant(
        session, tournament=t, user=bob, agent=b, agent_name="b", total_score=None
    )
    await session.commit()

    rows = await _winners_query(session, t.id)
    assert len(rows) == 2
    # Both NULL-score rows tie at rank 1 (competition ranking, ties share a rank).
    assert [r.rank for r in rows] == [1, 1]


@pytest.mark.anyio
async def test_hall_of_fame_query_empty_when_no_tournaments(
    session: AsyncSession,
):
    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 0
    assert entries == []


@pytest.mark.anyio
async def test_hall_of_fame_aggregates_two_tournaments(session: AsyncSession):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    t1 = await _make_tournament(session, name="T1")
    t2 = await _make_tournament(session, name="T2")
    p1 = await _make_participant(
        session,
        tournament=t1,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=10.0,
    )
    # Releasing the first participant unblocks the
    # ``uq_participant_agent_active`` partial-unique index so the same
    # agent can be seated again in a later tournament.
    p1.released_at = datetime(2026, 5, 1, 13, 0, 0)
    await session.flush()
    await _make_participant(
        session,
        tournament=t2,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=15.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].total_score == 25.0
    assert entries[0].tournaments_count == 2
    assert entries[0].agent_name == "alfa"
    assert entries[0].owner_username == "alice"


@pytest.mark.anyio
async def test_hall_of_fame_aggregates_versions_of_same_agent(
    session: AsyncSession,
):
    alice = await _make_user(session, "alice")
    v1 = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="1",
        description="old desc",
    )
    v2 = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="2",
        description="new desc",
    )
    t1 = await _make_tournament(session, name="T1")
    t2 = await _make_tournament(session, name="T2")
    await _make_participant(
        session,
        tournament=t1,
        user=alice,
        agent=v1,
        agent_name="alfa",
        total_score=10.0,
    )
    await _make_participant(
        session,
        tournament=t2,
        user=alice,
        agent=v2,
        agent_name="alfa",
        total_score=20.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1  # one logical agent, two versions collapsed
    assert entries[0].total_score == 30.0
    # I-1 regression guard: latest version (highest Agent.id) wins.
    assert entries[0].agent_description == "new desc"


@pytest.mark.anyio
async def test_hall_of_fame_excludes_builtins(session: AsyncSession):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    t = await _make_tournament(session)
    await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=10.0,
    )
    await _make_participant(
        session,
        tournament=t,
        user=None,
        agent=None,
        builtin_strategy="el_farol/random",
        agent_name="el_farol/random",
        total_score=8.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].agent_name == "alfa"


@pytest.mark.anyio
async def test_hall_of_fame_excludes_private_tournaments(
    session: AsyncSession,
):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    pub = await _make_tournament(session, name="public")
    priv = await _make_tournament(session, name="private", join_token="secret")
    p_pub = await _make_participant(
        session,
        tournament=pub,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=10.0,
    )
    # Release the public-tournament participant so the same agent can be
    # seated in the private tournament without violating the partial
    # unique index on (agent_id) WHERE released_at IS NULL.
    p_pub.released_at = datetime(2026, 5, 1, 13, 0, 0)
    await session.flush()
    await _make_participant(
        session,
        tournament=priv,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=99.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].total_score == 10.0  # private tournament excluded


@pytest.mark.anyio
async def test_hall_of_fame_excludes_other_tenants(session: AsyncSession):
    alice = await _make_user(session, "alice")
    agent = await _make_agent(session, owner=alice, name="alfa")
    t = await _make_tournament(session, tenant_id="other-tenant")
    await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=agent,
        agent_name="alfa",
        total_score=42.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 0


@pytest.mark.anyio
async def test_hall_of_fame_archived_lineage(session: AsyncSession):
    alice = await _make_user(session, "alice")
    deleted = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="1",
        deleted_at=datetime(2026, 5, 1, 0, 0, 0),
    )
    t = await _make_tournament(session)
    await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=deleted,
        agent_name="alfa",
        total_score=10.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    assert entries[0].agent_name == "alfa (archived)"
    assert entries[0].agent_description is None


@pytest.mark.anyio
async def test_hall_of_fame_ordering_and_tiebreaker(session: AsyncSession):
    alice = await _make_user(session, "alice")
    bob = await _make_user(session, "bob")
    a = await _make_agent(session, owner=alice, name="z")
    b = await _make_agent(session, owner=bob, name="a")
    t = await _make_tournament(session, num_players=2)
    # Tie on score — tiebreaker is owner_id ASC, then name ASC.
    await _make_participant(
        session,
        tournament=t,
        user=alice,
        agent=a,
        agent_name="z",
        total_score=10.0,
    )
    await _make_participant(
        session,
        tournament=t,
        user=bob,
        agent=b,
        agent_name="a",
        total_score=10.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 2
    # alice has lower id (created first), so she wins the tiebreaker.
    assert entries[0].owner_username == "alice"
    assert entries[1].owner_username == "bob"


@pytest.mark.anyio
async def test_hall_of_fame_pagination(session: AsyncSession):
    t = await _make_tournament(session, num_players=4)
    for i, score in enumerate([40, 30, 20, 10], start=1):
        u = await _make_user(session, f"u{i}")
        ag = await _make_agent(session, owner=u, name=f"agent{i}")
        await _make_participant(
            session,
            tournament=t,
            user=u,
            agent=ag,
            agent_name=f"agent{i}",
            total_score=float(score),
        )
    await session.commit()

    total, page1 = await _hall_of_fame_query(session, limit=2, offset=0)
    total2, page2 = await _hall_of_fame_query(session, limit=2, offset=2)
    assert total == 4 and total2 == 4
    assert [e.rank for e in page1] == [1, 2]
    assert [e.rank for e in page2] == [3, 4]
    assert page1[0].total_score == 40.0
    assert page2[0].total_score == 20.0


@pytest.mark.anyio
async def test_hall_of_fame_mixed_extant_and_deleted_versions(
    session: AsyncSession,
):
    """When an agent has v1 soft-deleted and v2 live, the lineage row
    must show no ``(archived)`` suffix and use v2's description (the
    latest non-deleted version)."""
    alice = await _make_user(session, "alice")
    v1_deleted = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="1",
        description="old description",
        deleted_at=datetime(2026, 5, 1, 10, 0, 0),
    )
    v2_live = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="2",
        description="new description",
    )
    t1 = await _make_tournament(session, name="T1")
    t2 = await _make_tournament(session, name="T2")

    # Same uq_participant_agent_active workaround used elsewhere — the
    # first participant must be released before the second flush since
    # both reference (different versions of) the same agent name.
    p1 = await _make_participant(
        session,
        tournament=t1,
        user=alice,
        agent=v1_deleted,
        agent_name="alfa",
        total_score=10.0,
    )
    p1.released_at = datetime(2026, 5, 1, 13, 0, 0)
    await session.flush()
    await _make_participant(
        session,
        tournament=t2,
        user=alice,
        agent=v2_live,
        agent_name="alfa",
        total_score=20.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    e = entries[0]
    assert e.total_score == 30.0
    # Lineage has at least one extant version → no archived suffix.
    assert e.agent_name == "alfa"
    # Description comes from the latest non-deleted version (v2).
    assert e.agent_description == "new description"


@pytest.mark.anyio
async def test_hall_of_fame_latest_desc_picks_newest_version_not_most_edited(
    session: AsyncSession,
):
    """Regression for Copilot review on PR #118: ``latest_desc`` must
    pick the newest VERSION (highest Agent.id), not the most-recently-
    updated row. If v1 is edited after v2 was created, v2 (higher id)
    must still win."""
    alice = await _make_user(session, "alice")
    # v1 is created first → lower Agent.id.
    v1 = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="1",
        description="v1 desc",
    )
    # v2 is created second → higher Agent.id.
    v2 = await _make_agent(
        session,
        owner=alice,
        name="alfa",
        version="2",
        description="v2 desc",
    )
    # Now simulate "v1 edited after v2 created" — bump v1.updated_at
    # to be later than v2.updated_at by direct ORM update.
    v1.updated_at = datetime(2026, 12, 31, 23, 59, 59)
    v1.description = "v1 edited later"
    await session.flush()

    t1 = await _make_tournament(session, name="T1")
    t2 = await _make_tournament(session, name="T2")
    p1 = await _make_participant(
        session,
        tournament=t1,
        user=alice,
        agent=v1,
        agent_name="alfa",
        total_score=10.0,
    )
    p1.released_at = datetime(2026, 5, 1, 13, 0, 0)
    await session.flush()
    await _make_participant(
        session,
        tournament=t2,
        user=alice,
        agent=v2,
        agent_name="alfa",
        total_score=20.0,
    )
    await session.commit()

    total, entries = await _hall_of_fame_query(session, limit=50, offset=0)
    assert total == 1
    # v2 (higher Agent.id) wins regardless of v1's later updated_at.
    assert entries[0].agent_description == "v2 desc"
