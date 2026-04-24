"""Tests for tournament UI routes."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

# Eager-register RBAC models onto the shared Base so Base.metadata
# picks up `roles` / `role_permissions` / `user_roles` in the
# per-test init_database() call below. `_seed_default_roles` runs on
# startup and would fail without these tables.
import atp.dashboard.rbac.models  # noqa: F401
import atp.dashboard.tournament.models  # noqa: F401  (Participant etc.)
from atp.dashboard.database import init_database, set_database
from atp.dashboard.models import User
from atp.dashboard.tournament.models import (
    Action,
    ActionSource,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def client():
    # LABS-TSA PR-4: `httpx.ASGITransport` does not trigger FastAPI
    # lifespan events, so the module-level `_database` singleton that
    # `_seed_tournament` below relies on would be left un-initialised
    # and hit "no such table" at runtime. Initialise the in-memory DB
    # explicitly for the duration of each test.
    db = await init_database(url="sqlite+aiosqlite:///:memory:")
    try:
        app = create_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        await db.close()
        set_database(None)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_tournament_list_returns_200(client: AsyncClient):
    resp = await client.get("/ui/tournaments")
    assert resp.status_code == 200
    assert "Tournaments" in resp.text


@pytest.mark.anyio
async def test_tournament_list_partial_returns_200(client: AsyncClient):
    resp = await client.get("/ui/tournaments?partial=1")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_tournament_detail_404_for_missing(client: AsyncClient):
    resp = await client.get("/ui/tournaments/99999")
    assert resp.status_code == 404
    assert "Not Found" in resp.text


async def _seed_tournament(client: AsyncClient) -> int:
    """Create a completed 2-player, 2-round PD tournament in the test DB."""
    import uuid

    from atp.dashboard.database import get_database

    uid = uuid.uuid4().hex[:8]
    async with get_database().session_factory() as session:
        user = User(
            username=f"test_admin_{uid}",
            email=f"a_{uid}@test.com",
            is_admin=True,
            hashed_password="x",
        )
        session.add(user)
        await session.flush()

        now = datetime.utcnow()
        t = Tournament(
            game_type="prisoners_dilemma",
            config={"name": "Test PD"},
            status=TournamentStatus.COMPLETED,
            num_players=2,
            total_rounds=2,
            round_deadline_s=30,
            created_by=user.id,
            created_at=now - timedelta(minutes=5),
            starts_at=now - timedelta(minutes=4),
            ends_at=now,
            pending_deadline=now,
        )
        session.add(t)
        await session.flush()

        # LABS-TSA PR-4: Participants must satisfy the agent-xor-builtin
        # CHECK — real-agent participants need agent_id NOT NULL.
        from atp.dashboard.models import Agent

        a1 = Agent(
            name=f"alice_{uid}",
            agent_type="mcp",
            owner_id=user.id,
            purpose="tournament",
        )
        session.add(a1)
        await session.flush()
        p1 = Participant(
            tournament_id=t.id,
            user_id=user.id,
            agent_id=a1.id,
            agent_name="alice",
            total_score=6.0,
        )
        p2_user = User(
            username=f"bot_bob_{uid}", email=f"b_{uid}@test.com", hashed_password="x"
        )
        session.add(p2_user)
        await session.flush()
        a2 = Agent(
            name=f"bob_{uid}",
            agent_type="mcp",
            owner_id=p2_user.id,
            purpose="tournament",
        )
        session.add(a2)
        await session.flush()
        p2 = Participant(
            tournament_id=t.id,
            user_id=p2_user.id,
            agent_id=a2.id,
            agent_name="bob",
            total_score=6.0,
        )
        session.add_all([p1, p2])
        await session.flush()

        for rn in (1, 2):
            r = Round(
                tournament_id=t.id,
                round_number=rn,
                status=RoundStatus.COMPLETED,
                started_at=now - timedelta(minutes=4 - rn),
            )
            session.add(r)
            await session.flush()
            for p in (p1, p2):
                session.add(
                    Action(
                        round_id=r.id,
                        participant_id=p.id,
                        action_data={"choice": "cooperate"},
                        submitted_at=now,
                        payoff=3.0,
                        source=ActionSource.SUBMITTED,
                    )
                )
        await session.commit()
        return t.id


@pytest.mark.anyio
async def test_tournament_detail_returns_200(client: AsyncClient):
    tid = await _seed_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    assert "Test PD" in resp.text
    assert "alice" in resp.text
    assert "bob" in resp.text


@pytest.mark.anyio
async def test_tournament_detail_shows_round_history(client: AsyncClient):
    tid = await _seed_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    assert "cooperate" in resp.text


@pytest.mark.anyio
async def test_tournament_detail_partial_live(client: AsyncClient):
    tid = await _seed_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}?partial=live")
    assert resp.status_code == 200


async def _seed_el_farol_tournament(client: AsyncClient) -> int:
    """Create a completed 2-player, 1-round El Farol tournament."""
    import uuid

    from atp.dashboard.database import get_database

    uid = uuid.uuid4().hex[:8]
    async with get_database().session_factory() as session:
        user = User(
            username=f"ef_admin_{uid}",
            email=f"ef_a_{uid}@test.com",
            is_admin=True,
            hashed_password="x",
        )
        session.add(user)
        await session.flush()

        now = datetime.utcnow()
        t = Tournament(
            game_type="el_farol",
            config={"name": "Test EF", "num_slots": 16},
            status=TournamentStatus.COMPLETED,
            num_players=2,
            total_rounds=1,
            round_deadline_s=30,
            created_by=user.id,
            created_at=now - timedelta(minutes=5),
            starts_at=now - timedelta(minutes=4),
            ends_at=now,
            pending_deadline=now,
        )
        session.add(t)
        await session.flush()

        # LABS-TSA PR-4 CHECK: agent_id must be populated for real
        # agent participants.
        from atp.dashboard.models import Agent

        a1 = Agent(
            name=f"ef_alice_{uid}",
            agent_type="mcp",
            owner_id=user.id,
            purpose="tournament",
        )
        session.add(a1)
        await session.flush()
        p1 = Participant(
            tournament_id=t.id,
            user_id=user.id,
            agent_id=a1.id,
            agent_name="alice",
            total_score=2.0,
        )
        p2_user = User(
            username=f"ef_bob_{uid}", email=f"ef_b_{uid}@test.com", hashed_password="x"
        )
        session.add(p2_user)
        await session.flush()
        a2 = Agent(
            name=f"ef_bob_{uid}",
            agent_type="mcp",
            owner_id=p2_user.id,
            purpose="tournament",
        )
        session.add(a2)
        await session.flush()
        p2 = Participant(
            tournament_id=t.id,
            user_id=p2_user.id,
            agent_id=a2.id,
            agent_name="bob",
            total_score=0.0,
        )
        session.add_all([p1, p2])
        await session.flush()

        r = Round(
            tournament_id=t.id,
            round_number=1,
            status=RoundStatus.COMPLETED,
            started_at=now - timedelta(minutes=3),
        )
        session.add(r)
        await session.flush()
        session.add(
            Action(
                round_id=r.id,
                participant_id=p1.id,
                action_data={"slots": [0, 1, 2]},
                submitted_at=now,
                payoff=2.0,
                source=ActionSource.SUBMITTED,
            )
        )
        session.add(
            Action(
                round_id=r.id,
                participant_id=p2.id,
                action_data={"slots": []},
                submitted_at=now,
                payoff=0.0,
                source=ActionSource.SUBMITTED,
            )
        )
        await session.commit()
        return t.id


@pytest.mark.anyio
async def test_el_farol_round_history_shows_slots(client: AsyncClient):
    tid = await _seed_el_farol_tournament(client)
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    assert "0, 1, 2" in resp.text
    assert "stay home" in resp.text
    # Must not fall back to the PD "—" placeholder for a matched action
    assert "cooperate" not in resp.text
    assert "defect" not in resp.text


async def _seed_pd_tournament_with_reasoning(
    client: AsyncClient,
    *,
    tournament_status: TournamentStatus,
    reasoning_alice: str = "opening move: cooperate",
    reasoning_bob: str | None = "mirror strategy",
) -> tuple[int, int, int]:
    """Seed a 2-player PD tournament with reasoning on each action.

    Returns (tournament_id, alice_user_id, bob_user_id).
    """
    import uuid

    from atp.dashboard.database import get_database

    uid = uuid.uuid4().hex[:8]
    async with get_database().session_factory() as session:
        admin = User(
            username=f"pd_admin_{uid}",
            email=f"pd_admin_{uid}@test.com",
            is_admin=True,
            hashed_password="x",
        )
        session.add(admin)
        await session.flush()

        alice = User(
            username=f"pd_alice_{uid}",
            email=f"pd_alice_{uid}@test.com",
            hashed_password="x",
        )
        bob = User(
            username=f"pd_bob_{uid}",
            email=f"pd_bob_{uid}@test.com",
            hashed_password="x",
        )
        session.add_all([alice, bob])
        await session.flush()

        now = datetime.utcnow()
        t = Tournament(
            game_type="prisoners_dilemma",
            config={"name": "Test PD-R"},
            status=tournament_status,
            num_players=2,
            total_rounds=1,
            round_deadline_s=30,
            created_by=admin.id,
            created_at=now - timedelta(minutes=5),
            starts_at=now - timedelta(minutes=4),
            ends_at=now if tournament_status == TournamentStatus.COMPLETED else None,
            pending_deadline=now,
        )
        session.add(t)
        await session.flush()

        # LABS-TSA PR-4 CHECK: populate agent_id.
        from atp.dashboard.models import Agent

        a_alice = Agent(
            name=f"pdr_alice_{alice.id}",
            agent_type="mcp",
            owner_id=alice.id,
            purpose="tournament",
        )
        a_bob = Agent(
            name=f"pdr_bob_{bob.id}",
            agent_type="mcp",
            owner_id=bob.id,
            purpose="tournament",
        )
        session.add_all([a_alice, a_bob])
        await session.flush()
        p_alice = Participant(
            tournament_id=t.id,
            user_id=alice.id,
            agent_id=a_alice.id,
            agent_name="alice",
            total_score=3.0,
        )
        p_bob = Participant(
            tournament_id=t.id,
            user_id=bob.id,
            agent_id=a_bob.id,
            agent_name="bob",
            total_score=3.0,
        )
        session.add_all([p_alice, p_bob])
        await session.flush()

        r = Round(
            tournament_id=t.id,
            round_number=1,
            status=RoundStatus.COMPLETED,
            started_at=now - timedelta(minutes=3),
        )
        session.add(r)
        await session.flush()
        session.add(
            Action(
                round_id=r.id,
                participant_id=p_alice.id,
                action_data={"choice": "cooperate"},
                submitted_at=now,
                payoff=3.0,
                source=ActionSource.SUBMITTED,
                reasoning=reasoning_alice,
            )
        )
        session.add(
            Action(
                round_id=r.id,
                participant_id=p_bob.id,
                action_data={"choice": "cooperate"},
                submitted_at=now,
                payoff=3.0,
                source=ActionSource.SUBMITTED,
                reasoning=reasoning_bob,
            )
        )
        await session.commit()
        return t.id, alice.id, bob.id


@pytest.mark.anyio
async def test_reasoning_visible_on_completed_tournament(client: AsyncClient):
    tid, _, _ = await _seed_pd_tournament_with_reasoning(
        client, tournament_status=TournamentStatus.COMPLETED
    )
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    # Both reasonings rendered — completion gate lets everyone see
    assert "opening move: cooperate" in resp.text
    assert "mirror strategy" in resp.text
    assert "💭" in resp.text


@pytest.mark.anyio
async def test_reasoning_hidden_for_anon_on_active_tournament(client: AsyncClient):
    tid, _, _ = await _seed_pd_tournament_with_reasoning(
        client, tournament_status=TournamentStatus.ACTIVE
    )
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    # Active tournament, anonymous caller (no bearer) — reasoning must be
    # masked
    assert "opening move: cooperate" not in resp.text
    assert "mirror strategy" not in resp.text


@pytest.mark.anyio
async def test_reasoning_xss_is_escaped(client: AsyncClient):
    payload = "<script>alert(1)</script>{{ 7*7 }}"
    tid, _, _ = await _seed_pd_tournament_with_reasoning(
        client,
        tournament_status=TournamentStatus.COMPLETED,
        reasoning_alice=payload,
        reasoning_bob=None,
    )
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    # Script tag must be escaped; Jinja template rendering (7*7) must
    # NOT be evaluated.
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text
    # Jinja template tags inside user content must render as literal text
    # rather than be evaluated server-side. Check the raw substring directly;
    # searching for the evaluated value ("49") is too loose — it can
    # appear incidentally in a UUID hex fragment or a CSS color value.
    assert "{{ 7*7 }}" in resp.text


@pytest.mark.anyio
async def test_reasoning_absent_when_field_none(client: AsyncClient):
    """Even for a completed tournament, no 💭 icon is rendered on actions
    that have no reasoning."""
    tid, _, _ = await _seed_pd_tournament_with_reasoning(
        client,
        tournament_status=TournamentStatus.COMPLETED,
        reasoning_alice=None,
        reasoning_bob=None,
    )
    resp = await client.get(f"/ui/tournaments/{tid}")
    assert resp.status_code == 200
    assert "💭" not in resp.text
