"""Integration tests for /ui/admin/tournaments/*.

Self-contained: builds an app with in-memory DB, seeds an admin and a
regular user, and exposes JWT headers for both. Avoids collision with
other integration tests that rely on their own database setup.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import create_access_token, get_password_hash
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, User
from atp.dashboard.tournament.models import (
    Action,
    ActionSource,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def admin_ui_ctx(monkeypatch) -> AsyncGenerator[dict, None]:
    """Yield an app + client + seeded admin/regular headers.

    Uses ``monkeypatch`` so the env mutations here are reverted on
    teardown and cannot leak into later tests. Note that
    ``ATP_SECRET_KEY`` is read once at import time of
    ``atp.dashboard.auth`` into the module-level ``SECRET_KEY``; the
    env-var override here therefore only affects code paths that read
    ``os.environ`` directly (config, rate-limit), while JWT signing /
    verification still uses the already-initialized module SECRET_KEY.
    That is acceptable for integration tests because both sides of the
    JWT round-trip share the same module-global.
    """
    monkeypatch.setenv("ATP_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ATP_DISABLE_AUTH", "false")
    monkeypatch.setenv("ATP_RATE_LIMIT_ENABLED", "false")
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=False,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    async with db.session() as session:
        admin = User(
            username="admin_ui_test",
            email="admin_ui@test.com",
            hashed_password=get_password_hash("pass"),
            is_admin=True,
            is_active=True,
        )
        regular = User(
            username="regular_ui_test",
            email="regular_ui@test.com",
            hashed_password=get_password_hash("pass"),
            is_admin=False,
            is_active=True,
        )
        session.add_all([admin, regular])
        await session.commit()
        await session.refresh(admin)
        await session.refresh(regular)

    admin_jwt = create_access_token(
        data={"sub": admin.username, "user_id": admin.id}, is_admin=True
    )
    regular_jwt = create_access_token(
        data={"sub": regular.username, "user_id": regular.id}, is_admin=False
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "admin_headers": {"Authorization": f"Bearer {admin_jwt}"},
            "regular_headers": {"Authorization": f"Bearer {regular_jwt}"},
            "admin_id": admin.id,
            "regular_id": regular.id,
            "db": db,
        }

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    get_config.cache_clear()


@pytest.mark.anyio
async def test_admin_landing_rejects_anonymous(admin_ui_ctx):
    """No Authorization header → 401."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_admin_landing_rejects_non_admin(admin_ui_ctx):
    """Authenticated regular user → 403."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin", headers=admin_ui_ctx["regular_headers"])
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_landing_renders_for_admin(admin_ui_ctx):
    """Authenticated admin → 200 with admin content."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin", headers=admin_ui_ctx["admin_headers"])
    assert resp.status_code == 200
    assert "Admin" in resp.text
    assert "Tournaments" in resp.text


@pytest.mark.anyio
async def test_admin_tournaments_list_rejects_regular_user(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments", headers=admin_ui_ctx["regular_headers"]
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_tournaments_list_renders_for_admin(admin_ui_ctx):
    """Empty DB case: page still renders with a 'no tournaments yet' fallback."""
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments", headers=admin_ui_ctx["admin_headers"]
    )
    assert resp.status_code == 200
    assert "Tournaments (admin)" in resp.text
    assert "New tournament" in resp.text
    assert "No tournaments yet" in resp.text


@pytest.mark.anyio
async def test_admin_new_tournament_form_renders(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments/new", headers=admin_ui_ctx["admin_headers"]
    )
    assert resp.status_code == 200
    assert 'name="name"' in resp.text
    assert 'name="game_type"' in resp.text
    assert 'name="num_players"' in resp.text
    assert 'name="total_rounds"' in resp.text
    assert 'name="round_deadline_s"' in resp.text
    assert "el_farol" in resp.text


@pytest.mark.anyio
async def test_admin_create_tournament_redirects_to_detail(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "name": "El Farol smoke A",
            "game_type": "el_farol",
            "num_players": "6",
            "total_rounds": "10",
            "round_deadline_s": "30",
        },
        headers=admin_ui_ctx["admin_headers"],
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/ui/admin/tournaments/")


@pytest.mark.anyio
async def test_admin_tournament_detail_404_for_unknown_id(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments/9999999",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_admin_tournament_detail_renders_live_tournament_with_cancel(
    admin_ui_ctx,
):
    """Fresh-created tournament is in ``pending`` status — Cancel must show."""
    client = admin_ui_ctx["client"]
    create_resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "name": "El Farol detail test",
            "game_type": "el_farol",
            "num_players": "6",
            "total_rounds": "10",
            "round_deadline_s": "30",
        },
        headers=admin_ui_ctx["admin_headers"],
        follow_redirects=False,
    )
    assert create_resp.status_code == 303
    detail_url = create_resp.headers["location"]

    resp = await client.get(detail_url, headers=admin_ui_ctx["admin_headers"])
    assert resp.status_code == 200
    assert "Tournament #" in resp.text
    assert "el_farol" in resp.text
    assert "Cancel tournament" in resp.text
    # Post-mortem label must NOT appear on a live tournament.
    assert "Post-mortem" not in resp.text


@pytest.mark.anyio
async def test_admin_tournament_detail_rejects_non_admin(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    # Create first as admin so we have a tournament to visit.
    create_resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "name": "El Farol authz test",
            "game_type": "el_farol",
            "num_players": "6",
            "total_rounds": "10",
            "round_deadline_s": "30",
        },
        headers=admin_ui_ctx["admin_headers"],
        follow_redirects=False,
    )
    detail_url = create_resp.headers["location"]
    resp = await client.get(detail_url, headers=admin_ui_ctx["regular_headers"])
    assert resp.status_code == 403


async def _seed_live_el_farol_in_ctx(ctx: dict) -> int:
    """Seed an ACTIVE El Farol with 1 completed round + 1 in-progress round.

    Returns the tournament id. Uses the same DB as the app in ``ctx``.
    """
    db: Database = ctx["db"]
    admin_id: int = ctx["admin_id"]
    now = datetime.now().replace(microsecond=0)

    async with db.session() as session:
        t = Tournament(
            game_type="el_farol",
            status=TournamentStatus.ACTIVE.value,
            num_players=2,
            total_rounds=3,
            round_deadline_s=30,
            created_by=admin_id,
            created_at=now - timedelta(minutes=2),
            starts_at=now - timedelta(minutes=1),
            pending_deadline=now - timedelta(minutes=1),
        )
        session.add(t)
        await session.flush()

        uid = t.id
        u_alpha = User(
            username=f"bot_alpha_{uid}",
            email=f"alpha_{uid}@t.com",
            hashed_password="x",
            is_active=True,
        )
        u_beta = User(
            username=f"bot_beta_{uid}",
            email=f"beta_{uid}@t.com",
            hashed_password="x",
            is_active=True,
        )
        session.add_all([u_alpha, u_beta])
        await session.flush()

        # LABS-TSA PR-4: non-builtin Participants need agent_id.
        a_alpha = Agent(
            tenant_id="default",
            name="alpha",
            agent_type="mcp",
            owner_id=u_alpha.id,
            config={},
            purpose="tournament",
        )
        a_beta = Agent(
            tenant_id="default",
            name="beta",
            agent_type="mcp",
            owner_id=u_beta.id,
            config={},
            purpose="tournament",
        )
        session.add_all([a_alpha, a_beta])
        await session.flush()

        p_alpha = Participant(
            tournament_id=t.id,
            user_id=u_alpha.id,
            agent_id=a_alpha.id,
            agent_name="alpha",
            total_score=1.0,
        )
        p_beta = Participant(
            tournament_id=t.id,
            user_id=u_beta.id,
            agent_id=a_beta.id,
            agent_name="beta",
            total_score=0.0,
        )
        session.add_all([p_alpha, p_beta])
        await session.flush()

        r1 = Round(
            tournament_id=t.id,
            round_number=1,
            status=RoundStatus.COMPLETED.value,
            started_at=now - timedelta(minutes=1),
            deadline=now - timedelta(seconds=30),
        )
        session.add(r1)
        await session.flush()
        session.add(
            Action(
                round_id=r1.id,
                participant_id=p_alpha.id,
                action_data={"slots": [0]},
                submitted_at=now - timedelta(seconds=45),
                source=ActionSource.SUBMITTED.value,
                payoff=1.0,
            )
        )
        session.add(
            Action(
                round_id=r1.id,
                participant_id=p_beta.id,
                action_data={"slots": []},
                submitted_at=now - timedelta(seconds=29),
                source=ActionSource.TIMEOUT_DEFAULT.value,
                payoff=0.0,
            )
        )

        r2 = Round(
            tournament_id=t.id,
            round_number=2,
            status=RoundStatus.IN_PROGRESS.value,
            started_at=now,
            deadline=now + timedelta(seconds=25),
        )
        session.add(r2)
        await session.commit()
        return t.id


async def _seed_completed_el_farol_in_ctx(ctx: dict) -> int:
    """Seed a COMPLETED El Farol with 2 rounds, 2 participants."""
    db: Database = ctx["db"]
    admin_id: int = ctx["admin_id"]
    now = datetime.now().replace(microsecond=0)

    async with db.session() as session:
        t = Tournament(
            game_type="el_farol",
            status=TournamentStatus.COMPLETED.value,
            num_players=2,
            total_rounds=2,
            round_deadline_s=30,
            created_by=admin_id,
            created_at=now - timedelta(minutes=10),
            starts_at=now - timedelta(minutes=9),
            ends_at=now - timedelta(minutes=1),
            pending_deadline=now - timedelta(minutes=9),
        )
        session.add(t)
        await session.flush()

        uid = t.id
        u_a = User(
            username=f"bot_done_a_{uid}",
            email=f"done_a_{uid}@t.com",
            hashed_password="x",
            is_active=True,
        )
        u_b = User(
            username=f"bot_done_b_{uid}",
            email=f"done_b_{uid}@t.com",
            hashed_password="x",
            is_active=True,
        )
        session.add_all([u_a, u_b])
        await session.flush()

        # LABS-TSA PR-4: non-builtin Participants need agent_id.
        a_a = Agent(
            tenant_id="default",
            name="alpha_done",
            agent_type="mcp",
            owner_id=u_a.id,
            config={},
            purpose="tournament",
        )
        a_b = Agent(
            tenant_id="default",
            name="beta_done",
            agent_type="mcp",
            owner_id=u_b.id,
            config={},
            purpose="tournament",
        )
        session.add_all([a_a, a_b])
        await session.flush()

        p_a = Participant(
            tournament_id=t.id,
            user_id=u_a.id,
            agent_id=a_a.id,
            agent_name="alpha_done",
            total_score=3.0,
            released_at=now - timedelta(minutes=1),
        )
        p_b = Participant(
            tournament_id=t.id,
            user_id=u_b.id,
            agent_id=a_b.id,
            agent_name="beta_done",
            total_score=1.0,
            released_at=now - timedelta(minutes=1),
        )
        session.add_all([p_a, p_b])
        await session.flush()

        for rn in (1, 2):
            r = Round(
                tournament_id=t.id,
                round_number=rn,
                status=RoundStatus.COMPLETED.value,
                started_at=now - timedelta(minutes=5 - rn),
                deadline=now - timedelta(minutes=5 - rn, seconds=-30),
            )
            session.add(r)
            await session.flush()
            for p in (p_a, p_b):
                session.add(
                    Action(
                        round_id=r.id,
                        participant_id=p.id,
                        action_data={"slots": [0, 1]},
                        submitted_at=now - timedelta(minutes=5 - rn),
                        source=ActionSource.SUBMITTED.value,
                        payoff=1.0,
                    )
                )
        await session.commit()
        return t.id


@pytest.mark.anyio
async def test_admin_detail_has_htmx_polling_attributes_for_live(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    resp = await client.get(
        f"/ui/admin/tournaments/{tid}",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 200
    assert f'hx-get="/ui/admin/tournaments/{tid}/activity"' in resp.text
    assert 'hx-trigger="load, every 2s"' in resp.text


@pytest.mark.anyio
async def test_admin_detail_post_mortem_has_no_polling_no_cancel(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_completed_el_farol_in_ctx(admin_ui_ctx)
    resp = await client.get(
        f"/ui/admin/tournaments/{tid}",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 200
    # Polling must be absent.
    assert "hx-trigger" not in resp.text
    # Cancel button must be absent.
    assert "Cancel tournament" not in resp.text
    # Post-mortem marker must be present.
    assert "Post-mortem" in resp.text
    # Activity block is server-rendered inline; table/heatmap ids must
    # be in the HTML.
    assert "admin-activity-table" in resp.text
    assert "admin-activity-heatmap" in resp.text


@pytest.mark.anyio
async def test_admin_activity_fragment_rejects_non_admin(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    resp = await client.get(
        f"/ui/admin/tournaments/{tid}/activity",
        headers=admin_ui_ctx["regular_headers"],
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_activity_fragment_renders_table_and_heatmap(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    resp = await client.get(
        f"/ui/admin/tournaments/{tid}/activity",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 200
    # Table present with both agents.
    assert "admin-activity-table" in resp.text
    assert "alpha" in resp.text
    assert "beta" in resp.text
    # Heatmap with at least one submitted and one timeout cell class.
    assert "admin-activity-heatmap" in resp.text
    assert 'class="cell submitted"' in resp.text
    assert 'class="cell timeout"' in resp.text
    # Deadline countdown visible for live tournaments.
    assert "activity-deadline" in resp.text


async def _get_participant_id(ctx: dict, tournament_id: int, agent_name: str) -> int:
    """Look up a participant id by agent_name in the given tournament."""
    db: Database = ctx["db"]
    from sqlalchemy import select

    async with db.session() as session:
        stmt = select(Participant).where(
            Participant.tournament_id == tournament_id,
            Participant.agent_name == agent_name,
        )
        p = (await session.execute(stmt)).scalars().first()
        assert p is not None, f"seed bug: {agent_name} not found"
        return p.id


@pytest.mark.anyio
async def test_activity_fragment_renders_kick_button_for_live(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    pid = await _get_participant_id(admin_ui_ctx, tid, "alpha")
    resp = await client.get(
        f"/ui/admin/tournaments/{tid}/activity",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 200
    assert f'hx-delete="/api/v1/tournaments/{tid}/participants/{pid}"' in resp.text


@pytest.mark.anyio
async def test_activity_fragment_no_kick_button_for_completed_tournament(
    admin_ui_ctx,
):
    client = admin_ui_ctx["client"]
    tid = await _seed_completed_el_farol_in_ctx(admin_ui_ctx)
    resp = await client.get(
        f"/ui/admin/tournaments/{tid}/activity",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 200
    # Completed tournaments must not expose Kick buttons.
    assert "hx-delete=" not in resp.text


@pytest.mark.anyio
async def test_kick_endpoint_rejects_non_admin(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    pid = await _get_participant_id(admin_ui_ctx, tid, "alpha")
    resp = await client.delete(
        f"/api/v1/tournaments/{tid}/participants/{pid}",
        headers=admin_ui_ctx["regular_headers"],
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_kick_endpoint_returns_204_on_first_kick(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    pid = await _get_participant_id(admin_ui_ctx, tid, "beta")
    resp = await client.delete(
        f"/api/v1/tournaments/{tid}/participants/{pid}",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 204


@pytest.mark.anyio
async def test_kick_endpoint_returns_409_on_double_kick(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    pid = await _get_participant_id(admin_ui_ctx, tid, "beta")
    await client.delete(
        f"/api/v1/tournaments/{tid}/participants/{pid}",
        headers=admin_ui_ctx["admin_headers"],
    )
    resp = await client.delete(
        f"/api/v1/tournaments/{tid}/participants/{pid}",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_kick_endpoint_returns_404_for_unknown_participant(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    tid = await _seed_live_el_farol_in_ctx(admin_ui_ctx)
    resp = await client.delete(
        f"/api/v1/tournaments/{tid}/participants/9999999",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_kick_endpoint_returns_409_on_completed_tournament(admin_ui_ctx):
    """Copilot review PR #58 comment 5 — reject kick on post-mortem."""
    client = admin_ui_ctx["client"]
    tid = await _seed_completed_el_farol_in_ctx(admin_ui_ctx)
    pid = await _get_participant_id(admin_ui_ctx, tid, "alpha_done")
    resp = await client.delete(
        f"/api/v1/tournaments/{tid}/participants/{pid}",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_admin_activity_fragment_404_for_unknown_id(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get(
        "/ui/admin/tournaments/9999999/activity",
        headers=admin_ui_ctx["admin_headers"],
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_admin_create_tournament_rejects_invalid_input(admin_ui_ctx):
    """El Farol requires 2 <= num_players <= 20; 100 must 400."""
    client = admin_ui_ctx["client"]
    resp = await client.post(
        "/ui/admin/tournaments/new",
        data={
            "name": "El Farol too big",
            "game_type": "el_farol",
            "num_players": "100",
            "total_rounds": "10",
            "round_deadline_s": "30",
        },
        headers=admin_ui_ctx["admin_headers"],
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert (
        "Could not create" in resp.text
        or "Validation" in resp.text
        or "el_farol" in resp.text
    )


# ---------------------------------------------------------------------------
# /ui/admin/users
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_admin_users_list_rejects_anonymous(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin/users")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_admin_users_list_rejects_regular_user(admin_ui_ctx):
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin/users", headers=admin_ui_ctx["regular_headers"])
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_admin_users_list_renders_zero_activity(admin_ui_ctx):
    """Fresh users with no tournaments/agents render with zero counters."""
    client = admin_ui_ctx["client"]
    resp = await client.get("/ui/admin/users", headers=admin_ui_ctx["admin_headers"])
    assert resp.status_code == 200
    assert "admin_ui_test" in resp.text
    assert "regular_ui_test" in resp.text
    # Both users are seeded without any associated tournaments/agents
    # so every counter column should render zero. Count the zero-cells
    # conservatively: 2 users × 3 counters = at least 6 occurrences of
    # ">0<" in the HTML.
    assert resp.text.count(">0<") >= 6


@pytest.mark.anyio
async def test_admin_users_list_counts_activity(admin_ui_ctx):
    """Counters include tournaments_created, tournaments_joined, agents_owned."""
    client = admin_ui_ctx["client"]
    db = admin_ui_ctx["db"]
    regular_id = admin_ui_ctx["regular_id"]

    async with db.session() as session:
        agent = Agent(
            name="qa-bot",
            agent_type="http",
            owner_id=regular_id,
            config={},
        )
        session.add(agent)
        await session.flush()

        tournament = Tournament(
            game_type="el_farol",
            num_players=3,
            total_rounds=2,
            round_deadline_s=30,
            status=TournamentStatus.PENDING,
            created_by=regular_id,
            config={"name": "activity test"},
            pending_deadline=datetime.now() + timedelta(minutes=5),
        )
        session.add(tournament)
        await session.flush()

        # ck_participants_agent_xor_builtin requires exactly one of
        # agent_id / builtin_strategy, so link to the agent just seeded.
        session.add(
            Participant(
                tournament_id=tournament.id,
                user_id=regular_id,
                agent_name="qa-bot",
                agent_id=agent.id,
            )
        )
        await session.commit()

    resp = await client.get("/ui/admin/users", headers=admin_ui_ctx["admin_headers"])
    assert resp.status_code == 200
    # Locate the regular user's row and verify each counter column shows
    # 1 (one tournament created, one participation, one agent). The
    # admin row is still all zeros so we can't just search for ">1<"
    # globally.
    row_start = resp.text.index("regular_ui_test")
    row_end = resp.text.index("</tr>", row_start)
    row = resp.text[row_start:row_end]
    assert row.count(">1<") == 3
