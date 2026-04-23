"""Integration tests for Phase 7 El Farol additive columns on GameResult.

Verifies that the additive nullable columns (match_id, game_version, num_days,
num_slots, max_intervals, max_total_slots, capacity_ratio, capacity_threshold,
actions_json, day_aggregates_json, round_payoffs_json, agents_json) round-trip
correctly via the ORM, and that legacy rows still read with NULL for the new
columns.
"""

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, GameResult

# --- Fixtures (self-contained; mirrors test_game_routes.py) -----------------


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    """Create and configure an in-memory SQLite test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
async def async_session(
    test_database: Database,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for a single test."""
    async with test_database.session() as session:
        yield session


# --- Tests ------------------------------------------------------------------


class TestElFarolColumnsOnGameResult:
    """Phase 7: additive El Farol columns on GameResult."""

    @pytest.mark.anyio
    async def test_writes_and_reads_new_columns(
        self,
        async_session: AsyncSession,
    ) -> None:
        """GIVEN all new columns populated
        WHEN row is inserted and re-read
        THEN every scalar and JSON column round-trips identically.
        """
        # GIVEN
        actions = [
            {
                "day": 0,
                "agent_id": "p0",
                "slot_ids": [1, 2],
                "interval_count": 1,
                "total_slots": 2,
                "reasoning": "go early",
            }
        ]
        day_aggregates = [
            {
                "day": 0,
                "attendance": 5,
                "over_capacity": False,
                "total_slots_used": 5,
            }
        ]
        round_payoffs = [{"p0": 3.0, "p1": 3.0}]
        agents = [{"agent_id": "p0", "display_name": "tft", "user_id": "u1"}]

        game = GameResult(
            game_name="El Farol",
            game_type="el_farol",
            num_players=2,
            num_rounds=100,
            num_episodes=1,
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            match_id="m-1",
            game_version="1.0.0",
            num_days=100,
            num_slots=16,
            max_intervals=2,
            max_total_slots=8,
            capacity_ratio=0.6,
            capacity_threshold=9,
            actions_json=actions,
            day_aggregates_json=day_aggregates,
            round_payoffs_json=round_payoffs,
            agents_json=agents,
        )

        # WHEN
        async_session.add(game)
        await async_session.commit()
        await async_session.refresh(game)

        result = await async_session.execute(
            select(GameResult).where(GameResult.id == game.id)
        )
        gr = result.scalar_one()

        # THEN — scalars
        assert gr.match_id == "m-1"
        assert gr.game_version == "1.0.0"
        assert gr.num_days == 100
        assert gr.num_slots == 16
        assert gr.max_intervals == 2
        assert gr.max_total_slots == 8
        assert gr.capacity_ratio == 0.6
        assert gr.capacity_threshold == 9

        # THEN — JSON columns round-trip to the same shape
        assert gr.actions_json == actions
        assert gr.day_aggregates_json == day_aggregates
        assert gr.round_payoffs_json == round_payoffs
        assert gr.agents_json == agents

    @pytest.mark.anyio
    async def test_legacy_row_has_nulls_for_new_columns(
        self,
        async_session: AsyncSession,
    ) -> None:
        """GIVEN a row populated only with legacy fields
        WHEN re-read
        THEN all Phase 7 additive columns are None.
        """
        # GIVEN
        legacy = GameResult(
            game_name="Prisoner's Dilemma",
            game_type="normal_form",
            num_players=2,
            num_rounds=10,
            num_episodes=5,
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            episodes_json=[
                {"episode": 0, "payoffs": {"player_1": 3.0, "player_2": 3.0}}
            ],
            players_json=[{"player_id": "player_1", "strategy": "tit_for_tat"}],
        )

        # WHEN
        async_session.add(legacy)
        await async_session.commit()
        await async_session.refresh(legacy)

        result = await async_session.execute(
            select(GameResult).where(GameResult.id == legacy.id)
        )
        gr = result.scalar_one()

        # THEN — every new column is NULL
        assert gr.match_id is None
        assert gr.game_version is None
        assert gr.num_days is None
        assert gr.num_slots is None
        assert gr.max_intervals is None
        assert gr.max_total_slots is None
        assert gr.capacity_ratio is None
        assert gr.capacity_threshold is None
        assert gr.actions_json is None
        assert gr.day_aggregates_json is None
        assert gr.round_payoffs_json is None
        assert gr.agents_json is None

    @pytest.mark.anyio
    async def test_match_id_index_usable_for_lookup(
        self,
        async_session: AsyncSession,
    ) -> None:
        """GIVEN several rows with distinct match_id values
        WHEN querying by match_id
        THEN exactly the matching row is returned.
        """
        # GIVEN
        rows = []
        for mid in ("m-1", "m-2", "m-3"):
            r = GameResult(
                game_name="El Farol",
                game_type="el_farol",
                num_players=2,
                num_rounds=10,
                num_episodes=1,
                status="completed",
                created_at=datetime.now(),
                completed_at=datetime.now(),
                match_id=mid,
            )
            async_session.add(r)
            rows.append(r)
        await async_session.commit()
        for r in rows:
            await async_session.refresh(r)

        target_id = next(r.id for r in rows if r.match_id == "m-2")

        # WHEN
        result = await async_session.execute(
            select(GameResult).where(GameResult.match_id == "m-2")
        )
        found = result.scalars().all()

        # THEN
        assert len(found) == 1
        assert found[0].id == target_id
        assert found[0].match_id == "m-2"

    @pytest.mark.anyio
    async def test_agents_json_stores_agent_record_shape(
        self,
        async_session: AsyncSession,
    ) -> None:
        """GIVEN agents_json with full AgentRecord-shaped dicts
        WHEN re-read
        THEN list length and every field preserved per agent.
        """
        # GIVEN
        agents = [
            {
                "agent_id": f"p{i}",
                "display_name": f"agent_{i}",
                "user_id": f"u{i}",
                "user_display": f"User {i}",
                "family": "gpt-4",
                "adapter_type": "openai",
                "model_id": "gpt-4o-mini",
                "color": f"#00{i}{i}{i}{i}",
            }
            for i in range(3)
        ]
        game = GameResult(
            game_name="El Farol",
            game_type="el_farol",
            num_players=3,
            num_rounds=50,
            num_episodes=1,
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            agents_json=agents,
        )

        # WHEN
        async_session.add(game)
        await async_session.commit()
        await async_session.refresh(game)

        result = await async_session.execute(
            select(GameResult).where(GameResult.id == game.id)
        )
        gr = result.scalar_one()

        # THEN
        assert gr.agents_json is not None
        assert len(gr.agents_json) == 3
        for i, a in enumerate(gr.agents_json):
            assert a["agent_id"] == f"p{i}"
            assert a["display_name"] == f"agent_{i}"
            assert a["user_id"] == f"u{i}"
            assert a["user_display"] == f"User {i}"
            assert a["family"] == "gpt-4"
            assert a["adapter_type"] == "openai"
            assert a["model_id"] == "gpt-4o-mini"
            assert a["color"] == f"#00{i}{i}{i}{i}"

    @pytest.mark.anyio
    async def test_round_payoffs_json_stores_list_of_dicts(
        self,
        async_session: AsyncSession,
    ) -> None:
        """GIVEN 10 per-round payoff dicts
        WHEN re-read
        THEN length and individual entries are preserved exactly.
        """
        # GIVEN
        round_payoffs = [{"player_0": i * 1.5, "player_1": i * 2.0} for i in range(10)]
        game = GameResult(
            game_name="El Farol",
            game_type="el_farol",
            num_players=2,
            num_rounds=10,
            num_episodes=1,
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            round_payoffs_json=round_payoffs,
        )

        # WHEN
        async_session.add(game)
        await async_session.commit()
        await async_session.refresh(game)

        result = await async_session.execute(
            select(GameResult).where(GameResult.id == game.id)
        )
        gr = result.scalar_one()

        # THEN
        assert gr.round_payoffs_json is not None
        assert len(gr.round_payoffs_json) == 10
        assert gr.round_payoffs_json[5] == {"player_0": 7.5, "player_1": 10.0}

    @pytest.mark.anyio
    async def test_capacity_columns_typing(
        self,
        async_session: AsyncSession,
    ) -> None:
        """GIVEN capacity_ratio=0.6 and capacity_threshold=9
        WHEN re-read
        THEN values preserve Python float / int types and equality.
        """
        # GIVEN
        game = GameResult(
            game_name="El Farol",
            game_type="el_farol",
            num_players=2,
            num_rounds=10,
            num_episodes=1,
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            capacity_ratio=0.6,
            capacity_threshold=9,
        )

        # WHEN
        async_session.add(game)
        await async_session.commit()
        await async_session.refresh(game)

        result = await async_session.execute(
            select(GameResult).where(GameResult.id == game.id)
        )
        gr = result.scalar_one()

        # THEN
        assert isinstance(gr.capacity_ratio, float)
        assert isinstance(gr.capacity_threshold, int)
        assert gr.capacity_ratio == 0.6
        assert gr.capacity_threshold == 9
