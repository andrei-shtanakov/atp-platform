"""LABS-TSA PR-4 — namespaced builtin strategy registry."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.tournament.builtins import (
    BuiltinNotFoundError,
    list_builtins_for_game,
    resolve_builtin,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    """Create a v2 app bound to the shared in-memory test database."""
    app = create_test_app()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


class TestBuiltinRegistry:
    def test_list_el_farol_builtins(self) -> None:
        names = {b.name for b in list_builtins_for_game("el_farol")}
        assert "el_farol/traditionalist" in names
        assert "el_farol/contrarian" in names
        assert "el_farol/gambler" in names

    def test_resolve_returns_strategy_instance(self) -> None:
        strategy = resolve_builtin(
            "el_farol/traditionalist", tournament_id=1, participant_id=1
        )
        assert strategy is not None
        # Strategy base class provides choose_action — smoke only, no call.
        assert hasattr(strategy, "choose_action")

    def test_resolve_seeded_strategy_is_deterministic(self) -> None:
        # Strategies with a ``seed`` kwarg (e.g. Gambler) must
        # produce the same RNG state for the same (tournament,
        # participant) pair across repeated resolves.
        a = resolve_builtin("el_farol/gambler", tournament_id=7, participant_id=3)
        b = resolve_builtin("el_farol/gambler", tournament_id=7, participant_id=3)
        # ._rng is the ``random.Random`` instance seeded in __init__
        assert a._rng.random() == b._rng.random()

    def test_unknown_raises(self) -> None:
        with pytest.raises(BuiltinNotFoundError):
            resolve_builtin("el_farol/nonexistent", tournament_id=1, participant_id=1)

    def test_unnamespaced_raises(self) -> None:
        with pytest.raises(BuiltinNotFoundError):
            # bare name without game/ prefix
            resolve_builtin("traditionalist", tournament_id=1, participant_id=1)


class TestBuiltinsEndpoint:
    @pytest.mark.anyio
    async def test_list_endpoint(self, v2_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/el_farol/builtins")
        assert resp.status_code == 200
        body = resp.json()
        assert body["game_type"] == "el_farol"
        names = {b["name"] for b in body["builtins"]}
        assert "el_farol/traditionalist" in names

    @pytest.mark.anyio
    async def test_unknown_game_returns_empty(self, v2_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/not_a_game/builtins")
        assert resp.status_code == 200
        assert resp.json()["builtins"] == []
