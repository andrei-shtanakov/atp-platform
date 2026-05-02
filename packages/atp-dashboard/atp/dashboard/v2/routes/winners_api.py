"""JSON API for the El Farol leaderboard.

Mounted under ``/api`` (via ``v2.routes.__init__``), so the absolute
URL is ``/api/public/leaderboard/el-farol``. The ``/public/leaderboard``
prefix lines up with the existing ``public_leaderboard.py`` convention
for anonymous read endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Response

from atp.dashboard.query_cache import QueryCache
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.services import winners as winners_service
from atp.dashboard.v2.services.winners import (
    LeaderboardPayload,
    get_hof_cache,
    utc_now,
)

router = APIRouter(prefix="/public/leaderboard", tags=["winners-api"])

# 60 s parity with QueryCache TTL — see services/winners.py for the
# rationale on keeping per-tournament and HoF TTLs aligned.
_CACHE_CONTROL = "public, s-maxage=60"


@router.get("/el-farol", response_model=LeaderboardPayload)
async def get_hall_of_fame_json(
    session: DBSession,
    response: Response,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> LeaderboardPayload:
    """Return the El Farol Hall of Fame as JSON.

    Strict bounds: ``limit`` must be 1..200, ``offset`` >= 0; both are
    422 on violation (no silent clamp).
    """
    cache = get_hof_cache()
    key = QueryCache._make_key("hall_of_fame", limit=limit, offset=offset)
    cached = cache.get(key)
    if cached is not None:
        total, entries = cached
    else:
        # Reference via module attribute so tests can monkeypatch
        # ``winners_service._hall_of_fame_query`` and have the route
        # pick up the patched callable.
        total, entries = await winners_service._hall_of_fame_query(
            session, limit=limit, offset=offset
        )
        cache.put(key, (total, entries))

    response.headers["Cache-Control"] = _CACHE_CONTROL
    return LeaderboardPayload(
        generated_at=utc_now(),
        total=total,
        limit=limit,
        offset=offset,
        entries=entries,
    )
