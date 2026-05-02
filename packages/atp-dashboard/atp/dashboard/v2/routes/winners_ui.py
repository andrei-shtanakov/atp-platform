"""HTML routes for the El Farol winners surfaces."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from atp.dashboard.models import DEFAULT_TENANT_ID
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.rate_limit import limiter
from atp.dashboard.v2.services import winners as winners_service
from atp.dashboard.v2.services.winners import (
    CAPACITY_RATIO,
    get_hof_cache,
    get_winners_cache,
    hof_cache_key,
    winners_cache_key,
)

router = APIRouter(tags=["winners-ui"])

_CACHE_CONTROL = "public, s-maxage=60"
_NOT_FOUND = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


def _capacity_for(num_players: int) -> int:
    return max(1, int(CAPACITY_RATIO * num_players))


def _format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


@router.get("/ui/tournaments/{tournament_id}/winners", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def get_winners_html(
    request: Request,
    tournament_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Render the winners poster for a completed public El Farol tournament."""
    t = await session.get(Tournament, tournament_id)
    if t is None:
        raise _NOT_FOUND
    if t.tenant_id != DEFAULT_TENANT_ID:
        raise _NOT_FOUND
    if t.game_type != "el_farol":
        raise _NOT_FOUND
    if t.status != TournamentStatus.COMPLETED:
        raise _NOT_FOUND
    if t.join_token is not None:
        raise _NOT_FOUND

    cache = get_winners_cache()
    key = winners_cache_key(tournament_id)
    entries = cache.get(key)
    if entries is None:
        # Module-attribute reference so tests can monkeypatch.
        entries = await winners_service._winners_query(session, tournament_id)
        cache.put(key, entries)

    name = (t.config or {}).get("name") or f"Tournament #{t.id}"
    duration: str | None = None
    if t.starts_at and t.ends_at:
        duration = _format_duration(int((t.ends_at - t.starts_at).total_seconds()))

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="ui/winners_tournament.html",
        context={
            "active_page": "tournaments",
            "tournament": t,
            "tournament_name": name,
            "header": {
                "players": t.num_players,
                "days": t.total_rounds,
                "capacity": _capacity_for(t.num_players),
                "duration": duration,
            },
            "entries": entries,
        },
        headers={"Cache-Control": _CACHE_CONTROL},
    )


@router.get("/ui/leaderboard/el-farol", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def get_hall_of_fame_html(
    request: Request,
    session: DBSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> HTMLResponse:
    """Render the public El Farol Hall of Fame leaderboard page."""
    cache = get_hof_cache()
    key = hof_cache_key(limit=limit, offset=offset)
    cached = cache.get(key)
    if cached is not None:
        total, entries = cached
    else:
        # Module-attribute reference so tests can monkeypatch.
        total, entries = await winners_service._hall_of_fame_query(
            session, limit=limit, offset=offset
        )
        cache.put(key, (total, entries))

    page = (offset // limit) + 1
    total_pages = max(1, (total + limit - 1) // limit)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="ui/winners_hall_of_fame.html",
        context={
            "active_page": "el_farol_hall_of_fame",
            "entries": entries,
            "total": total,
            "limit": limit,
            "offset": offset,
            "page": page,
            "total_pages": total_pages,
        },
        headers={"Cache-Control": _CACHE_CONTROL},
    )
