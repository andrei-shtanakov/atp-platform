"""Shared models and query helpers for the El Farol winners dashboard.

The two query helpers (``_winners_query`` and ``_hall_of_fame_query``)
are the single source of truth for both the JSON API
(``routes/winners_api.py``) and the HTML pages
(``routes/winners_ui.py``). Cache keys are built from query parameters
via the QueryCache key builder.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import Agent, User
from atp.dashboard.query_cache import QueryCache
from atp.dashboard.tournament.models import Action, Participant
from atp.dashboard.v2.services.el_farol_constants import (
    CAPACITY_RATIO as _CAPACITY_RATIO,
)

# Pin the schema version on the JSON contract. Bump on any
# breaking change to ``LeaderboardPayload``.
SCHEMA_VERSION = 1


class WinnerEntry(BaseModel):
    """One row on the per-tournament winners poster.

    ``agent_name`` is the historical display name from
    ``Participant.agent_name``; ``agent_description`` and
    ``owner_username`` are read from the live ``Agent`` and ``User`` rows
    (i.e. "as of now"). Builtin participants render with
    ``owner_username='system'`` and a fixed description.
    """

    rank: int
    agent_name: str
    agent_description: str | None
    owner_username: str
    score: float | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    # "mixed" if the participant emitted multiple distinct model ids
    model_id: str | None


class HallEntry(BaseModel):
    """One row on the Hall of Fame, keyed by logical agent.

    Logical agent identity = ``(tenant_id, owner_id, agent.name)``.
    Versions of the same agent (different ``Agent.version`` rows under
    the same ``(owner_id, name)``) are aggregated together.
    """

    rank: int
    owner_username: str
    agent_name: str
    agent_description: str | None
    total_score: float
    tournaments_count: int


class LeaderboardPayload(BaseModel):
    """Public JSON contract for the Hall of Fame.

    ``schema_version`` lets external consumers refuse mismatched shapes.
    ``generated_at`` is the server time when the response was assembled
    — useful for debugging caches and for clients that want to display
    "as of HH:MM".
    """

    schema_version: int = Field(default=SCHEMA_VERSION)
    generated_at: datetime
    total: int
    limit: int
    offset: int
    entries: list[HallEntry]


# 60 s TTL is deliberate. Per-tournament rows are immutable post-
# ``completed`` and could safely use a much longer TTL, but keeping
# both caches at 60 s makes invalidation trivial if we ever add a
# mutable post-completion field (e.g. moderator-added annotations).
_WINNERS_CACHE_TTL_S = 60
# Per-tournament winners cache is keyed by tournament_id; cardinality is
# bounded by the number of completed public tournaments.
_WINNERS_CACHE_MAX = 256

# Hall of Fame cache is keyed by (limit, offset) pagination pairs;
# cardinality is bounded by reasonable browsing depth.
_HOF_CACHE_MAX = 64

_winners_cache: QueryCache[list[WinnerEntry]] | None = None
_hof_cache: QueryCache[tuple[int, list[HallEntry]]] | None = None


def get_winners_cache() -> QueryCache[list[WinnerEntry]]:
    """Lazy singleton for the per-tournament winners cache."""
    global _winners_cache
    if _winners_cache is None:
        _winners_cache = QueryCache(
            max_size=_WINNERS_CACHE_MAX,
            ttl_seconds=_WINNERS_CACHE_TTL_S,
        )
    return _winners_cache


def get_hof_cache() -> QueryCache[tuple[int, list[HallEntry]]]:
    """Lazy singleton for the Hall of Fame paginated cache.

    Cache value is ``(total_count, page_entries)`` so a single fetch
    yields both pieces the route needs.
    """
    global _hof_cache
    if _hof_cache is None:
        _hof_cache = QueryCache(
            max_size=_HOF_CACHE_MAX,
            ttl_seconds=_WINNERS_CACHE_TTL_S,
        )
    return _hof_cache


def reset_caches_for_tests() -> None:
    """Drop both caches. Tests call this to keep TTL state isolated."""
    global _winners_cache, _hof_cache
    _winners_cache = None
    _hof_cache = None


# Re-export for ergonomic import in routes.
CAPACITY_RATIO = _CAPACITY_RATIO


async def _winners_query(
    session: AsyncSession, tournament_id: int
) -> list[WinnerEntry]:
    """Aggregate winners for one tournament.

    Pulls per-participant totals together with optional LLM telemetry
    summed across the participant's actions. The display name comes
    from ``Participant.agent_name`` (historical at join time);
    description and owner are looked up live from ``Agent`` / ``User``.
    """
    stmt = (
        select(
            Participant.id.label("participant_id"),
            Participant.agent_name.label("display_name"),
            Participant.agent_id.label("agent_id"),
            Participant.builtin_strategy.label("builtin_strategy"),
            Participant.total_score.label("total_score"),
            Agent.description.label("agent_description"),
            Agent.deleted_at.label("agent_deleted_at"),
            User.username.label("owner_username"),
            func.sum(Action.tokens_in).label("tokens_in"),
            func.sum(Action.tokens_out).label("tokens_out"),
            func.sum(Action.cost_usd).label("cost_usd"),
            func.min(Action.model_id).label("sample_model"),
            func.count(func.distinct(Action.model_id)).label("distinct_models"),
        )
        .join(Agent, Agent.id == Participant.agent_id, isouter=True)
        .join(User, User.id == Agent.owner_id, isouter=True)
        .join(Action, Action.participant_id == Participant.id, isouter=True)
        .where(Participant.tournament_id == tournament_id)
        .group_by(Participant.id, Agent.id, User.id)
        .order_by(
            Participant.total_score.desc().nulls_last(),
            Participant.id.asc(),
        )
    )
    result = await session.execute(stmt)
    rows = result.all()

    # Dense ranking — ties share a rank, the next non-tied score jumps
    # to len(seen_so_far) + 1. Pure post-processing keeps the SQL
    # portable across SQLite (test DB) and Postgres (prod DB).
    entries: list[WinnerEntry] = []
    rank = 0
    prev_score: float | None = None
    seen = 0
    for row in rows:
        seen += 1
        if row.total_score != prev_score:
            rank = seen
            prev_score = row.total_score

        if row.agent_id is None:
            owner_username = "system"
            description = "built-in strategy"
            display_name = row.display_name
        else:
            owner_username = row.owner_username or "—"
            description = row.agent_description
            display_name = row.display_name
            if row.agent_deleted_at is not None:
                display_name = f"{display_name} (archived)"

        if row.distinct_models is None or row.distinct_models == 0:
            model_id: str | None = None
        elif row.distinct_models == 1:
            model_id = row.sample_model
        else:
            model_id = "mixed"

        entries.append(
            WinnerEntry(
                rank=rank,
                agent_name=display_name,
                agent_description=description,
                owner_username=owner_username,
                score=row.total_score,
                tokens_in=row.tokens_in,
                tokens_out=row.tokens_out,
                cost_usd=row.cost_usd,
                model_id=model_id,
            )
        )

    return entries
