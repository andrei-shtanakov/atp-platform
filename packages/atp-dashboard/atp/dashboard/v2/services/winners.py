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

from atp.dashboard.query_cache import QueryCache
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
