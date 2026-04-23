"""El Farol dashboard fetch adapter (LABS-99).

Serves the `window.ATP` payload the dashboard JS (shipped under
`/static/v2/js/el_farol/`) consumes to render a match. The endpoint
reads the per-match ``GameResult`` row populated by PR #63 Phase 7 and
reshapes it into the nested layout the mockup-derived renderer expects.

Shape contract — see ``docs/mockups/el-farol/cards.html`` (LABS-97).
Any change here is a client-breaking change; bump ``SHAPE_VERSION``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from atp.dashboard.models import GameResult
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.v2.dependencies import DBSession

SHAPE_VERSION = 1

router = APIRouter(prefix="/v1/games", tags=["games", "dashboard"])


# ------------------------------------------------------------------
# Pydantic models — exact shape the mockup JS expects on window.ATP
# ------------------------------------------------------------------


class DashboardAgent(BaseModel):
    """One agent entry in ``window.ATP.AGENTS``."""

    id: str
    color: str = "#6e7781"
    user: str = "unknown"
    # ``profile`` is a mockup-only concept (never persisted) — kept here
    # so the renderer can treat it as opaque metadata; backend emits the
    # agent's family if known, otherwise empty string.
    profile: str = ""


class SlotPayoff(BaseModel):
    slot: int
    attendance: int
    payoff: int


class IntervalPayoff(BaseModel):
    index: int
    interval: list[int]
    payoff: int


class DashboardDecision(BaseModel):
    """One agent's action on one day — ``window.ATP.DATA[d].decisions[i]``."""

    agent: str
    intervals: list[list[int]] = Field(
        default_factory=list,
        description=(
            "2-tuple of intervals: [[start,end],[start,end]] or [[start,end],[]]."
        ),
    )
    picks: list[int] = Field(default_factory=list)
    numVisits: int = 0
    intent: str = ""
    slotPayoffs: list[SlotPayoff] = Field(default_factory=list)
    intervalPayoffs: list[IntervalPayoff] = Field(default_factory=list)
    payoff: float = 0.0
    numOver: int = 0
    numUnder: int = 0
    # Tier-2 OTel fields — all optional, render as "—" on the client.
    tokens_in: int | None = None
    tokens_out: int | None = None
    decide_ms: int | None = None
    cost_usd: float | None = None
    model_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    retry_count: int | None = None
    validation_error: str | None = None


class DashboardRound(BaseModel):
    """One day — ``window.ATP.DATA[d]``."""

    round: int = Field(description="1-indexed day number")
    slotAttendance: list[int]
    decisions: list[DashboardDecision]
    overSlots: int


class DashboardPayload(BaseModel):
    """Full ``window.ATP`` payload for one El Farol match."""

    shape_version: int = SHAPE_VERSION
    match_id: str
    game_version: str | None = None
    AGENTS: list[DashboardAgent]
    NUM_SLOTS: int
    MAX_TOTAL: int
    MAX_INTERVALS: int
    CAPACITY: int
    CAPACITY_RATIO: float | None = None
    NUM_DAYS: int
    WEEK_LEN: int = 10
    DATA: list[DashboardRound]


# ------------------------------------------------------------------
# Reshape helpers
# ------------------------------------------------------------------


def _build_agents(row: GameResult) -> list[DashboardAgent]:
    """Extract agent roster from ``agents_json`` with ``players_json`` fallback."""
    agents_src = row.agents_json or []
    if agents_src:
        return [
            DashboardAgent(
                id=a.get("agent_id") or a.get("id") or "unknown",
                color=a.get("color") or "#6e7781",
                user=a.get("user_id") or "unknown",
                profile=a.get("family") or "",
            )
            for a in agents_src
        ]
    # Legacy fallback — shouldn't normally hit because the 404 path
    # upstream rejects rows without actions_json. Keep as defence.
    players = row.players_json or []
    return [
        DashboardAgent(
            id=p.get("player_id") or p.get("name") or "unknown",
            user="unknown",
        )
        for p in players
    ]


def _build_rounds(
    row: GameResult, agents: list[DashboardAgent]
) -> list[DashboardRound]:
    """Group flat ActionRecord list into per-day rounds.

    ``actions_json`` is a flat ``list[ActionRecord.asdict()]`` ordered
    episode-major / day-major. ``day_aggregates_json`` carries per-day
    slot attendance / over_slots. Cross-indexes both by day.
    """
    actions = row.actions_json or []
    aggregates = {a["day"]: a for a in (row.day_aggregates_json or [])}

    # bucket actions by day, preserving agent order from the roster
    agent_order = {a.id: i for i, a in enumerate(agents)}
    by_day: dict[int, list[dict[str, Any]]] = {}
    for a in actions:
        day = int(a.get("day", 0))
        by_day.setdefault(day, []).append(a)

    rounds: list[DashboardRound] = []
    for day in sorted(by_day):
        day_actions = by_day[day]
        day_actions.sort(key=lambda x: agent_order.get(x.get("agent_id"), 10_000))
        agg = aggregates.get(day, {})
        slot_attendance = list(agg.get("slot_attendance") or [])

        decisions = [
            _build_decision(a, slot_attendance, row.capacity_threshold)
            for a in day_actions
        ]
        rounds.append(
            DashboardRound(
                round=day,
                slotAttendance=slot_attendance,
                decisions=decisions,
                overSlots=int(agg.get("over_slots", 0)),
            )
        )
    return rounds


def _build_decision(
    action: dict[str, Any],
    slot_attendance: list[int],
    capacity_threshold: int | None,
) -> DashboardDecision:
    """Reshape one ActionRecord dict into a DashboardDecision.

    ``slotPayoffs`` are derived from ``slot_attendance`` and the match's
    ``capacity_threshold`` using the canonical El Farol rule: a slot
    pays +1 when ``attendance < capacity_threshold`` (happy), -1
    otherwise (crowded). Matches ``over_slots`` accounting in
    ``atp/cli/commands/game.py::_compute_day_aggregates``. When
    ``capacity_threshold`` is unknown (None), slot payoffs are omitted
    — the client degrades to aggregate counters.
    """
    intervals_raw = action.get("intervals") or {}
    # ActionRecord serialises IntervalPair as {"first": [a,b] or [], "second": ...}
    if isinstance(intervals_raw, dict):
        first = list(intervals_raw.get("first") or [])
        second = list(intervals_raw.get("second") or [])
        intervals_out = [first, second]
    elif isinstance(intervals_raw, list):
        # Already a list-of-lists — defensive passthrough
        intervals_out = [list(iv) for iv in intervals_raw]
    else:
        intervals_out = [[], []]

    picks = list(action.get("picks") or [])
    slot_payoffs: list[SlotPayoff] = []
    if capacity_threshold is not None:
        for slot in picks:
            att = slot_attendance[slot] if 0 <= slot < len(slot_attendance) else 0
            p = 1 if att < capacity_threshold else -1
            slot_payoffs.append(SlotPayoff(slot=slot, attendance=att, payoff=p))

    return DashboardDecision(
        agent=action.get("agent_id") or "unknown",
        intervals=intervals_out,
        picks=picks,
        numVisits=int(action.get("num_visits", 0)),
        intent=action.get("intent") or "",
        slotPayoffs=slot_payoffs,
        intervalPayoffs=[],  # derived client-side if ever needed
        payoff=float(action.get("payoff", 0.0)),
        numOver=int(action.get("num_over", 0)),
        numUnder=int(action.get("num_under", 0)),
        tokens_in=action.get("tokens_in"),
        tokens_out=action.get("tokens_out"),
        decide_ms=action.get("decide_ms"),
        cost_usd=action.get("cost_usd"),
        model_id=action.get("model_id"),
        trace_id=action.get("trace_id"),
        span_id=action.get("span_id"),
        retry_count=action.get("retry_count"),
        validation_error=action.get("validation_error"),
    )


def _reshape(row: GameResult) -> DashboardPayload:
    """Transform a GameResult row into the DashboardPayload payload."""
    agents = _build_agents(row)
    rounds = _build_rounds(row, agents)
    return DashboardPayload(
        match_id=row.match_id or str(row.id),
        game_version=row.game_version,
        AGENTS=agents,
        NUM_SLOTS=row.num_slots or 16,
        MAX_TOTAL=row.max_total_slots or 8,
        MAX_INTERVALS=row.max_intervals or 2,
        CAPACITY=row.capacity_threshold or 0,
        CAPACITY_RATIO=row.capacity_ratio,
        NUM_DAYS=row.num_days or len(rounds),
        DATA=rounds,
    )


# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------


async def _load_match(session: Any, match_id: str) -> GameResult:
    """Fetch a GameResult by match_id column or fallback to numeric id."""
    stmt = select(GameResult).where(GameResult.match_id == match_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None and match_id.isdigit():
        stmt = select(GameResult).where(GameResult.id == int(match_id))
        row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"match {match_id!r} not found",
        )
    return row


@router.get(
    "/{match_id}/dashboard",
    response_model=DashboardPayload,
    summary="El Farol dashboard payload",
    responses={
        404: {"description": "match not found or predates the Phase 7 schema"},
    },
)
async def get_el_farol_dashboard(
    session: DBSession,
    match_id: str,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
) -> DashboardPayload:
    """Return the ``window.ATP`` payload for one El Farol match.

    The Cards dashboard (LABS-102) loads this endpoint once at page
    init, assigns the response to ``window.ATP``, and renders client-
    side — no subsequent polling, the match is immutable once
    completed.

    Legacy rows (pre-PR #63 Phase 7) lack ``actions_json`` and
    ``day_aggregates_json``; the endpoint 404s with a pointer to the
    migration story instead of serving a half-populated payload that
    would silently break the renderer.
    """
    row = await _load_match(session, match_id)

    if not row.actions_json or not row.day_aggregates_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"match {match_id!r} predates the El Farol dashboard schema "
                "(PR #63, Phase 7). Actions and day aggregates are not "
                "available for this row."
            ),
        )

    return _reshape(row)


# Literal type so the renderer can assert at runtime we're on the
# shape it was written against — client imports SHAPE_VERSION and
# refuses to mount if it diverges.
ShapeVersion = Literal[1]
