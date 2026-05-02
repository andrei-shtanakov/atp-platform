"""Tournament → DashboardPayload reshape for El Farol (LABS-106).

Mirror of ``el_farol_dashboard._reshape`` but reads authoritative
``Round``/``Action``/``Participant`` rows instead of the pre-serialised
``actions_json``/``day_aggregates_json`` on ``GameResult`` (which are
intentionally NULL for tournament-written rows — see
``TournamentService._write_game_result_for_tournament``).

Output model (``DashboardPayload``) is imported from
``el_farol_dashboard`` so any bump of ``SHAPE_VERSION`` is picked up
here automatically.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    Tournament,
)
from atp.dashboard.v2.routes.el_farol_dashboard import (
    DashboardAgent,
    DashboardDecision,
    DashboardPayload,
    DashboardRound,
    SlotPayoff,
)
from atp.dashboard.v2.services.el_farol_constants import CAPACITY_RATIO

_DEFAULT_COLOR = "#6e7781"
# El Farol only — for other games, derive prefix from tournament.game_type
# (see Participant.builtin_strategy format "{game}/{strategy}" in models.py)
_BUILTIN_PREFIX = "el_farol/"
_NUM_SLOTS = 16
# Local alias for backwards-compat with the existing module-private name.
_CAPACITY_RATIO = CAPACITY_RATIO


def _normalised_intervals(
    action_data: dict | None, num_slots: int = _NUM_SLOTS
) -> list[list[int]]:
    """Return up to two clamped ``[start, end]`` pairs from action_data.

    Drops malformed pairs and pairs whose clamped range is empty. Order
    is preserved (the tournament wire shape is already a list of pairs).
    """
    if not action_data:
        return []
    pairs: list[list[int]] = []
    for pair in action_data.get("intervals") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        start, end = pair[0], pair[1]
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        lo = max(0, int(start))
        hi = min(num_slots - 1, int(end))
        if lo <= hi:
            pairs.append([lo, hi])
    return pairs


def _intervals_to_slots(
    action_data: dict | None, num_slots: int = _NUM_SLOTS
) -> list[int]:
    """Expand ``{"intervals": [[start, end], ...]}`` into bounded slot indices."""
    slots: set[int] = set()
    for lo, hi in _normalised_intervals(action_data, num_slots):
        slots.update(range(lo, hi + 1))
    return sorted(slots)


def _intervals_for_dashboard(
    action_data: dict | None, num_slots: int = _NUM_SLOTS
) -> list[list[int]]:
    """Pad normalised intervals to the dashboard's 2-tuple shape.

    DashboardDecision.intervals is documented as a 2-tuple
    (``[[start,end],[start,end]]`` or ``[[start,end],[]]``); the drawer
    and the generated ``make_move(...)`` example read both slots.
    """
    pairs = _normalised_intervals(action_data, num_slots)[:2]
    while len(pairs) < 2:
        pairs.append([])
    return pairs


def _build_agents_from_participants(
    participants: list[Participant],
) -> list[DashboardAgent]:
    """Convert tournament Participant rows to DashboardAgent instances.

    Strips the ``el_farol/`` prefix from ``builtin_strategy`` so the
    profile field reads as the bare strategy name (e.g. ``"traditionalist"``).
    Builtins with no ``user_id`` map to ``user="unknown"``; real users map
    to ``user=str(user_id)``.
    """
    agents: list[DashboardAgent] = []
    for p in participants:
        profile = (p.builtin_strategy or "").removeprefix(_BUILTIN_PREFIX)
        agents.append(
            DashboardAgent(
                id=p.agent_name,
                color=_DEFAULT_COLOR,
                user=str(p.user_id) if p.user_id is not None else "unknown",
                profile=profile,
            )
        )
    return agents


def _build_decision_from_action(
    agent_id: str,
    action: Action,
    slot_attendance: list[int],
    capacity_threshold: int,
) -> DashboardDecision:
    """Build a DashboardDecision for one (agent, action) pair.

    Applies the canonical El Farol payoff rule:
    ``attendance < capacity_threshold`` → +1 (under-cap),
    ``attendance >= capacity_threshold`` → −1 (over-cap).

    Forwards tier-2 telemetry columns (model_id / tokens / cost_usd /
    decide_ms) verbatim so the drawer's DEBUG · OBSERVABILITY panel
    surfaces real values when ``submit_action`` captured them — agent
    self-reported via ``ActionTelemetry`` for tokens/cost/model_id, with
    a server-side fallback for decide_ms (see service.submit_action).
    """
    action_data = action.action_data or {}
    picks = _intervals_to_slots(action_data)
    slot_payoffs = [
        SlotPayoff(
            slot=slot,
            attendance=(
                slot_attendance[slot] if 0 <= slot < len(slot_attendance) else 0
            ),
            payoff=(
                1
                if (slot_attendance[slot] if 0 <= slot < len(slot_attendance) else 0)
                < capacity_threshold
                else -1
            ),
        )
        for slot in picks
    ]
    num_over = sum(1 for sp in slot_payoffs if sp.payoff == -1)
    num_under = sum(1 for sp in slot_payoffs if sp.payoff == 1)
    return DashboardDecision(
        agent=agent_id,
        intervals=_intervals_for_dashboard(action_data),
        picks=picks,
        numVisits=len(picks),
        intent="",
        slotPayoffs=slot_payoffs,
        intervalPayoffs=[],
        payoff=float(action.payoff) if action.payoff is not None else 0.0,
        numOver=num_over,
        numUnder=num_under,
        model_id=action.model_id,
        tokens_in=action.tokens_in,
        tokens_out=action.tokens_out,
        cost_usd=action.cost_usd,
        decide_ms=action.decide_ms,
    )


def _build_rounds_from_actions(
    actions_by_round: dict[int, list[tuple[int, Action]]],
    agents: list[DashboardAgent],
    num_slots: int,
    capacity_threshold: int,
) -> list[DashboardRound]:
    """Convert per-round action pairs into DashboardRound entries.

    Each pair is ``(agent_roster_index, Action)``.  Slot attendance and
    over-cap counts are computed first so every per-decision payoff is
    consistent with the canonical rule.  The full ``Action`` row is
    threaded through so the decision projection can read tier-2
    telemetry columns (model_id, tokens_in/out, cost_usd, decide_ms).
    """
    rounds: list[DashboardRound] = []
    for round_number in sorted(actions_by_round):
        pairs = actions_by_round[round_number]
        slot_attendance = [0] * num_slots
        for _, action in pairs:
            for s in _intervals_to_slots(action.action_data, num_slots):
                slot_attendance[s] += 1
        over_slots = sum(1 for c in slot_attendance if c >= capacity_threshold)
        decisions = [
            _build_decision_from_action(
                agent_id=agents[agent_idx].id,
                action=action,
                slot_attendance=slot_attendance,
                capacity_threshold=capacity_threshold,
            )
            for agent_idx, action in pairs
        ]
        rounds.append(
            DashboardRound(
                round=round_number,
                slotAttendance=slot_attendance,
                decisions=decisions,
                overSlots=over_slots,
            )
        )
    return rounds


async def _reshape_from_tournament(
    tournament_id: int,
    session: AsyncSession,
    match_id: str | None = None,
) -> DashboardPayload:
    """Project a tournament's ORM graph into the El Farol DashboardPayload.

    Reads the authoritative ``Tournament`` / ``Participant`` / ``Round`` /
    ``Action`` rows (no dependence on the NULL-on-tournament
    ``actions_json`` / ``day_aggregates_json`` columns of ``GameResult``).
    Pending or otherwise non-completed rounds are skipped so an in-flight
    tournament still renders the rounds resolved so far.

    ``match_id`` should be the ``GameResult.match_id`` of the row being
    rendered.  The dashboard JS keys per-match localStorage off
    ``window.__ATP_MATCH__.match_id``, so the payload's match id has to
    match the ``/ui/matches/{id}`` URL — otherwise the scrubber and
    pinned-cards state get split across surrogate ids.  Defaults to a
    surrogate when no match row is available (e.g. unit tests).
    """
    tournament = await session.get(Tournament, tournament_id)
    if tournament is None:
        raise LookupError(f"tournament {tournament_id} not found")

    part_stmt = (
        select(Participant)
        .where(Participant.tournament_id == tournament_id)
        .order_by(Participant.id)
    )
    participants = list((await session.execute(part_stmt)).scalars().all())
    agents = _build_agents_from_participants(participants)
    pidx_by_pid = {p.id: i for i, p in enumerate(participants)}

    round_stmt = (
        select(Round)
        .where(Round.tournament_id == tournament_id)
        .where(Round.status == "completed")
        .options(selectinload(Round.actions))
        .order_by(Round.round_number)
    )
    rounds_orm = list((await session.execute(round_stmt)).scalars().all())

    actions_by_round: dict[int, list[tuple[int, Action]]] = {}
    for r in rounds_orm:
        sorted_actions = sorted(
            r.actions, key=lambda a: pidx_by_pid.get(a.participant_id, 10_000)
        )
        actions_by_round[r.round_number] = [
            (pidx_by_pid[a.participant_id], a)
            for a in sorted_actions
            if a.participant_id in pidx_by_pid
        ]

    capacity_threshold = max(1, int(_CAPACITY_RATIO * tournament.num_players))

    data_rounds = _build_rounds_from_actions(
        actions_by_round=actions_by_round,
        agents=agents,
        num_slots=_NUM_SLOTS,
        capacity_threshold=capacity_threshold,
    )

    return DashboardPayload(
        match_id=match_id or f"tournament-{tournament_id}",
        game_version=None,
        AGENTS=agents,
        NUM_SLOTS=_NUM_SLOTS,
        MAX_TOTAL=_NUM_SLOTS,
        MAX_INTERVALS=0,
        CAPACITY=capacity_threshold,
        CAPACITY_RATIO=_CAPACITY_RATIO,
        # NUM_DAYS must equal len(DATA) — the dashboard JS indexes
        # DATA[d] for every d < NUM_DAYS, so any padding from
        # tournament.total_rounds would crash the scrubber for
        # cancelled / in-flight tournaments.
        NUM_DAYS=len(data_rounds),
        DATA=data_rounds,
    )
