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

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.tournament.models import Participant
from atp.dashboard.v2.routes.el_farol_dashboard import (
    DashboardAgent,
    DashboardDecision,
    DashboardPayload,
    DashboardRound,
    SlotPayoff,
)

_DEFAULT_COLOR = "#6e7781"
# El Farol only — for other games, derive prefix from tournament.game_type
# (see Participant.builtin_strategy format "{game}/{strategy}" in models.py)
_BUILTIN_PREFIX = "el_farol/"


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
    action_data: dict[str, Any],
    payoff: float | None,
    slot_attendance: list[int],
    capacity_threshold: int,
) -> DashboardDecision:
    """Build a DashboardDecision for one (agent, action) pair.

    Applies the canonical El Farol payoff rule:
    ``attendance < capacity_threshold`` → +1 (under-cap),
    ``attendance >= capacity_threshold`` → −1 (over-cap).
    """
    picks = [int(s) for s in (action_data or {}).get("slots") or []]
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
        intervals=[[], []],
        picks=picks,
        numVisits=len(picks),
        intent="",
        slotPayoffs=slot_payoffs,
        intervalPayoffs=[],
        payoff=float(payoff) if payoff is not None else 0.0,
        numOver=num_over,
        numUnder=num_under,
    )


def _build_rounds_from_actions(
    actions_by_round: dict[int, list[tuple[int, dict[str, Any], float | None]]],
    agents: list[DashboardAgent],
    num_slots: int,
    capacity_threshold: int,
) -> list[DashboardRound]:
    """Convert per-round action triples into DashboardRound entries.

    Each triple is ``(agent_roster_index, action_data, payoff)``.  Slot
    attendance and over-cap counts are computed first so every
    per-decision payoff is consistent with the canonical rule.
    """
    rounds: list[DashboardRound] = []
    for round_number in sorted(actions_by_round):
        triples = actions_by_round[round_number]
        slot_attendance = [0] * num_slots
        for _, action_data, _ in triples:
            for s in (action_data or {}).get("slots") or []:
                if 0 <= int(s) < num_slots:
                    slot_attendance[int(s)] += 1
        over_slots = sum(1 for c in slot_attendance if c >= capacity_threshold)
        decisions = [
            _build_decision_from_action(
                agent_id=agents[agent_idx].id,
                action_data=action_data,
                payoff=payoff,
                slot_attendance=slot_attendance,
                capacity_threshold=capacity_threshold,
            )
            for agent_idx, action_data, payoff in triples
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
) -> DashboardPayload:
    raise NotImplementedError("Task 1 stub; filled in Task 4")
