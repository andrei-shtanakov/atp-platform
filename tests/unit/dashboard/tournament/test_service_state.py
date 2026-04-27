"""Tests for TournamentService.get_state_for."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus


@pytest.mark.anyio
async def test_get_state_for_round_1_no_history(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    tournament, _ = await svc.create_tournament(
        creator=admin_user,
        name="t",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(tournament.id, alice, "alice")
    await svc.join(tournament.id, bob, "bob")

    state = await svc.get_state_for(tournament.id, alice)

    assert state.tournament_id == tournament.id
    assert state.round_number == 1
    assert state.game_type == "prisoners_dilemma"
    assert state.your_history == []
    assert state.opponent_history == []
    assert state.your_cumulative_score == 0.0
    assert state.opponent_cumulative_score == 0.0
    assert state.action_schema["options"] == ["cooperate", "defect"]
    assert state.your_turn is True
    assert state.total_rounds == 3


@pytest.mark.anyio
async def test_state_el_farol_has_pending_submission(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    state = await svc.get_state_for(t.id, alice)
    assert state.game_type == "el_farol"
    assert state.pending_submission is True
    assert hasattr(state, "attendance_by_round")
    assert hasattr(state, "capacity_threshold")


@pytest.mark.anyio
async def test_state_el_farol_pending_flips_after_submit(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    await svc.submit_action(t.id, alice, action={"intervals": [[0, 0]]})

    state = await svc.get_state_for(t.id, alice)
    assert state.pending_submission is False


@pytest.mark.anyio
async def test_state_pd_has_your_turn_still(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="pd",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    state = await svc.get_state_for(t.id, alice)
    assert state.game_type == "prisoners_dilemma"
    assert isinstance(state.your_turn, bool)


@pytest.mark.anyio
async def test_state_after_tournament_completes_flags_are_false(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """After the final round resolves, your_turn must NOT be True."""
    from atp.dashboard.tournament.service import TournamentService

    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="pd-done",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    await svc.submit_action(t.id, alice, action={"choice": "cooperate"})
    await svc.submit_action(t.id, bob, action={"choice": "defect"})

    state = await svc.get_state_for(t.id, alice)
    assert state.game_type == "prisoners_dilemma"
    assert state.your_turn is False  # tournament over — nothing to submit


# ---------------------------------------------------------------------------
# Wire-layer action_schema override for el_farol
# ---------------------------------------------------------------------------
#
# The El Farol game-env layer still advertises the canonical slot-list
# schema (correct for that layer — its ``validate_action`` accepts
# ``{"slots": [...]}``), but the tournament wire contract switched to
# intervals. ``service.get_state_for`` must replace the advertised schema
# with the intervals shape so clients building submissions from
# ``state.action_schema`` produce payloads that pass the ``ElFarolAction``
# validator. Other game types must be left untouched.


@pytest.mark.anyio
async def test_get_state_for_el_farol_advertises_intervals_action_schema(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """El Farol round state must advertise the intervals wire contract,
    NOT the game-env's slot-list schema. Without this override, clients
    that build submissions from ``state.action_schema`` would generate
    ``{"slots": [...]}`` payloads that fail at the ``extra='forbid'``
    boundary on ``ElFarolAction``.
    """
    from atp.dashboard.tournament.service import TournamentService

    # GIVEN an active el_farol tournament with both players joined
    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef-schema",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    # WHEN we fetch the per-player state
    state = await svc.get_state_for(t.id, alice)

    # THEN action_schema is the intervals contract (overridden at the wire boundary)
    assert state.game_type == "el_farol"
    schema = state.action_schema
    assert schema["type"] == "list[list[int]]"
    assert schema["max_intervals"] == 2
    assert schema["max_slots_total"] == 8
    assert isinstance(schema["constraints"], list)
    # Non-adjacency is the rule that historically tripped slot-list clients;
    # pin its presence so future schema edits can't quietly drop it.
    assert any("non-adjacent" in c for c in schema["constraints"])
    # The slot-list shape's marker key MUST NOT leak through the override.
    assert "max_length" not in schema
    assert "unique" not in schema


@pytest.mark.anyio
async def test_get_state_for_el_farol_value_range_reflects_num_slots(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """``value_range`` must be derived from the live ``num_slots`` field
    on the formatted state — not hardcoded — so tournaments with different
    bar widths advertise the right inclusive upper bound.
    """
    from atp.dashboard.tournament.service import TournamentService

    # GIVEN an active el_farol tournament (default num_slots=16)
    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="ef-range",
        game_type="el_farol",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    # WHEN we fetch the state
    state = await svc.get_state_for(t.id, alice)

    # THEN value_range matches [0, num_slots-1] from the same state object
    assert state.action_schema["value_range"] == [0, state.num_slots - 1]


@pytest.mark.anyio
async def test_get_state_for_prisoners_dilemma_action_schema_unchanged(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """The wire-layer override is targeted at el_farol ONLY. Other game
    types must continue to surface whatever the game-env layer produces —
    no intervals-shape leakage.
    """
    from atp.dashboard.tournament.service import TournamentService

    # GIVEN an active prisoners_dilemma tournament
    svc = TournamentService(session, event_bus)
    t, _ = await svc.create_tournament(
        creator=admin_user,
        name="pd-schema",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    # WHEN we fetch the state
    state = await svc.get_state_for(t.id, alice)

    # THEN action_schema is the PD discrete-choice schema, NOT the el_farol shape
    schema = state.action_schema
    assert "max_intervals" not in schema
    assert "max_slots_total" not in schema
    assert schema.get("type") != "list[list[int]]"
    # Sanity: the original PD-shaped schema is still being passed through.
    assert "options" in schema
    assert sorted(schema["options"]) == ["cooperate", "defect"]
