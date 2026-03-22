"""Game evaluation routes.

This module provides endpoints for viewing game-theoretic
evaluation results, tournament standings, and cross-play matrices.

Permissions:
    - GET /games/: RESULTS_READ
    - GET /games/{id}: RESULTS_READ
    - GET /games/{id}/export/csv: RESULTS_READ
    - GET /games/{id}/export/json: RESULTS_READ
    - GET /tournaments/: RESULTS_READ
    - GET /tournaments/{id}: RESULTS_READ
    - GET /crossplay/{id}: RESULTS_READ
"""

import csv
import html
import io
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from atp.dashboard.models import GameResult, TournamentResult
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    GamePlayerResponse,
    GameResultDetail,
    GameResultList,
    GameResultSummary,
    MatchupResponse,
    TournamentResultDetail,
    TournamentResultList,
    TournamentResultSummary,
    TournamentStandingResponse,
)
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(tags=["games"])


# ==================== Game Results ====================


@router.get("/games", response_model=GameResultList)
async def list_game_results(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    game_name: str | None = None,
    game_type: str | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> GameResultList:
    """List game evaluation results.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        game_name: Filter by game name.
        game_type: Filter by game type.
        limit: Maximum number of results.
        offset: Offset for pagination.

    Returns:
        Paginated list of game results.
    """
    stmt = select(GameResult)
    if game_name:
        stmt = stmt.where(GameResult.game_name == game_name)
    if game_type:
        stmt = stmt.where(GameResult.game_type == game_type)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    stmt = stmt.order_by(GameResult.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    game_results = result.scalars().all()

    return GameResultList(
        total=total,
        items=[GameResultSummary.model_validate(r) for r in game_results],
        limit=limit,
        offset=offset,
    )


@router.get("/games/{game_id}", response_model=GameResultDetail)
async def get_game_result(
    session: DBSession,
    game_id: int,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
) -> GameResultDetail:
    """Get detailed game evaluation result.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        game_id: Game result ID.

    Returns:
        Detailed game result with players, payoffs, and metrics.

    Raises:
        HTTPException: If game result not found.
    """
    stmt = select(GameResult).where(GameResult.id == game_id)
    result = await session.execute(stmt)
    game_result = result.scalar_one_or_none()

    if game_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game result {game_id} not found",
        )

    players = [GamePlayerResponse(**p) for p in (game_result.players_json or [])]

    return GameResultDetail(
        id=game_result.id,
        game_name=game_result.game_name,
        game_type=game_result.game_type,
        num_players=game_result.num_players,
        num_rounds=game_result.num_rounds,
        num_episodes=game_result.num_episodes,
        status=game_result.status,
        created_at=game_result.created_at,
        completed_at=game_result.completed_at,
        players=players,
        payoff_matrix=game_result.payoff_matrix_json,
        strategy_timeline=game_result.strategy_timeline_json,
        cooperation_dynamics=(game_result.cooperation_dynamics_json),
        episodes=game_result.episodes_json,
        metadata=game_result.metadata_json or {},
    )


# ==================== Game Export ====================


@router.get("/games/{game_id}/export/csv")
async def export_game_csv(
    session: DBSession,
    game_id: int,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
) -> StreamingResponse:
    """Export game results as CSV.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        game_id: Game result ID.

    Returns:
        CSV file as streaming response.

    Raises:
        HTTPException: If game result not found.
    """
    stmt = select(GameResult).where(GameResult.id == game_id)
    result = await session.execute(stmt)
    game_result = result.scalar_one_or_none()

    if game_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game result {game_id} not found",
        )

    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(
        [
            "episode",
            "player_id",
            "payoff",
            "strategy",
            "cooperation_rate",
        ]
    )

    player_info: dict[str, dict[str, Any]] = {}
    for p in game_result.players_json or []:
        # Sanitize player data to prevent CSV injection
        player_id = html.escape(str(p.get("player_id", "")))
        player_info[player_id] = {
            k: html.escape(str(v)) if isinstance(v, str) else v for k, v in p.items()
        }

    for ep in game_result.episodes_json or []:
        payoffs = ep.get("payoffs", {})
        for player_id, payoff in payoffs.items():
            player_id = html.escape(str(player_id))
            info = player_info.get(player_id, {})
            writer.writerow(
                [
                    ep.get("episode", ""),
                    player_id,
                    round(payoff, 4),
                    info.get("strategy", ""),
                    (
                        round(float(str(info.get("cooperation_rate", 0))), 4)
                        if info.get("cooperation_rate") is not None
                        else ""
                    ),
                ]
            )

    buf.seek(0)
    # Sanitize filename to prevent path traversal
    safe_game_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in game_result.game_name
    )
    filename = f"game_{safe_game_name}_{game_id}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": (f'attachment; filename="{filename}"')},
    )


@router.get("/games/{game_id}/export/json")
async def export_game_json(
    session: DBSession,
    game_id: int,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
) -> StreamingResponse:
    """Export game results as JSON for Jupyter analysis.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        game_id: Game result ID.

    Returns:
        JSON file as streaming response.

    Raises:
        HTTPException: If game result not found.
    """
    stmt = select(GameResult).where(GameResult.id == game_id)
    result = await session.execute(stmt)
    game_result = result.scalar_one_or_none()

    if game_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game result {game_id} not found",
        )

    data = {
        "game_name": game_result.game_name,
        "game_type": game_result.game_type,
        "num_players": game_result.num_players,
        "num_rounds": game_result.num_rounds,
        "num_episodes": game_result.num_episodes,
        "players": game_result.players_json or [],
        "payoff_matrix": game_result.payoff_matrix_json,
        "strategy_timeline": (game_result.strategy_timeline_json),
        "cooperation_dynamics": (game_result.cooperation_dynamics_json),
        "episodes": game_result.episodes_json or [],
        "metadata": game_result.metadata_json or {},
    }

    json_str = json.dumps(data, indent=2, default=str)
    # Sanitize filename to prevent path traversal
    safe_game_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in game_result.game_name
    )
    filename = f"game_{safe_game_name}_{game_id}.json"
    return StreamingResponse(
        iter([json_str]),
        media_type="application/json",
        headers={"Content-Disposition": (f'attachment; filename="{filename}"')},
    )


# ==================== Tournament Results ====================


@router.get("/tournaments", response_model=TournamentResultList)
async def list_tournament_results(
    session: DBSession,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
    game_name: str | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> TournamentResultList:
    """List tournament results.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        game_name: Filter by game name.
        limit: Maximum number of results.
        offset: Offset for pagination.

    Returns:
        Paginated list of tournament results.
    """
    stmt = select(TournamentResult)
    if game_name:
        stmt = stmt.where(TournamentResult.game_name == game_name)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    stmt = stmt.order_by(TournamentResult.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    tournaments = result.scalars().all()

    return TournamentResultList(
        total=total,
        items=[TournamentResultSummary.model_validate(t) for t in tournaments],
        limit=limit,
        offset=offset,
    )


@router.get(
    "/tournaments/{tournament_id}",
    response_model=TournamentResultDetail,
)
async def get_tournament_result(
    session: DBSession,
    tournament_id: int,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
) -> TournamentResultDetail:
    """Get detailed tournament result.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        tournament_id: Tournament result ID.

    Returns:
        Detailed tournament result with standings and matchups.

    Raises:
        HTTPException: If tournament not found.
    """
    stmt = select(TournamentResult).where(TournamentResult.id == tournament_id)
    result = await session.execute(stmt)
    tournament = result.scalar_one_or_none()

    if tournament is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Tournament result {tournament_id} not found"),
        )

    standings = [
        TournamentStandingResponse(**s) for s in (tournament.standings_json or [])
    ]
    matchups = [MatchupResponse(**m) for m in (tournament.matchups_json or [])]

    return TournamentResultDetail(
        id=tournament.id,
        name=tournament.name,
        game_name=tournament.game_name,
        tournament_type=tournament.tournament_type,
        num_agents=tournament.num_agents,
        episodes_per_matchup=tournament.episodes_per_matchup,
        status=tournament.status,
        created_at=tournament.created_at,
        completed_at=tournament.completed_at,
        standings=standings,
        matchups=matchups,
        cross_play_matrix=tournament.cross_play_matrix_json,
    )


# ==================== Cross-Play ====================


@router.get("/crossplay/{tournament_id}")
async def get_crossplay_matrix(
    session: DBSession,
    tournament_id: int,
    _: Annotated[None, Depends(require_permission(Permission.RESULTS_READ))],
) -> dict[str, Any]:
    """Get cross-play matrix for a tournament.

    Requires RESULTS_READ permission.

    Args:
        session: Database session.
        tournament_id: Tournament result ID.

    Returns:
        Cross-play matrix with agent payoffs.

    Raises:
        HTTPException: If tournament not found.
    """
    stmt = select(TournamentResult).where(TournamentResult.id == tournament_id)
    result = await session.execute(stmt)
    tournament = result.scalar_one_or_none()

    if tournament is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Tournament result {tournament_id} not found"),
        )

    return {
        "tournament_id": tournament.id,
        "tournament_name": tournament.name,
        "game_name": tournament.game_name,
        "cross_play_matrix": (tournament.cross_play_matrix_json or {}),
    }
