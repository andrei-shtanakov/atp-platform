"""Player-private RoundState — what one participant sees on their turn.

The shape is intentionally generic enough to fit any 2-player game in
the v1 slice (PD now, Stag Hunt and Battle of Sexes nearly free in
Plan 2). For N-player games (El Farol etc.), the formatter on the
game class returns the same RoundState shape with N-aware fields
populated; this stays out of scope for the v1 slice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoundState:
    tournament_id: int
    round_number: int
    game_type: str
    your_history: list[str]
    opponent_history: list[str]
    your_cumulative_score: float
    opponent_cumulative_score: float
    action_schema: dict[str, Any]
    your_turn: bool
    total_rounds: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
