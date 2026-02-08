"""Protocol mapping between game-environments and ATP."""

from atp_games.mapping.action_mapper import ActionMapper
from atp_games.mapping.observation_mapper import ObservationMapper

__all__ = ["ActionMapper", "ObservationMapper"]
