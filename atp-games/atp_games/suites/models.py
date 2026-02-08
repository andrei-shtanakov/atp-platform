"""Pydantic models for game suite configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GameMetricConfig(BaseModel):
    """Configuration for a single evaluation metric."""

    type: str
    weight: float = Field(default=1.0, ge=0.0)
    config: dict[str, Any] = Field(default_factory=dict)


class GameEvaluationConfig(BaseModel):
    """Evaluation configuration for a game suite."""

    episodes: int = Field(default=50, ge=1)
    metrics: list[GameMetricConfig] = Field(default_factory=list)
    thresholds: dict[str, dict[str, float]] = Field(
        default_factory=dict,
    )


class GameReportingConfig(BaseModel):
    """Reporting configuration for a game suite."""

    include_strategy_profile: bool = True
    include_payoff_matrix: bool = True
    include_round_by_round: bool = False
    export_formats: list[str] = Field(default_factory=list)


class GameAgentConfig(BaseModel):
    """Configuration for a game agent."""

    name: str
    adapter: str
    endpoint: str | None = None
    strategy: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class GameConfig(BaseModel):
    """Game configuration within a suite."""

    type: str
    variant: str = "one_shot"
    config: dict[str, Any] = Field(default_factory=dict)


class GameSuiteConfig(BaseModel):
    """Complete game suite configuration parsed from YAML."""

    type: str = "game_suite"
    name: str
    version: str = "1.0"
    game: GameConfig
    agents: list[GameAgentConfig] = Field(min_length=2)
    evaluation: GameEvaluationConfig = Field(
        default_factory=GameEvaluationConfig,
    )
    reporting: GameReportingConfig = Field(
        default_factory=GameReportingConfig,
    )
