"""Data models for scoring."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class NormalizationConfig(BaseModel):
    """Configuration for metric normalization."""

    max_steps: int | None = Field(None, description="Maximum expected steps", ge=1)
    optimal_steps: int | None = Field(
        None, description="Optimal (minimum) steps expected", ge=1
    )
    max_tokens: int | None = Field(None, description="Maximum expected tokens", ge=1)
    max_cost_usd: float | None = Field(
        None, description="Maximum expected cost in USD", gt=0
    )

    @field_validator("optimal_steps")
    @classmethod
    def validate_optimal_steps(cls, v: int | None, info: Any) -> int | None:
        """Ensure optimal_steps <= max_steps if both provided."""
        if v is not None:
            max_steps = info.data.get("max_steps")
            if max_steps is not None and v > max_steps:
                raise ValueError(
                    f"optimal_steps ({v}) cannot exceed max_steps ({max_steps})"
                )
        return v


class ComponentScore(BaseModel):
    """Individual component score with metadata."""

    name: str = Field(..., description="Component name", min_length=1)
    raw_value: float | None = Field(
        None, description="Raw metric value before normalization"
    )
    normalized_value: float = Field(
        ..., description="Normalized score (0.0-1.0)", ge=0.0, le=1.0
    )
    weight: float = Field(..., description="Weight applied to this component", ge=0.0)
    weighted_value: float = Field(..., description="normalized_value * weight", ge=0.0)
    details: dict[str, Any] | None = Field(
        None, description="Additional details about calculation"
    )


class ScoreBreakdown(BaseModel):
    """Complete breakdown of score calculation."""

    quality: ComponentScore = Field(..., description="Quality component score")
    completeness: ComponentScore = Field(
        ..., description="Completeness component score"
    )
    efficiency: ComponentScore = Field(..., description="Efficiency component score")
    cost: ComponentScore = Field(..., description="Cost component score")

    @property
    def final_score(self) -> float:
        """Calculate final score (0-100)."""
        weighted_sum = (
            self.quality.weighted_value
            + self.completeness.weighted_value
            + self.efficiency.weighted_value
            + self.cost.weighted_value
        )
        return round(weighted_sum * 100, 2)

    @property
    def components(self) -> list[ComponentScore]:
        """Return all components as a list."""
        return [self.quality, self.completeness, self.efficiency, self.cost]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "final_score": self.final_score,
            "components": {
                "quality": {
                    "normalized": self.quality.normalized_value,
                    "weight": self.quality.weight,
                    "weighted": self.quality.weighted_value,
                    "details": self.quality.details,
                },
                "completeness": {
                    "normalized": self.completeness.normalized_value,
                    "weight": self.completeness.weight,
                    "weighted": self.completeness.weighted_value,
                    "details": self.completeness.details,
                },
                "efficiency": {
                    "normalized": self.efficiency.normalized_value,
                    "weight": self.efficiency.weight,
                    "weighted": self.efficiency.weighted_value,
                    "details": self.efficiency.details,
                },
                "cost": {
                    "normalized": self.cost.normalized_value,
                    "weight": self.cost.weight,
                    "weighted": self.cost.weighted_value,
                    "details": self.cost.details,
                },
            },
        }


class ScoredTestResult(BaseModel):
    """Test result with computed score."""

    test_id: str = Field(..., description="Test identifier")
    score: float = Field(
        ..., description="Final composite score (0-100)", ge=0.0, le=100.0
    )
    breakdown: ScoreBreakdown = Field(..., description="Score breakdown")
    passed: bool = Field(..., description="Whether all evaluations passed")
