"""Pydantic model of the agent-eval-case format.

Mirrors ``method/agent-eval-case.schema.json`` (JSON Schema draft 2020-12) so a
case YAML can be parsed and validated in Python. Field/enum names match the
schema exactly. The cross-field rules from the schema's ``allOf`` (rubric/gold
presence, volatility turns, exclusive ``none`` tool) are enforced as validators.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Tag controlled-vocabulary pattern from the JSON schema.
_TAG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

Status = Literal["draft", "active", "retired"]
SuiteType = Literal["regression", "probe", "held_out"]
Capability = Literal[
    "correctness",
    "calibration",
    "efficiency",
    "safety_compliance",
    "recoverability",
    "adaptation",
]
ConstructionAxis = Literal[
    "information_conditions",
    "horizon_autonomy",
    "action_surface",
    "adversarial_environment",
    "requirements_volatility",
    "output_structure",
]
AxisLevel = Literal["clean", "mild", "moderate", "severe", "very_severe"]
ToolName = Literal["web_search", "file_read", "file_write", "api_call", "none"]
SideEffects = Literal["none", "reversible", "irreversible"]
ArtifactType = Literal["text", "file", "transcript", "table", "image", "url"]
GraderType = Literal[
    "exact", "regex", "programmatic", "rubric", "model_graded", "human"
]
TurnRole = Literal["user", "inject", "assistant"]


class Artifact(BaseModel):
    """Input material supplied to the agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-zA-Z0-9._-]+$")
    type: ArtifactType
    path: str | None = None
    content: str | None = None
    note: str | None = None


class RubricItem(BaseModel):
    """One weighted graded criterion."""

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(..., min_length=1)
    weight: float = Field(..., ge=0.0, le=1.0)


class Turn(BaseModel):
    """One turn of a multi-turn interaction script."""

    model_config = ConfigDict(extra="forbid")

    role: TurnRole
    content: str = Field(..., min_length=1)


class Environment(BaseModel):
    """Tooling and side-effect surface available to the agent."""

    model_config = ConfigDict(extra="forbid")

    tools: list[ToolName] = Field(..., min_length=1)
    side_effects: SideEffects

    @model_validator(mode="after")
    def validate_tools(self) -> Environment:
        """Tools are unique (schema uniqueItems) and 'none' is the sole entry."""
        if len(self.tools) != len(set(self.tools)):
            raise ValueError("environment.tools must not contain duplicates")
        if "none" in self.tools and len(self.tools) > 1:
            raise ValueError("tool 'none' must be the only entry in environment.tools")
        return self


class Grader(BaseModel):
    """How the run is scored: the binary critical_check plus an optional rubric."""

    model_config = ConfigDict(extra="forbid")

    type: GraderType
    gold: str | None = None
    rubric: list[RubricItem] | None = Field(default=None, min_length=1)
    critical_check: str = Field(..., min_length=1)
    scoring: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_grader_requirements(self) -> Grader:
        """Mirror schema allOf: rubric/model_graded need a rubric; exact needs gold."""
        if self.type in ("rubric", "model_graded") and not self.rubric:
            raise ValueError(f"grader type '{self.type}' requires a rubric")
        if self.type == "exact" and not self.gold:
            raise ValueError("grader type 'exact' requires a gold reference")
        return self


class Provenance(BaseModel):
    """Authorship and origin of the case."""

    model_config = ConfigDict(extra="forbid")

    author: str
    created: str
    source: str | None = None

    @field_validator("created")
    @classmethod
    def validate_created_is_iso_date(cls, v: str) -> str:
        """Schema declares format: date — require an ISO YYYY-MM-DD string."""
        try:
            date.fromisoformat(v)
        except ValueError as e:
            raise ValueError(
                f"provenance.created must be an ISO date (YYYY-MM-DD): {v!r}"
            ) from e
        return v


class AgentEvalCase(BaseModel):
    """A single agent-eval-case: an input task plus its trap and grading logic."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^case-[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: int = Field(..., ge=1)
    family: str = Field(..., pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    status: Status
    suite_type: SuiteType
    capability: Capability
    construction_axis: ConstructionAxis
    axis_level: AxisLevel
    tags: list[str] = Field(default_factory=list)
    instruction: str = Field(..., min_length=1)
    artifacts: list[Artifact] = Field(default_factory=list)
    environment: Environment
    constraints: list[str] = Field(default_factory=list)
    expected_failure_mode: str = Field(..., min_length=1)
    distractor: str | None = None
    grader: Grader
    turns: list[Turn] = Field(default_factory=list)
    provenance: Provenance

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Tags are unique and match the controlled-vocabulary pattern."""
        if len(v) != len(set(v)):
            raise ValueError("tags must be unique")
        bad = [t for t in v if not _TAG_RE.match(t)]
        if bad:
            raise ValueError(f"tags must match {_TAG_RE.pattern}; offending: {bad}")
        return v

    @model_validator(mode="after")
    def validate_volatility_turns(self) -> AgentEvalCase:
        """requirements_volatility cases must script an injected constraint."""
        if self.construction_axis == "requirements_volatility":
            if len(self.turns) < 2 or not any(t.role == "inject" for t in self.turns):
                raise ValueError(
                    "requirements_volatility cases require >=2 turns including an "
                    "'inject' turn"
                )
        return self
