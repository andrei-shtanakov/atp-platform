"""Pydantic model of the agent-eval-case format.

Mirrors ``method/agent-eval-case.schema.json`` (JSON Schema draft 2020-12) so a
case YAML can be parsed and validated in Python. Field/enum names match the
schema exactly. The cross-field rules from the schema's ``allOf`` (rubric/gold
presence, volatility turns, exclusive ``none`` tool) are enforced as validators.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Tag controlled-vocabulary pattern from the JSON schema.
_TAG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

# Lowercase token pattern for routing fields (task_type, language).
_TOKEN_RE = r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$"

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
    "exact",
    "regex",
    "programmatic",
    "rubric",
    "model_graded",
    "human",
]
TurnRole = Literal["user", "inject", "assistant"]
RunMode = Literal["text_out", "read_only_corpus", "workspace"]
# Keep this to modes the harness can actually execute, so a case cannot declare
# fidelity the runtime cannot deliver (ADR-007 §3).
WIRED_RUN_MODES = frozenset({"text_out", "read_only_corpus"})


def _validate_safe_relative_path(value: str, *, field_name: str = "path") -> str:
    """Validate ATP-safe relative corpus paths and glob patterns."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    if "\x00" in value:
        raise ValueError(f"{field_name} contains an unsafe null byte")
    if value.startswith("/") or value.startswith("~"):
        raise ValueError(f"{field_name} must be a relative path")
    normalized = value.replace("\\", "/")
    if "//" in normalized:
        raise ValueError(f"{field_name} contains an empty path segment")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{field_name} contains an unsafe path segment")
    return value


class Artifact(BaseModel):
    """Input material supplied to the agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-zA-Z0-9._-]+$")
    type: ArtifactType
    path: str | None = None
    content: str | None = None
    note: str | None = None


class CorpusDigest(BaseModel):
    """Digest manifest declaration for a read-only artifact corpus."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sha256"]
    manifest_path: str
    normalization: Literal["lf"]

    @field_validator("manifest_path")
    @classmethod
    def validate_manifest_path(cls, v: str) -> str:
        """Manifest paths are safe and relative to the corpus root."""
        return _validate_safe_relative_path(v, field_name="manifest_path")


class ArtifactCorpus(BaseModel):
    """A corpus folder exposed to an agent through read-only file tools."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    root: str
    include: list[str] = Field(..., min_length=1)
    exclude: list[str] = Field(default_factory=list)
    digest: CorpusDigest
    metadata_path: str | None = None

    @field_validator("root")
    @classmethod
    def validate_root(cls, v: str) -> str:
        """Corpus root is relative to the case file."""
        return _validate_safe_relative_path(v, field_name="root")

    @field_validator("include", "exclude")
    @classmethod
    def validate_patterns(cls, v: list[str]) -> list[str]:
        """Glob patterns still obey safe relative path rules."""
        for pattern in v:
            _validate_safe_relative_path(pattern, field_name="path pattern")
        return v

    @field_validator("metadata_path")
    @classmethod
    def validate_metadata_path(cls, v: str | None) -> str | None:
        """Optional metadata path is relative to the corpus root."""
        if v is None:
            return None
        return _validate_safe_relative_path(v, field_name="metadata_path")


class CorpusFileMetadata(BaseModel):
    """Semantic metadata for a selected corpus file."""

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    status: str | None = None
    document_id: str | None = None


class CorpusMetadata(BaseModel):
    """Optional semantic metadata keyed by corpus-relative file path."""

    model_config = ConfigDict(extra="forbid")

    files: dict[str, CorpusFileMetadata] = Field(default_factory=dict)

    @field_validator("files")
    @classmethod
    def validate_file_paths(
        cls, v: dict[str, CorpusFileMetadata]
    ) -> dict[str, CorpusFileMetadata]:
        """Metadata keys use the same safe relative corpus path rules."""
        for path in v:
            _validate_safe_relative_path(path, field_name="metadata file path")
        return v


class RubricItem(BaseModel):
    """One weighted graded criterion."""

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(..., min_length=1)
    weight: float = Field(..., ge=0.0, le=1.0)


class ExpectedFinding(BaseModel):
    """A planted defect the agent MUST surface (anchor + rule synonyms)."""

    model_config = ConfigDict(extra="forbid")

    rule_ids: list[str] = Field(..., min_length=1)
    anchor: str = Field(..., min_length=1)
    severity: Literal["critical", "major", "minor"] = "critical"


class ForbiddenAnchor(BaseModel):
    """A compliant line the agent MUST NOT flag (false-positive trap)."""

    model_config = ConfigDict(extra="forbid")

    anchor: str = Field(..., min_length=1)


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
    checker: str | None = None
    gold: str | None = None
    rubric: list[RubricItem] | None = Field(default=None, min_length=1)
    critical_check: str = Field(..., min_length=1)
    scoring: str = Field(..., min_length=1)
    expected_findings: list[ExpectedFinding] | None = None
    must_not_flag: list[ForbiddenAnchor] | None = None
    config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_grader_requirements(self) -> Grader:
        """Mirror schema allOf: rubric/model_graded need a rubric; exact needs gold.

        A ``checker`` is only valid under ``type='programmatic'``. The
        ``findings_match`` checker requires expected_findings to be present (use []
        for compliant cases with no planted defect).
        """
        if self.type in ("rubric", "model_graded") and not self.rubric:
            raise ValueError(f"grader type '{self.type}' requires a rubric")
        if self.type == "exact" and not self.gold:
            raise ValueError("grader type 'exact' requires a gold reference")
        if self.checker is not None and not self.checker.strip():
            # Empty/whitespace checker is falsey at dispatch time and would
            # silently fall back to judge grading — reject it as malformed.
            raise ValueError("grader.checker must be a non-empty string")
        if self.checker is not None and self.type != "programmatic":
            raise ValueError("grader.checker requires type 'programmatic'")
        if self.checker == "findings_match" and self.expected_findings is None:
            raise ValueError(
                "checker 'findings_match' requires expected_findings "
                "(use [] for a compliant case with no planted defect)"
            )
        if self.checker == "json_path":
            assertions = (self.config or {}).get("assertions")
            if not isinstance(assertions, list) or not assertions:
                raise ValueError(
                    "checker 'json_path' requires grader.config.assertions "
                    "(a non-empty list)"
                )
        if self.checker == "citation_grounding":
            expected = (self.config or {}).get("expected")
            if not isinstance(expected, list) or not expected:
                raise ValueError(
                    "checker 'citation_grounding' requires grader.config.expected "
                    "(a non-empty list)"
                )
        return self


class BehaviorAssertion(BaseModel):
    """A native ATP behavior assertion carried by an agent-eval-case."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["behavior"]
    critical: bool = False
    config: dict[str, Any]


class OutputContract(BaseModel):
    """The structured artifact the agent must return for this case."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_name: str = Field(..., min_length=1)
    content_type: str = "application/json"
    # On-disk key is "schema"; Python name avoids the BaseModel.schema shadow.
    json_schema: dict[str, Any] = Field(..., alias="schema")
    format_instruction: str | None = None


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
    task_type: str | None = Field(default=None, pattern=_TOKEN_RE)
    language: str | None = Field(default=None, pattern=_TOKEN_RE)
    tags: list[str] = Field(default_factory=list)
    instruction: str = Field(..., min_length=1)
    artifacts: list[Artifact] = Field(default_factory=list)
    environment: Environment
    constraints: list[str] = Field(default_factory=list)
    expected_failure_mode: str = Field(..., min_length=1)
    distractor: str | None = None
    grader: Grader
    behavior_assertions: list[BehaviorAssertion] = Field(default_factory=list)
    output_contract: OutputContract | None = None
    run_mode: RunMode = "text_out"
    artifact_corpus: ArtifactCorpus | None = None
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

    @model_validator(mode="after")
    def validate_artifact_corpus_run_mode(self) -> AgentEvalCase:
        """Corpus declarations and read_only_corpus mode are paired."""
        if self.artifact_corpus is not None and self.run_mode != "read_only_corpus":
            raise ValueError("artifact_corpus requires run_mode 'read_only_corpus'")
        if self.run_mode == "read_only_corpus" and self.artifact_corpus is None:
            raise ValueError("run_mode 'read_only_corpus' requires artifact_corpus")
        return self

    @model_validator(mode="after")
    def validate_run_mode_wired(self) -> AgentEvalCase:
        """Reject run_mode tiers the harness cannot deliver yet (ADR-007 §3)."""
        if self.run_mode not in WIRED_RUN_MODES:
            raise ValueError(
                f"run_mode '{self.run_mode}' is declared but not wired; "
                f"supported: {sorted(WIRED_RUN_MODES)}"
            )
        return self
