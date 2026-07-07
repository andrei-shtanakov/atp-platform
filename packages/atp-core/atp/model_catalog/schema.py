"""Model-catalog schema (ADR-ECO-003b).

The `models` plane is the user-runtime contract: strict on the known fields,
tolerant of unknown ones. `harnesses`/`agents` are the dev-SSOT planes, typed in
SP-E; a referential validator ties agents to declared harnesses when both planes
are present (a models-only user catalog is a no-op — SP-A fork A).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ModelEntry(BaseModel):
    """One model in the `models` plane."""

    model_config = ConfigDict(extra="allow")

    vendor: str
    status: Literal["active", "deprecated", "retired"]
    aliases: list[str] = []


class HarnessEntry(BaseModel):
    """One harness in the dev-SSOT `harnesses` plane."""

    model_config = ConfigDict(extra="allow")

    kind: str
    shim: str
    model_env: str
    model_flag: str | None = None
    routable: bool = False


class AgentEntry(BaseModel):
    """One agent in the dev-SSOT `agents` plane."""

    model_config = ConfigDict(extra="allow")

    harness: str
    model: str
    tested: bool = False
    routable: bool = False


class ModelCatalog(BaseModel):
    """A parsed model catalog."""

    model_config = ConfigDict(extra="allow")

    models: dict[str, ModelEntry]
    harnesses: dict[str, HarnessEntry] | None = None
    agents: list[AgentEntry] | None = None

    @model_validator(mode="after")
    def _agents_reference_declared_harnesses(self) -> ModelCatalog:
        # Referential integrity fires only when BOTH planes are present
        # (present-empty counts as present); a models-only user catalog is a
        # no-op, preserving SP-A fork A.
        if self.harnesses is None or self.agents is None:
            return self
        declared = set(self.harnesses)
        undeclared = sorted(
            {a.harness for a in self.agents if a.harness not in declared}
        )
        if undeclared:
            raise ValueError(f"agents reference undeclared harness(es): {undeclared}")
        return self
