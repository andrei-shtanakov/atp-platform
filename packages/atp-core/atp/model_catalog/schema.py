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


class CatalogDefaults(BaseModel):
    """The catalog's optional [defaults] plane (runtime defaults)."""

    model_config = ConfigDict(extra="allow")

    default_model: str | None = None


class ModelCatalog(BaseModel):
    """A parsed model catalog."""

    model_config = ConfigDict(extra="allow")

    models: dict[str, ModelEntry]
    harnesses: dict[str, HarnessEntry] | None = None
    agents: list[AgentEntry] | None = None
    defaults: CatalogDefaults | None = None

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

    @model_validator(mode="after")
    def _default_model_in_models(self) -> ModelCatalog:
        # Fires only when a default_model is set AND models is non-empty: the
        # default must be a models key or a ModelEntry alias (typo-catcher). A
        # catalog with no [defaults], or with empty models, is a no-op.
        if self.defaults is None or not self.defaults.default_model or not self.models:
            return self
        known = set(self.models) | {
            alias for entry in self.models.values() for alias in entry.aliases
        }
        if self.defaults.default_model not in known:
            raise ValueError(
                f"defaults.default_model {self.defaults.default_model!r} is not a "
                "known model id or alias"
            )
        return self
