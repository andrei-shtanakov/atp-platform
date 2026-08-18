"""Model-catalog schema (ADR-ECO-003b).

The `models` plane is the user-runtime contract: strict on the known fields,
tolerant of unknown ones. `harnesses`/`agents` are the dev-SSOT planes, typed in
SP-E; cross-plane validators tie them together when both planes are present (a
models-only user catalog is a no-op — SP-A fork A).

Rule vocabulary V1..V7 comes from the shared catalog-conformance contract
(vendored at `tests/unit/model_catalog/fixtures/catalog-conformance/v1/`;
owner: devtools). Every rule message is prefixed with its code so a consumer —
and the conformance suite — can tell *which* rule fired:

* V1 `agents.harness` not declared in `[harnesses.*]` — error
* V2 `agents.model` not declared in `[models.*]` — error
* V3 `agents` references a `status="retired"` model — error
* V4 duplicate `agent_id` (`<harness>@<model>`) — error
* V5 `agents.routable = true` under a `routable = false` harness — error
* V6 `agents` references a `status="deprecated"` model — warning (`CatalogWarning`)
* V7 unknown enum value — an unknown `status` is a hard schema failure via the
  `Literal` below (rejection is a conformant response to the "flag" class); an
  unknown harness `kind` is a warning, so a catalog naming a launch mechanism we
  do not recognize still loads. The two halves are deliberately asymmetric:
  `status` gates enrollment (`retired`/`deprecated` decide whether an agent may
  run), while `kind` only describes how a harness is launched.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from atp.model_catalog.errors import CatalogWarning

# The harness-kind vocabulary is owned by ADR-ECO-003, not by this loader; it is
# restated here so an unfamiliar value is *observable* (V7). Deliberately a
# warning, not a Literal: a catalog that adds a launch mechanism must still load.
KNOWN_HARNESS_KINDS = frozenset({"cli", "api-baseline", "local"})


class ModelEntry(BaseModel):
    """One model in the `models` plane."""

    model_config = ConfigDict(extra="allow")

    vendor: str
    status: Literal["active", "deprecated", "retired"]
    aliases: list[str] = []


class HarnessEntry(BaseModel):
    """One harness in the dev-SSOT `harnesses` plane.

    ``shim`` is ATP-side sweep machinery, not part of the shared cross-repo
    catalog contract, so it is optional here: requiring it would make ATP reject
    contract-valid catalogs for a reason no sibling loader shares. The pipe-check
    harness enforces its presence for the harnesses it actually spawns.
    """

    model_config = ConfigDict(extra="allow")

    kind: str
    shim: str | None = None
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

    @property
    def agent_id(self) -> str:
        """The ecosystem join key: ``<harness>@<model>`` (ADR-ECO-003)."""
        return f"{self.harness}@{self.model}"


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

    def _agent_ids_referencing_status(self, status: str) -> list[str]:
        """agent_ids enrolled on a model declared with ``status`` (sorted, unique)."""
        agents = self.agents or []
        return sorted(
            {
                a.agent_id
                for a in agents
                if a.model in self.models and self.models[a.model].status == status
            }
        )

    @model_validator(mode="after")
    def _v1_agents_reference_declared_harnesses(self) -> ModelCatalog:
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
            raise ValueError(
                f"V1: agents reference undeclared harness(es): {undeclared}"
            )
        return self

    @model_validator(mode="after")
    def _v2_agents_reference_declared_models(self) -> ModelCatalog:
        # Like V1, armed only when both planes carry content: a catalog with an
        # empty `models` plane declares no models to check against.
        if self.agents is None or not self.models:
            return self
        undeclared = sorted(
            {a.model for a in self.agents if a.model not in self.models}
        )
        if undeclared:
            raise ValueError(
                f"V2: agents reference undeclared model id(s): {undeclared}"
            )
        return self

    @model_validator(mode="after")
    def _v3_agents_do_not_reference_retired_models(self) -> ModelCatalog:
        # retired = "do not enroll this pair" (ADR-ECO-003a). A retired model may
        # stay declared as a regression guard as long as nothing references it.
        if self.agents is None or not self.models:
            return self
        retired = self._agent_ids_referencing_status("retired")
        if retired:
            raise ValueError(f"V3: agent(s) enrolled on a retired model: {retired}")
        return self

    @model_validator(mode="after")
    def _v4_agent_ids_are_unique(self) -> ModelCatalog:
        # agent_id is the byte-exact cross-repo join key; a duplicate makes the
        # enrollment ambiguous no matter which planes are present.
        if self.agents is None:
            return self
        counts = Counter(a.agent_id for a in self.agents)
        dupes = sorted(k for k, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"V4: duplicate agent_id(s): {dupes}")
        return self

    @model_validator(mode="after")
    def _v5_routable_agents_need_routable_harness(self) -> ModelCatalog:
        if self.harnesses is None or self.agents is None:
            return self
        conflicts = sorted(
            {
                a.agent_id
                for a in self.agents
                if a.routable
                and a.harness in self.harnesses
                and not self.harnesses[a.harness].routable
            }
        )
        if conflicts:
            raise ValueError(
                f"V5: routable agent(s) under a non-routable harness: {conflicts}"
            )
        return self

    @model_validator(mode="after")
    def _v6_warn_on_deprecated_model_references(self) -> ModelCatalog:
        # Non-fatal by the reference contract: deprecated still runs, but a
        # silent acceptance would hide a pending retirement.
        if self.agents is None or not self.models:
            return self
        deprecated = self._agent_ids_referencing_status("deprecated")
        if deprecated:
            warnings.warn(
                f"V6: agent(s) enrolled on a deprecated model: {deprecated}",
                CatalogWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _v7_warn_on_unknown_harness_kind(self) -> ModelCatalog:
        # The `status` half of V7 is caught by ModelEntry's Literal before any
        # model validator runs; `kind` is checked here so a catalog whose only
        # deviation is an unknown kind is still not accepted silently.
        if self.harnesses is None:
            return self
        unknown = sorted(
            {
                f"{name}={h.kind}"
                for name, h in self.harnesses.items()
                if h.kind not in KNOWN_HARNESS_KINDS
            }
        )
        if unknown:
            warnings.warn(
                f"V7: harness(es) with an unknown kind: {unknown}",
                CatalogWarning,
                stacklevel=2,
            )
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
