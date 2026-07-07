"""Model-catalog schema (ADR-ECO-003b).

The `models` plane is the user-runtime contract: strict on the known fields,
tolerant of unknown ones (forward-compat + dev-SSOT extras). `harnesses`/`agents`
are optional dev-SSOT passthrough planes, kept raw so the dev-SSOT file parses;
they are formalized when the harness migrates (SP-E).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ModelEntry(BaseModel):
    """One model in the `models` plane."""

    model_config = ConfigDict(extra="allow")

    vendor: str
    status: Literal["active", "deprecated", "retired"]
    aliases: list[str] = []


class ModelCatalog(BaseModel):
    """A parsed model catalog."""

    model_config = ConfigDict(extra="allow")

    models: dict[str, ModelEntry]
    harnesses: dict[str, Any] | None = None
    agents: list[dict[str, Any]] | None = None
