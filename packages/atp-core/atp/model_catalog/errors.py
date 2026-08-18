"""Typed errors for the model catalog (ADR-ECO-003b).

Distinct types so the CLI maps to clear messages + non-zero exit and future
evaluator/harness consumers can catch programmatically.
"""

from __future__ import annotations


class CatalogError(Exception):
    """Base for all model-catalog errors."""


class CatalogNotConfiguredError(CatalogError):
    """No usable catalog location.

    Two message forms (the text differentiates them): 'not configured' (needs
    `atp models init`) vs 'configured but invalid' (an env path must be absolute).
    """


class CatalogMissingFileError(CatalogError):
    """A catalog location was configured explicitly but the file is not there.

    Distinct from :class:`CatalogNotConfiguredError`: the user did point us at a
    file, so falling through to a lower resolution layer (or reporting "not
    configured") would silently ignore an explicit instruction.
    """


class CatalogTOMLError(CatalogError):
    """The catalog file exists but is not valid TOML."""


class CatalogSchemaError(CatalogError):
    """The catalog parsed as TOML but failed schema validation."""


class CatalogWarning(UserWarning):
    """A non-fatal catalog problem the loader must not accept silently.

    Carries the "flag" class of the shared catalog-conformance contract (V6:
    an agent referencing a deprecated model).
    """
