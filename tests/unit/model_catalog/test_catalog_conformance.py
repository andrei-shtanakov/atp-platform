"""Conformance of `atp.model_catalog` against the shared catalog fixture set.

The set is owned by devtools and vendored here as a pinned copy
(`fixtures/catalog-conformance/v1/`, see its `PIN`). It exists to make loader
divergence between the three ecosystem loaders (Maestro, ATP, arbiter)
*observable* — ADR-ECO-003b risk №1. Accepted from inbox issue #292
(slug `catalog-conformance-wiring`).

Every `[[case]]` and `[[pathres]]` in the vendored `expectations.toml` becomes a
test. Negative cases assert the *specific* rule fired (the loader messages carry
`V1:`..`V6:` prefixes), not merely that something failed — a test that goes green
because of an unrelated rejection would hide exactly the divergence the set is
meant to surface.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
import warnings
from pathlib import Path

import pytest

from atp.model_catalog import (
    CatalogMissingFileError,
    CatalogNotConfiguredError,
    CatalogSchemaError,
    CatalogTOMLError,
    CatalogWarning,
    load_catalog,
)

CONTRACT_DIR = Path(__file__).parent / "fixtures" / "catalog-conformance" / "v1"
_EXPECTATIONS = tomllib.loads(
    (CONTRACT_DIR / "expectations.toml").read_text(encoding="utf-8")
)
CASES: list[dict[str, str]] = _EXPECTATIONS["case"]
PATHRES: list[dict[str, str]] = _EXPECTATIONS["pathres"]

# Files outside the hashed pin surface (the manifest itself and our PIN note).
_OFF_SURFACE = {"manifest.json", "PIN"}


def _clear_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Neutralize every resolution layer but `$ATP_CATALOG`.

    The shared set covers the `$ATP_CATALOG` layer only (ADR-ECO-003b D2); XDG
    is pointed at an empty directory so ATP's extra layers cannot answer.
    """
    monkeypatch.delenv("ATP_CATALOG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))


# --------------------------------------------------------------------------
# Pin integrity: the vendored copy must be byte-identical to what devtools published
# --------------------------------------------------------------------------


def _surface_files() -> list[Path]:
    return sorted(
        p for p in CONTRACT_DIR.rglob("*") if p.is_file() and p.name not in _OFF_SURFACE
    )


def test_pin_file_names_the_source_commit() -> None:
    pin = (CONTRACT_DIR / "PIN").read_text(encoding="utf-8")
    assert "devtools@2533ff7b8c3afd74110b3838325bf76ba46ba186" in pin


def test_vendored_copy_matches_manifest() -> None:
    manifest = json.loads((CONTRACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    on_disk = {
        p.relative_to(CONTRACT_DIR).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in _surface_files()
    }
    declared = {e["path"]: e["sha256"] for e in manifest["files"]}
    assert on_disk == declared

    # tree_sha256 is defined over *sorted* "<path> <sha256>" lines, so sort here
    # rather than trusting the manifest's list order: a reordered (but otherwise
    # identical) manifest is still the same tree.
    tree = "\n".join(f"{path} {sha}" for path, sha in sorted(declared.items())) + "\n"
    assert hashlib.sha256(tree.encode("utf-8")).hexdigest() == manifest["tree_sha256"]


def test_every_fixture_on_disk_has_a_case() -> None:
    fixtures = {
        p.relative_to(CONTRACT_DIR).as_posix()
        for p in CONTRACT_DIR.rglob("fixtures/**/*.toml")
    }
    assert fixtures == {c["file"] for c in CASES}


# --------------------------------------------------------------------------
# [[case]] — loader behavior per expectation class
# --------------------------------------------------------------------------


def _case_id(case: dict[str, str]) -> str:
    stem = Path(case["file"]).stem
    return f"{stem}-{case['expect']}"


@pytest.fixture(params=CASES, ids=_case_id)
def case(request: pytest.FixtureRequest) -> dict[str, str]:
    return request.param


def test_case_conforms(case: dict[str, str]) -> None:
    path = CONTRACT_DIR / case["file"]
    expect = case["expect"]

    if expect == "valid":
        with warnings.catch_warnings():
            warnings.simplefilter("error", CatalogWarning)
            catalog = load_catalog(path)  # loads, and warning-free
        assert catalog.models
        return

    if expect == "parse-error":
        with pytest.raises(CatalogTOMLError):
            load_catalog(path)
        return

    if expect == "error":
        with pytest.raises(CatalogSchemaError) as excinfo:
            load_catalog(path)
        # The rule named by the fixture must be the one that fired.
        assert f"{case['code']}:" in str(excinfo.value)
        return

    if expect == "flag":
        # "flag" = at least warned about; outright rejection also conforms.
        # What is NOT conformant is silent acceptance as a healthy catalog.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                load_catalog(path)
            except CatalogSchemaError:
                return  # rejection conforms
        assert any(issubclass(w.category, CatalogWarning) for w in caught), (
            f"{case['file']} was accepted silently"
        )
        assert any(f"{case['code']}:" in str(w.message) for w in caught)
        return

    raise AssertionError(f"unknown expectation class: {expect!r}")


def test_v6_deprecated_reference_is_a_warning_not_a_rejection() -> None:
    """ATP's chosen answer to the V6 "flag" class: warn, keep loading.

    The contract allows rejection too; pinning the choice here means a future
    change to it is a deliberate edit, not a silent drift.
    """
    path = CONTRACT_DIR / "fixtures/warn/v6-deprecated-ref.toml"
    with pytest.warns(CatalogWarning, match="V6:"):
        catalog = load_catalog(path)
    assert catalog.agents is not None


def test_v7_unknown_status_is_a_hard_schema_failure() -> None:
    """ATP's answer to the `status` half of V7: reject via the Literal.

    The fixture varies `status` and `kind` at once, and the field-level Literal
    fires before any model validator, so this file never reaches the kind check —
    which is precisely why the set carries the kind-only fixture below.
    """
    path = CONTRACT_DIR / "fixtures/warn/v7-unknown-enum.toml"
    with pytest.raises(CatalogSchemaError, match="status"):
        load_catalog(path)


def test_v7_unknown_kind_is_a_warning_not_a_rejection() -> None:
    """ATP's answer to the `kind` half of V7: warn, keep loading.

    `kind` only describes how a harness is launched, so an unrecognized one must
    be visible without breaking a catalog that adds a launch mechanism before we
    know about it. The vocabulary belongs to ADR-ECO-003; we restate it to make
    the deviation observable, which is what the "flag" class asks for.
    """
    path = CONTRACT_DIR / "fixtures/warn/v7-unknown-kind.toml"
    with pytest.warns(CatalogWarning, match="V7:"):
        catalog = load_catalog(path)
    assert catalog.harnesses is not None


def test_v1_empty_harnesses_plane_fails_closed() -> None:
    """An empty `[harnesses]` plane declares zero harnesses (devtools#47 canon).

    The rival reading — a bare header as unarmed scaffolding — would make the
    catalog silently valid. ATP was already fail-closed here; this pins it.
    """
    path = CONTRACT_DIR / "fixtures/invalid/v1-empty-harnesses.toml"
    with pytest.raises(CatalogSchemaError, match="V1:"):
        load_catalog(path)


# --------------------------------------------------------------------------
# [[pathres]] — $ATP_CATALOG resolution layer (ADR-ECO-003b D2)
# --------------------------------------------------------------------------


@pytest.fixture(params=PATHRES, ids=lambda p: str(p["id"]))
def pathres(request: pytest.FixtureRequest) -> dict[str, str]:
    return request.param


def test_pathres_conforms(
    pathres: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch, tmp_path)
    env = pathres["env"]

    if env == "set":
        monkeypatch.setenv("ATP_CATALOG", str(CONTRACT_DIR / pathres["target"]))
    elif env == "set-missing":
        monkeypatch.setenv("ATP_CATALOG", str(tmp_path / "no-such-catalog.toml"))
    elif env != "unset":
        raise AssertionError(f"unknown pathres env: {env!r}")

    expect = pathres["expect"]
    if expect == "loaded":
        assert load_catalog().models
    elif expect == "not-configured":
        with pytest.raises(CatalogNotConfiguredError):
            load_catalog()
    elif expect == "missing-file-error":
        # Must surface the missing file, NOT fall through to a lower layer and
        # report "not configured" (that would be silently ignoring the setting).
        with pytest.raises(CatalogMissingFileError):
            load_catalog()
    else:
        raise AssertionError(f"unknown pathres expectation: {expect!r}")


def test_missing_atp_catalog_does_not_fall_through_to_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sharp edge behind `missing-file-error`: a usable XDG catalog must not
    quietly stand in for the file the user explicitly named."""
    xdg = tmp_path / "xdg"
    target = xdg / "atp" / "agents-catalog.toml"
    target.parent.mkdir(parents=True)
    target.write_text(
        (CONTRACT_DIR / "fixtures/valid/three-planes.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("ATP_CATALOG", str(tmp_path / "no-such-catalog.toml"))
    with pytest.raises(CatalogMissingFileError):
        load_catalog()
