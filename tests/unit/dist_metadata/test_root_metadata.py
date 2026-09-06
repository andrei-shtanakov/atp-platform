"""The root distribution must declare what the code it bundles imports.

``atp/`` holds symlinks into ``packages/*/atp/``, which hatchling dereferences
at build time, so the single published ``atp-platform`` wheel already contains
those members' code — while their own ``pyproject.toml`` requirements ship
nowhere. atp-platform 2.1.0 was uninstallable for exactly this reason: it
delegated them to distribution names that PyPI either does not have
(``atp-adapters``) or gives to an unrelated project (``atp-core``).

That makes the root dependency list a hand-written copy of the members' lists,
and a copy drifts silently: ``uv sync`` installs the members editable from the
workspace, so a requirement added to a member but forgotten at the root is
present in every dev and CI environment and missing only for real users. These
tests are the mechanical link the copy otherwise lacks.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[3]
ATP_PACKAGE = REPO_ROOT / "atp"

# Members whose own requirements ship with the atp-platform wheel are not these
# two: atp-platform-sdk is a real published dependency, and atp-method is a
# plugin users install separately.
NOT_BUNDLED = {"atp-sdk", "atp-method"}


def _load(pyproject: Path) -> dict[str, Any]:
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(r).name) for r in requirements}


def bundled_members() -> list[str]:
    """Workspace members reachable through the symlinks under ``atp/``.

    Derived from the tree rather than listed, so a new symlink cannot add code
    to the wheel without also being covered here.
    """
    members = set()
    for entry in sorted(ATP_PACKAGE.iterdir()):
        if not entry.is_symlink():
            continue
        target = Path(entry.readlink())
        parts = target.parts
        if "packages" in parts:
            member = parts[parts.index("packages") + 1]
            if member not in NOT_BUNDLED:
                members.add(member)
    return sorted(members)


@pytest.fixture(scope="module")
def root() -> dict[str, Any]:
    return _load(REPO_ROOT / "pyproject.toml")


def test_bundled_members_are_discovered() -> None:
    """Guards the discovery itself — an empty list would pass everything."""
    assert bundled_members() == ["atp-adapters", "atp-core", "atp-dashboard"]


def test_root_does_not_depend_on_bundled_members(root: dict[str, Any]) -> None:
    """The 2.1.0 failure, stated as an assertion.

    ``atp-core`` on PyPI is an unrelated project and ``atp-adapters`` does not
    exist there at all, so naming either one — in the base dependencies or in
    any extra — makes the published wheel resolve to a stranger's code or to
    nothing.
    """
    forbidden = {canonicalize_name(m) for m in bundled_members()}
    declared = list(root["dependencies"])
    for extra in root.get("optional-dependencies", {}).values():
        declared.extend(extra)
    assert not _names(declared) & forbidden


@pytest.mark.parametrize("member", bundled_members())
def test_root_covers_member_runtime_requirements(
    member: str, root: dict[str, Any]
) -> None:
    """Everything a bundled member imports must be installable from the root."""
    member_project = _load(REPO_ROOT / "packages" / member / "pyproject.toml")
    required = _names(member_project.get("dependencies", []))
    required -= {canonicalize_name(m) for m in bundled_members()}

    covered = _names(root["dependencies"])
    missing = required - covered
    assert not missing, (
        f"{member} requires {sorted(missing)}, which the root distribution does "
        f"not declare — installed users would get an ImportError"
    )


@pytest.mark.parametrize("member", bundled_members())
def test_root_extras_cover_member_extras(member: str, root: dict[str, Any]) -> None:
    """Optional requirements need the same coverage as mandatory ones.

    Compared as one union rather than extra-by-extra: the root is free to group
    them differently (``eco-server`` folded into the base install, for one), but
    a requirement reachable from a member extra and from no root extra is a
    package a user simply cannot install.
    """
    member_project = _load(REPO_ROOT / "packages" / member / "pyproject.toml")
    required: set[str] = set()
    for name, requirements in member_project.get("optional-dependencies", {}).items():
        if name == "dev":
            continue
        required |= _names(requirements)
    required -= {canonicalize_name(m) for m in bundled_members()}

    covered = _names(root["dependencies"])
    for requirements in root.get("optional-dependencies", {}).values():
        covered |= _names(requirements)
    covered -= {canonicalize_name(root["name"])}

    missing = required - covered
    assert not missing, (
        f"{member} offers {sorted(missing)} through an extra that no root extra exposes"
    )


@pytest.mark.parametrize("member", bundled_members())
def test_bundled_members_are_marked_unpublishable(member: str) -> None:
    """PyPI rejects any upload carrying a ``Private ::`` classifier.

    Defence in depth for the rule the tests above encode: these distributions
    have no PyPI identity of their own, and ``atp-core`` never can have one.
    """
    member_project = _load(REPO_ROOT / "packages" / member / "pyproject.toml")
    assert "Private :: Do Not Upload" in member_project.get("classifiers", [])
