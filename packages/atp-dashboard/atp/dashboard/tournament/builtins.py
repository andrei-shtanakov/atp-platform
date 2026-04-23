"""Namespaced builtin-strategy registry for private test tournaments.

Wire name: ``{game}/{strategy}``. Cross-game collisions (e.g.
"random" in PD, auction, congestion, blotto) are disambiguated by
the game prefix. ``resolve_builtin`` derives a stable seed from
``(tournament_id, participant_id)`` and passes it to the class
constructor only when the class accepts a ``seed`` kwarg.

The underlying class lookup uses the real
``game_envs.strategies.registry.StrategyRegistry`` — a classmethod
registry populated when ``game_envs`` (the top-level package) is
imported. Importing ``game_envs`` runs the registration block in
``game_envs/__init__.py`` which calls ``StrategyRegistry.register``
for every built-in strategy class.

This module narrows the global registry to classes defined in the
given game's strategies module so cross-game namespacing works even
when bare names collide (e.g. ``random`` existing in multiple games).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from game_envs.core.strategy import Strategy
from game_envs.strategies.registry import StrategyRegistry


class BuiltinNotFoundError(KeyError):
    """Raised when a namespaced builtin name does not resolve."""


@dataclass(frozen=True)
class BuiltinDescriptor:
    """A single builtin listed for a game, with its wire name + docstring."""

    name: str  # "el_farol/traditionalist"
    description: str


# Each entry lists the strategy module that defines the game's
# builtin classes. Importing the module is how we scope
# ``StrategyRegistry`` lookups to just that game (the global
# registry is flat but class ``__module__`` attributes let us
# filter). The registration itself happens in ``game_envs/__init__.py``
# — only games with entries in that top-level register block are
# listed here so advertised support matches runtime resolution.
# ``stag_hunt`` and ``battle_of_sexes`` strategy classes exist but
# are not yet wired into the upstream registry; add them here once
# ``game_envs/__init__.py`` registers them.
_GAME_STRATEGY_MODULES: dict[str, str] = {
    "el_farol": "game_envs.strategies.el_farol_strategies",
    "prisoners_dilemma": "game_envs.strategies.pd_strategies",
    "colonel_blotto": "game_envs.strategies.blotto_strategies",
    "auction": "game_envs.strategies.auction_strategies",
    "congestion": "game_envs.strategies.congestion_strategies",
    "public_goods": "game_envs.strategies.pg_strategies",
}


def _load_game_strategies(game_type: str) -> dict[str, type[Strategy]]:
    """Return ``{bare_name: class}`` for the given game.

    Imports ``game_envs`` (to ensure the top-level registration block
    has run) and the game's own strategies module, then filters the
    global ``StrategyRegistry`` to classes defined in that module.
    """
    module_path = _GAME_STRATEGY_MODULES.get(game_type)
    if module_path is None:
        return {}
    # Top-level import populates StrategyRegistry via the register()
    # block in game_envs/__init__.py.
    importlib.import_module("game_envs")
    module = importlib.import_module(module_path)
    out: dict[str, type[Strategy]] = {}
    for bare_name in StrategyRegistry.list_strategies():
        try:
            cls = StrategyRegistry.get(bare_name)
        except KeyError:
            continue
        if cls.__module__ == module.__name__:
            out[bare_name] = cls
    return out


def list_builtins_for_game(game_type: str) -> list[BuiltinDescriptor]:
    """Namespaced descriptors for every builtin the game offers.

    Returns an empty list for unknown games so the endpoint
    ``GET /api/v1/games/{game_type}/builtins`` can respond 200 with
    ``builtins: []`` instead of 404.
    """
    out: list[BuiltinDescriptor] = []
    for bare_name, cls in _load_game_strategies(game_type).items():
        desc = (cls.__doc__ or "").strip().split("\n", 1)[0]
        out.append(BuiltinDescriptor(name=f"{game_type}/{bare_name}", description=desc))
    out.sort(key=lambda b: b.name)
    return out


def _stable_seed(tournament_id: int, participant_id: int) -> int:
    """Derive a process-stable integer seed.

    Python's built-in ``hash()`` is PYTHONHASHSEED-randomised per
    process — not usable here. We hash the identity pair with
    blake2b and project the first 8 bytes to an unsigned 64-bit int.
    """
    digest = hashlib.blake2b(
        f"{tournament_id}:{participant_id}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def resolve_builtin(
    namespaced_name: str,
    *,
    tournament_id: int,
    participant_id: int,
) -> Strategy:
    """Instantiate a builtin strategy by its namespaced wire name.

    The class's constructor is inspected: only classes that accept a
    ``seed`` keyword argument receive one, so non-RNG-backed
    strategies like ``Traditionalist(window_size=6)`` are not forced
    to take a parameter they don't understand.

    Raises:
        BuiltinNotFoundError: name malformed or unknown, or game
            unknown.
    """
    if "/" not in namespaced_name:
        raise BuiltinNotFoundError(
            f"strategy name must be namespaced as 'game/name', got {namespaced_name!r}"
        )
    game_type, bare_name = namespaced_name.split("/", 1)
    strategies = _load_game_strategies(game_type)
    cls = strategies.get(bare_name)
    if cls is None:
        raise BuiltinNotFoundError(
            f"unknown builtin strategy {namespaced_name!r} for game {game_type!r}"
        )
    kwargs: dict[str, Any] = {}
    params = inspect.signature(cls).parameters
    if "seed" in params:
        kwargs["seed"] = _stable_seed(tournament_id, participant_id)
    return cls(**kwargs)
