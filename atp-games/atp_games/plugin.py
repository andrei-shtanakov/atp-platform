"""ATP plugin registration for game-theoretic evaluation."""

from __future__ import annotations

# Registry for suite loaders keyed by type discriminator
_suite_loaders: dict[str, type] = {}


def register_suite_loader(
    suite_type: str,
    loader_class: type,
) -> None:
    """Register a suite loader for a given type discriminator.

    Args:
        suite_type: The value of the ``type`` key in YAML.
        loader_class: The loader class to instantiate.
    """
    _suite_loaders[suite_type] = loader_class


def get_suite_loader(suite_type: str) -> type | None:
    """Get a registered suite loader by type.

    Args:
        suite_type: The suite type discriminator.

    Returns:
        The loader class, or None if not registered.
    """
    return _suite_loaders.get(suite_type)


def register() -> None:
    """Register atp-games components with ATP.

    Called by ATP plugin discovery via entry points.
    Registers game-theoretic evaluators and the game suite
    loader in the ATP plugin system.
    """
    from atp.evaluators.registry import get_registry

    from atp_games.evaluators.cooperation_evaluator import (
        CooperationEvaluator,
    )
    from atp_games.evaluators.equilibrium_evaluator import (
        EquilibriumEvaluator,
    )
    from atp_games.evaluators.exploitability_evaluator import (
        ExploitabilityEvaluator,
    )
    from atp_games.evaluators.fairness_evaluator import (
        FairnessEvaluator,
    )
    from atp_games.evaluators.payoff_evaluator import (
        PayoffEvaluator,
    )
    from atp_games.suites.game_suite_loader import GameSuiteLoader

    registry = get_registry()
    registry.register("payoff", PayoffEvaluator)
    registry.register("exploitability", ExploitabilityEvaluator)
    registry.register("cooperation", CooperationEvaluator)
    registry.register("equilibrium", EquilibriumEvaluator)
    registry.register("fairness", FairnessEvaluator)

    # Register assertion type mappings
    registry._register_assertion_mapping("average_payoff", "payoff")
    registry._register_assertion_mapping("exploitability", "exploitability")
    registry._register_assertion_mapping("cooperation_rate", "cooperation")
    registry._register_assertion_mapping("nash_distance", "equilibrium")
    registry._register_assertion_mapping("equilibrium_type", "equilibrium")
    registry._register_assertion_mapping("fairness", "fairness")

    # Register game suite loader
    register_suite_loader("game_suite", GameSuiteLoader)
