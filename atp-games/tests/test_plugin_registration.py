"""Tests for plugin registration of game evaluators."""

from atp.evaluators.registry import EvaluatorRegistry

from atp_games.evaluators.exploitability_evaluator import (
    ExploitabilityEvaluator,
)
from atp_games.evaluators.payoff_evaluator import PayoffEvaluator
from atp_games.plugin import register


class TestPluginRegistration:
    def test_register_adds_evaluators(self) -> None:
        """Plugin register() adds evaluators to registry."""
        # Use a fresh registry to avoid side effects
        registry = EvaluatorRegistry()
        assert not registry.is_registered("payoff")
        assert not registry.is_registered("exploitability")

        # Monkey-patch get_registry to return our test registry
        import atp.evaluators.registry as reg_mod

        original = reg_mod.get_registry
        reg_mod.get_registry = lambda: registry  # type: ignore[assignment]
        try:
            register()
        finally:
            reg_mod.get_registry = original

        assert registry.is_registered("payoff")
        assert registry.is_registered("exploitability")

    def test_registered_classes_correct(self) -> None:
        """Registered classes are the right evaluator types."""
        registry = EvaluatorRegistry()
        import atp.evaluators.registry as reg_mod

        original = reg_mod.get_registry
        reg_mod.get_registry = lambda: registry  # type: ignore[assignment]
        try:
            register()
        finally:
            reg_mod.get_registry = original

        payoff_cls = registry.get_evaluator_class("payoff")
        assert payoff_cls is PayoffEvaluator

        exploit_cls = registry.get_evaluator_class("exploitability")
        assert exploit_cls is ExploitabilityEvaluator

    def test_assertion_mappings_registered(self) -> None:
        """Assertion type mappings are registered."""
        registry = EvaluatorRegistry()
        import atp.evaluators.registry as reg_mod

        original = reg_mod.get_registry
        reg_mod.get_registry = lambda: registry  # type: ignore[assignment]
        try:
            register()
        finally:
            reg_mod.get_registry = original

        assert registry.supports_assertion("average_payoff")
        assert registry.supports_assertion("exploitability")

    def test_create_evaluator_instances(self) -> None:
        """Can create evaluator instances from registry."""
        registry = EvaluatorRegistry()
        import atp.evaluators.registry as reg_mod

        original = reg_mod.get_registry
        reg_mod.get_registry = lambda: registry  # type: ignore[assignment]
        try:
            register()
        finally:
            reg_mod.get_registry = original

        payoff_eval = registry.create("payoff")
        assert isinstance(payoff_eval, PayoffEvaluator)
        assert payoff_eval.name == "payoff"

        exploit_eval = registry.create("exploitability")
        assert isinstance(exploit_eval, ExploitabilityEvaluator)
        assert exploit_eval.name == "exploitability"
