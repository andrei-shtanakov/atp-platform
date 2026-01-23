"""Unit tests for EvaluatorRegistry."""

import pytest

from atp.evaluators.artifact import ArtifactEvaluator
from atp.evaluators.base import Evaluator
from atp.evaluators.behavior import BehaviorEvaluator
from atp.evaluators.registry import (
    EvaluatorNotFoundError,
    EvaluatorRegistry,
    create_evaluator,
    get_registry,
)


@pytest.fixture
def registry() -> EvaluatorRegistry:
    """Create a fresh EvaluatorRegistry instance."""
    return EvaluatorRegistry()


class TestEvaluatorRegistry:
    """Tests for EvaluatorRegistry class."""

    def test_builtin_evaluators_registered(self, registry: EvaluatorRegistry) -> None:
        """Test that built-in evaluators are registered."""
        assert registry.is_registered("artifact")
        assert registry.is_registered("behavior")

    def test_list_evaluators(self, registry: EvaluatorRegistry) -> None:
        """Test listing registered evaluators."""
        evaluators = registry.list_evaluators()
        assert "artifact" in evaluators
        assert "behavior" in evaluators

    def test_get_evaluator_class(self, registry: EvaluatorRegistry) -> None:
        """Test getting evaluator class by type."""
        artifact_class = registry.get_evaluator_class("artifact")
        assert artifact_class == ArtifactEvaluator

        behavior_class = registry.get_evaluator_class("behavior")
        assert behavior_class == BehaviorEvaluator

    def test_get_evaluator_class_not_found(self, registry: EvaluatorRegistry) -> None:
        """Test error when evaluator type not found."""
        with pytest.raises(EvaluatorNotFoundError) as exc_info:
            registry.get_evaluator_class("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_create_evaluator(self, registry: EvaluatorRegistry) -> None:
        """Test creating evaluator instances."""
        artifact_eval = registry.create("artifact")
        assert isinstance(artifact_eval, ArtifactEvaluator)
        assert artifact_eval.name == "artifact"

        behavior_eval = registry.create("behavior")
        assert isinstance(behavior_eval, BehaviorEvaluator)
        assert behavior_eval.name == "behavior"

    def test_create_evaluator_not_found(self, registry: EvaluatorRegistry) -> None:
        """Test error when creating unknown evaluator."""
        with pytest.raises(EvaluatorNotFoundError):
            registry.create("nonexistent")


class TestAssertionMappings:
    """Tests for assertion type to evaluator mappings."""

    def test_artifact_assertion_types(self, registry: EvaluatorRegistry) -> None:
        """Test artifact-related assertion types are supported."""
        assert registry.supports_assertion("artifact_exists")
        assert registry.supports_assertion("contains")
        assert registry.supports_assertion("schema")
        assert registry.supports_assertion("sections")

    def test_behavior_assertion_types(self, registry: EvaluatorRegistry) -> None:
        """Test behavior-related assertion types are supported."""
        assert registry.supports_assertion("behavior")
        assert registry.supports_assertion("must_use_tools")
        assert registry.supports_assertion("max_tool_calls")
        assert registry.supports_assertion("min_tool_calls")
        assert registry.supports_assertion("no_errors")
        assert registry.supports_assertion("forbidden_tools")

    def test_get_evaluator_for_assertion(self, registry: EvaluatorRegistry) -> None:
        """Test getting evaluator class for assertion type."""
        artifact_class = registry.get_evaluator_for_assertion("artifact_exists")
        assert artifact_class == ArtifactEvaluator

        behavior_class = registry.get_evaluator_for_assertion("must_use_tools")
        assert behavior_class == BehaviorEvaluator

    def test_get_evaluator_for_unknown_assertion(
        self, registry: EvaluatorRegistry
    ) -> None:
        """Test error when assertion type not supported."""
        with pytest.raises(EvaluatorNotFoundError) as exc_info:
            registry.get_evaluator_for_assertion("unknown_assertion")
        assert "assertion:unknown_assertion" in str(exc_info.value)

    def test_create_for_assertion(self, registry: EvaluatorRegistry) -> None:
        """Test creating evaluator for assertion type."""
        evaluator = registry.create_for_assertion("contains")
        assert isinstance(evaluator, ArtifactEvaluator)

        evaluator = registry.create_for_assertion("no_errors")
        assert isinstance(evaluator, BehaviorEvaluator)

    def test_list_assertion_types(self, registry: EvaluatorRegistry) -> None:
        """Test listing all supported assertion types."""
        assertions = registry.list_assertion_types()
        assert "artifact_exists" in assertions
        assert "contains" in assertions
        assert "must_use_tools" in assertions
        assert "no_errors" in assertions


class TestCustomEvaluatorRegistration:
    """Tests for custom evaluator registration."""

    def test_register_custom_evaluator(self, registry: EvaluatorRegistry) -> None:
        """Test registering a custom evaluator."""

        class CustomEvaluator(Evaluator):
            @property
            def name(self) -> str:
                return "custom"

            async def evaluate(self, task, response, trace, assertion):
                pass

        registry.register("custom", CustomEvaluator)
        assert registry.is_registered("custom")
        evaluator = registry.create("custom")
        assert isinstance(evaluator, CustomEvaluator)

    def test_unregister_evaluator(self, registry: EvaluatorRegistry) -> None:
        """Test unregistering an evaluator."""
        assert registry.is_registered("artifact")
        result = registry.unregister("artifact")
        assert result is True
        assert registry.is_registered("artifact") is False

        result = registry.unregister("artifact")
        assert result is False

    def test_unregister_removes_assertion_mappings(
        self, registry: EvaluatorRegistry
    ) -> None:
        """Test unregistering removes associated assertion mappings."""
        assert registry.supports_assertion("artifact_exists")
        registry.unregister("artifact")
        assert registry.supports_assertion("artifact_exists") is False


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self) -> None:
        """Test get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_create_evaluator_function(self) -> None:
        """Test create_evaluator helper function."""
        evaluator = create_evaluator("artifact")
        assert isinstance(evaluator, ArtifactEvaluator)

    def test_create_evaluator_function_not_found(self) -> None:
        """Test create_evaluator raises error for unknown type."""
        with pytest.raises(EvaluatorNotFoundError):
            create_evaluator("nonexistent")


class TestEvaluatorNotFoundError:
    """Tests for EvaluatorNotFoundError exception."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = EvaluatorNotFoundError("test_type")
        assert error.evaluator_type == "test_type"
        assert "test_type" in str(error)

    def test_error_inheritance(self) -> None:
        """Test error is an Exception."""
        error = EvaluatorNotFoundError("test")
        assert isinstance(error, Exception)
