"""Evaluator registry for managing and creating evaluators."""

from typing import Any

from .artifact import ArtifactEvaluator
from .base import Evaluator
from .behavior import BehaviorEvaluator
from .code_exec import CodeExecEvaluator
from .llm_judge import LLMJudgeEvaluator


class EvaluatorNotFoundError(Exception):
    """Raised when an evaluator type is not registered."""

    def __init__(self, evaluator_type: str) -> None:
        self.evaluator_type = evaluator_type
        super().__init__(f"Evaluator type not found: {evaluator_type}")


class EvaluatorRegistry:
    """
    Registry for evaluator types.

    Provides factory methods for creating evaluators from configuration.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in evaluators."""
        self._evaluators: dict[str, type[Evaluator]] = {}
        self._assertion_mappings: dict[str, str] = {}

        self.register("artifact", ArtifactEvaluator)
        self.register("behavior", BehaviorEvaluator)
        self.register("llm_judge", LLMJudgeEvaluator)
        self.register("code_exec", CodeExecEvaluator)

        self._register_assertion_mapping("artifact_exists", "artifact")
        self._register_assertion_mapping("contains", "artifact")
        self._register_assertion_mapping("schema", "artifact")
        self._register_assertion_mapping("sections", "artifact")

        self._register_assertion_mapping("behavior", "behavior")
        self._register_assertion_mapping("must_use_tools", "behavior")
        self._register_assertion_mapping("max_tool_calls", "behavior")
        self._register_assertion_mapping("min_tool_calls", "behavior")
        self._register_assertion_mapping("no_errors", "behavior")
        self._register_assertion_mapping("forbidden_tools", "behavior")

        self._register_assertion_mapping("llm_eval", "llm_judge")

        self._register_assertion_mapping("pytest", "code_exec")
        self._register_assertion_mapping("npm", "code_exec")
        self._register_assertion_mapping("custom_command", "code_exec")
        self._register_assertion_mapping("lint", "code_exec")

    def register(
        self,
        evaluator_type: str,
        evaluator_class: type[Evaluator],
    ) -> None:
        """
        Register an evaluator type.

        Args:
            evaluator_type: Unique identifier for the evaluator type.
            evaluator_class: Evaluator class to instantiate.
        """
        self._evaluators[evaluator_type] = evaluator_class

    def _register_assertion_mapping(
        self, assertion_type: str, evaluator_type: str
    ) -> None:
        """Map an assertion type to its evaluator."""
        self._assertion_mappings[assertion_type] = evaluator_type

    def unregister(self, evaluator_type: str) -> bool:
        """
        Unregister an evaluator type.

        Args:
            evaluator_type: Identifier of the evaluator to remove.

        Returns:
            True if evaluator was removed, False if it didn't exist.
        """
        if evaluator_type in self._evaluators:
            del self._evaluators[evaluator_type]
            mappings_to_remove = [
                k for k, v in self._assertion_mappings.items() if v == evaluator_type
            ]
            for k in mappings_to_remove:
                del self._assertion_mappings[k]
            return True
        return False

    def get_evaluator_class(self, evaluator_type: str) -> type[Evaluator]:
        """
        Get the evaluator class for a type.

        Args:
            evaluator_type: Evaluator type identifier.

        Returns:
            Evaluator class.

        Raises:
            EvaluatorNotFoundError: If evaluator type is not registered.
        """
        if evaluator_type not in self._evaluators:
            raise EvaluatorNotFoundError(evaluator_type)
        return self._evaluators[evaluator_type]

    def get_evaluator_for_assertion(self, assertion_type: str) -> type[Evaluator]:
        """
        Get the evaluator class for an assertion type.

        Args:
            assertion_type: The assertion type (e.g., 'artifact_exists').

        Returns:
            Evaluator class that handles this assertion type.

        Raises:
            EvaluatorNotFoundError: If no evaluator handles this assertion type.
        """
        evaluator_type = self._assertion_mappings.get(assertion_type)
        if evaluator_type is None:
            raise EvaluatorNotFoundError(f"assertion:{assertion_type}")
        return self.get_evaluator_class(evaluator_type)

    def create(
        self, evaluator_type: str, config: dict[str, Any] | None = None
    ) -> Evaluator:
        """
        Create an evaluator instance.

        Args:
            evaluator_type: Evaluator type identifier.
            config: Optional configuration for the evaluator (currently unused).

        Returns:
            Configured evaluator instance.

        Raises:
            EvaluatorNotFoundError: If evaluator type is not registered.
        """
        evaluator_class = self.get_evaluator_class(evaluator_type)
        return evaluator_class()

    def create_for_assertion(
        self, assertion_type: str, config: dict[str, Any] | None = None
    ) -> Evaluator:
        """
        Create an evaluator instance for an assertion type.

        Args:
            assertion_type: The assertion type.
            config: Optional configuration for the evaluator.

        Returns:
            Evaluator instance that handles this assertion type.

        Raises:
            EvaluatorNotFoundError: If no evaluator handles this assertion type.
        """
        evaluator_class = self.get_evaluator_for_assertion(assertion_type)
        return evaluator_class()

    def list_evaluators(self) -> list[str]:
        """
        List all registered evaluator types.

        Returns:
            List of evaluator type identifiers.
        """
        return list(self._evaluators.keys())

    def list_assertion_types(self) -> list[str]:
        """
        List all supported assertion types.

        Returns:
            List of assertion type identifiers.
        """
        return list(self._assertion_mappings.keys())

    def is_registered(self, evaluator_type: str) -> bool:
        """
        Check if an evaluator type is registered.

        Args:
            evaluator_type: Evaluator type identifier.

        Returns:
            True if evaluator is registered, False otherwise.
        """
        return evaluator_type in self._evaluators

    def supports_assertion(self, assertion_type: str) -> bool:
        """
        Check if an assertion type is supported.

        Args:
            assertion_type: Assertion type identifier.

        Returns:
            True if assertion type is supported, False otherwise.
        """
        return assertion_type in self._assertion_mappings


_default_registry: EvaluatorRegistry | None = None


def get_registry() -> EvaluatorRegistry:
    """
    Get the global evaluator registry.

    Returns:
        Global EvaluatorRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = EvaluatorRegistry()
    return _default_registry


def create_evaluator(
    evaluator_type: str, config: dict[str, Any] | None = None
) -> Evaluator:
    """
    Create an evaluator using the global registry.

    Args:
        evaluator_type: Evaluator type identifier.
        config: Optional configuration for the evaluator.

    Returns:
        Configured evaluator instance.
    """
    return get_registry().create(evaluator_type, config)
