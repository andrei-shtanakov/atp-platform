"""Test loader for parsing and validating test suites."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from atp.core.exceptions import ValidationError
from atp.loader.models import TestSuite
from atp.loader.parser import VariableSubstitution, YAMLParser
from atp.loader.schema import validate_schema


class TestLoader:
    """Load and validate test suites from YAML files."""

    def __init__(self, env: dict[str, str] | None = None):
        """Initialize test loader.

        Args:
            env: Custom environment for variable substitution, defaults to os.environ
        """
        self.parser = YAMLParser()
        self.substitution = VariableSubstitution(env=env)

    def load_file(self, file_path: str | Path) -> TestSuite:
        """Load test suite from YAML file.

        Args:
            file_path: Path to test suite YAML file

        Returns:
            Validated TestSuite object

        Raises:
            ParseError: If YAML parsing fails
            ValidationError: If validation fails
        """
        file_path = Path(file_path)

        # Parse YAML
        data = self.parser.parse_file(file_path)

        # Process and validate
        return self._process_data(data, str(file_path))

    def load_string(self, content: str) -> TestSuite:
        """Load test suite from YAML string.

        Args:
            content: YAML content as string

        Returns:
            Validated TestSuite object

        Raises:
            ParseError: If YAML parsing fails
            ValidationError: If validation fails
        """
        # Parse YAML
        data = self.parser.parse_string(content)

        # Process and validate
        return self._process_data(data, None)

    def _process_data(self, data: dict[str, Any], file_path: str | None) -> TestSuite:
        """Process parsed data: substitute variables, validate, and build model.

        Args:
            data: Parsed YAML data
            file_path: Optional file path for error messages

        Returns:
            Validated TestSuite object

        Raises:
            ValidationError: If validation fails
        """
        # Substitute variables
        try:
            data = self.substitution.substitute(data)
        except ValidationError as e:
            # Add file context if available
            if file_path and not e.file_path:
                raise ValidationError(
                    e.message, line=e.line, column=e.column, file_path=file_path
                ) from e
            raise

        # Validate against JSON Schema
        schema_errors = validate_schema(data)
        if schema_errors:
            error_msg = "Schema validation failed:\n  " + "\n  ".join(schema_errors)
            raise ValidationError(error_msg, file_path=file_path)

        # Semantic validation
        self._validate_semantics(data, file_path)

        # Build pydantic model
        try:
            suite = TestSuite(**data)
        except PydanticValidationError as e:
            # Convert pydantic errors to ATP ValidationError
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            error_msg = "Model validation failed:\n  " + "\n  ".join(errors)
            raise ValidationError(error_msg, file_path=file_path) from e

        # Apply defaults inheritance
        suite.apply_defaults()

        return suite

    def _validate_semantics(self, data: dict[str, Any], file_path: str | None) -> None:
        """Perform semantic validation on test suite data.

        Args:
            data: Test suite data
            file_path: Optional file path for error messages

        Raises:
            ValidationError: If semantic validation fails
        """
        errors = []

        # Check for duplicate test IDs
        test_ids = set()
        if "tests" in data and isinstance(data["tests"], list):
            for i, test in enumerate(data["tests"]):
                if not isinstance(test, dict):
                    continue

                test_id = test.get("id")
                if test_id:
                    if test_id in test_ids:
                        errors.append(f"Duplicate test ID '{test_id}' at tests[{i}]")
                    test_ids.add(test_id)

        # Check for duplicate agent names
        agent_names = set()
        if "agents" in data and isinstance(data["agents"], list):
            for i, agent in enumerate(data["agents"]):
                if not isinstance(agent, dict):
                    continue

                agent_name = agent.get("name")
                if agent_name:
                    if agent_name in agent_names:
                        errors.append(
                            f"Duplicate agent name '{agent_name}' at agents[{i}]"
                        )
                    agent_names.add(agent_name)

        # Check scoring weights sum to ~1.0
        def check_scoring_weights(weights: dict, path: str) -> None:
            if not isinstance(weights, dict):
                return

            total = sum(
                weights.get(k, 0.0)
                for k in [
                    "quality_weight",
                    "completeness_weight",
                    "efficiency_weight",
                    "cost_weight",
                ]
            )

            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"{path}: scoring weights sum to {total:.2f}, expected ~1.0"
                )

        # Check defaults scoring weights
        if "defaults" in data and isinstance(data["defaults"], dict):
            if "scoring" in data["defaults"]:
                check_scoring_weights(data["defaults"]["scoring"], "defaults.scoring")

        # Check individual test scoring weights
        if "tests" in data and isinstance(data["tests"], list):
            for i, test in enumerate(data["tests"]):
                if isinstance(test, dict) and "scoring" in test:
                    check_scoring_weights(test["scoring"], f"tests[{i}].scoring")

        # Validate multi-agent test configurations
        self._validate_multi_agent_tests(data, agent_names, errors)

        if errors:
            error_msg = "Semantic validation failed:\n  " + "\n  ".join(errors)
            raise ValidationError(error_msg, file_path=file_path)

    def _validate_multi_agent_tests(
        self, data: dict[str, Any], suite_agent_names: set[str], errors: list[str]
    ) -> None:
        """Validate multi-agent test configurations.

        Args:
            data: Test suite data
            suite_agent_names: Set of agent names defined at suite level
            errors: List to append error messages to
        """
        if "tests" not in data or not isinstance(data["tests"], list):
            return

        for i, test in enumerate(data["tests"]):
            if not isinstance(test, dict):
                continue

            test_agents = test.get("agents")
            mode = test.get("mode")
            test_id = test.get("id", f"tests[{i}]")

            # Validate test-level agents exist in suite-level agents
            if test_agents is not None and isinstance(test_agents, list):
                for agent_name in test_agents:
                    if agent_name not in suite_agent_names:
                        errors.append(
                            f"Test '{test_id}': agent '{agent_name}' not defined "
                            "in suite-level agents"
                        )

            # If mode is specified but no agents, that's an error
            if mode is not None and test_agents is None:
                errors.append(
                    f"Test '{test_id}': 'mode' is set to '{mode}' but 'agents' "
                    "is not specified"
                )

            # If multiple agents but no mode, that's an error
            if (
                test_agents is not None
                and isinstance(test_agents, list)
                and len(test_agents) > 1
                and mode is None
            ):
                errors.append(
                    f"Test '{test_id}': multiple agents specified but 'mode' "
                    "is not set (must be: comparison, collaboration, or handoff)"
                )

            # Validate collaboration mode requires at least 2 agents
            if (
                mode == "collaboration"
                and test_agents is not None
                and isinstance(test_agents, list)
                and len(test_agents) < 2
            ):
                errors.append(
                    f"Test '{test_id}': collaboration mode requires at least 2 agents"
                )

            # Validate handoff mode requires at least 2 agents
            if (
                mode == "handoff"
                and test_agents is not None
                and isinstance(test_agents, list)
                and len(test_agents) < 2
            ):
                errors.append(
                    f"Test '{test_id}': handoff mode requires at least 2 agents"
                )

            # Validate coordinator_agent exists in test agents
            collaboration_config = test.get("collaboration_config")
            if collaboration_config is not None and isinstance(
                collaboration_config, dict
            ):
                coordinator = collaboration_config.get("coordinator_agent")
                if (
                    coordinator is not None
                    and test_agents is not None
                    and isinstance(test_agents, list)
                    and coordinator not in test_agents
                ):
                    errors.append(
                        f"Test '{test_id}': coordinator_agent '{coordinator}' "
                        "must be in the test's agents list"
                    )

            # Validate mode-specific configs don't conflict
            has_comparison_config = test.get("comparison_config") is not None
            has_collaboration_config = test.get("collaboration_config") is not None
            has_handoff_config = test.get("handoff_config") is not None

            if mode == "comparison" and (
                has_collaboration_config or has_handoff_config
            ):
                errors.append(
                    f"Test '{test_id}': comparison mode should not have "
                    "collaboration_config or handoff_config"
                )
            elif mode == "collaboration" and (
                has_comparison_config or has_handoff_config
            ):
                errors.append(
                    f"Test '{test_id}': collaboration mode should not have "
                    "comparison_config or handoff_config"
                )
            elif mode == "handoff" and (
                has_comparison_config or has_collaboration_config
            ):
                errors.append(
                    f"Test '{test_id}': handoff mode should not have "
                    "comparison_config or collaboration_config"
                )
