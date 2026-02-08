"""YAML writer for ATP test suites with proper formatting."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import LiteralScalarString

if TYPE_CHECKING:
    from atp.generator.core import TestSuiteData
    from atp.loader.models import TestSuite


def _create_yaml_instance() -> YAML:
    """Create a configured YAML instance for writing.

    Returns:
        YAML instance with proper indentation settings.
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.preserve_quotes = True
    # Indentation: mapping=2, sequence=4, offset=2
    yaml.map_indent = 2
    yaml.sequence_indent = 4
    yaml.sequence_dash_offset = 2
    return yaml


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove None values and empty structures from dict.

    Args:
        data: Dictionary to clean.

    Returns:
        Cleaned dictionary without None values.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            cleaned = _clean_dict(value)
            if cleaned:  # Only add non-empty dicts
                result[key] = cleaned
        elif isinstance(value, list):
            cleaned_list = _clean_list(value)
            if cleaned_list:  # Only add non-empty lists
                result[key] = cleaned_list
        else:
            result[key] = value
    return result


def _clean_list(data: list[Any]) -> list[Any]:
    """Recursively clean list items.

    Args:
        data: List to clean.

    Returns:
        Cleaned list.
    """
    result: list[Any] = []
    for item in data:
        if item is None:
            continue
        if isinstance(item, dict):
            cleaned = _clean_dict(item)
            if cleaned:
                result.append(cleaned)
        elif isinstance(item, list):
            cleaned_list = _clean_list(item)
            if cleaned_list:
                result.append(cleaned_list)
        else:
            result.append(item)
    return result


def _to_commented_map(data: dict[str, Any]) -> CommentedMap:
    """Convert dict to CommentedMap for proper YAML formatting.

    Args:
        data: Dictionary to convert.

    Returns:
        CommentedMap suitable for ruamel.yaml.
    """
    cm = CommentedMap()
    for key, value in data.items():
        if isinstance(value, dict):
            cm[key] = _to_commented_map(value)
        elif isinstance(value, list):
            cm[key] = _to_commented_seq(value)
        elif isinstance(value, str) and "\n" in value:
            # Use literal block scalar for multiline strings
            cm[key] = LiteralScalarString(value)
        else:
            cm[key] = value
    return cm


def _to_commented_seq(data: list[Any]) -> CommentedSeq:
    """Convert list to CommentedSeq for proper YAML formatting.

    Args:
        data: List to convert.

    Returns:
        CommentedSeq suitable for ruamel.yaml.
    """
    cs = CommentedSeq()
    for item in data:
        if isinstance(item, dict):
            cs.append(_to_commented_map(item))
        elif isinstance(item, list):
            cs.append(_to_commented_seq(item))
        elif isinstance(item, str) and "\n" in item:
            cs.append(LiteralScalarString(item))
        else:
            cs.append(item)
    return cs


def _suite_data_to_dict(suite: TestSuiteData) -> dict[str, Any]:
    """Convert TestSuiteData to dictionary for YAML serialization.

    Args:
        suite: Test suite data to convert.

    Returns:
        Dictionary representation.
    """
    data: dict[str, Any] = {
        "test_suite": suite.name,
        "version": suite.version,
    }

    if suite.description:
        data["description"] = suite.description

    # Add defaults if not using default values
    defaults_dict = _defaults_to_dict(suite.defaults)
    if defaults_dict:
        data["defaults"] = defaults_dict

    # Add agents if present
    if suite.agents:
        data["agents"] = [_agent_to_dict(a) for a in suite.agents]

    # Add tests
    if suite.tests:
        data["tests"] = [_test_to_dict(t) for t in suite.tests]

    return data


def _defaults_to_dict(defaults: Any) -> dict[str, Any]:
    """Convert TestDefaults to dictionary.

    Args:
        defaults: TestDefaults instance.

    Returns:
        Dictionary representation (may be empty if all defaults).
    """
    result: dict[str, Any] = {}

    # Only include non-default values
    if defaults.runs_per_test != 1:
        result["runs_per_test"] = defaults.runs_per_test
    if defaults.timeout_seconds != 300:
        result["timeout_seconds"] = defaults.timeout_seconds

    # Add scoring if not using default weights
    scoring_dict = _scoring_to_dict(defaults.scoring)
    if scoring_dict:
        result["scoring"] = scoring_dict

    # Add constraints if present
    if defaults.constraints:
        constraints_dict = _constraints_to_dict(defaults.constraints)
        if constraints_dict:
            result["constraints"] = constraints_dict

    return result


def _scoring_to_dict(scoring: Any) -> dict[str, Any]:
    """Convert ScoringWeights to dictionary.

    Args:
        scoring: ScoringWeights instance.

    Returns:
        Dictionary representation (may be empty if all defaults).
    """
    result: dict[str, Any] = {}

    # Only include non-default values
    if scoring.quality_weight != 0.4:
        result["quality_weight"] = scoring.quality_weight
    if scoring.completeness_weight != 0.3:
        result["completeness_weight"] = scoring.completeness_weight
    if scoring.efficiency_weight != 0.2:
        result["efficiency_weight"] = scoring.efficiency_weight
    if scoring.cost_weight != 0.1:
        result["cost_weight"] = scoring.cost_weight

    return result


def _constraints_to_dict(constraints: Any) -> dict[str, Any]:
    """Convert Constraints to dictionary.

    Args:
        constraints: Constraints instance.

    Returns:
        Dictionary representation.
    """
    result: dict[str, Any] = {}

    if constraints.max_steps is not None:
        result["max_steps"] = constraints.max_steps
    if constraints.max_tokens is not None:
        result["max_tokens"] = constraints.max_tokens
    if constraints.timeout_seconds is not None:
        result["timeout_seconds"] = constraints.timeout_seconds
    if constraints.allowed_tools is not None:
        result["allowed_tools"] = constraints.allowed_tools
    if constraints.budget_usd is not None:
        result["budget_usd"] = constraints.budget_usd

    return result


def _agent_to_dict(agent: Any) -> dict[str, Any]:
    """Convert AgentConfig to dictionary.

    Args:
        agent: AgentConfig instance.

    Returns:
        Dictionary representation.
    """
    result: dict[str, Any] = {"name": agent.name}

    if agent.type:
        result["type"] = agent.type
    if agent.config:
        result["config"] = agent.config

    return result


def _test_to_dict(test: Any) -> dict[str, Any]:
    """Convert TestDefinition to dictionary.

    Args:
        test: TestDefinition instance.

    Returns:
        Dictionary representation.
    """
    result: dict[str, Any] = {
        "id": test.id,
        "name": test.name,
    }

    if test.description:
        result["description"] = test.description
    if test.tags:
        result["tags"] = list(test.tags)

    # Add task
    result["task"] = _task_to_dict(test.task)

    # Add constraints (only non-default values)
    constraints_dict = _constraints_to_dict(test.constraints)
    if constraints_dict:
        result["constraints"] = constraints_dict

    # Add assertions
    if test.assertions:
        result["assertions"] = [_assertion_to_dict(a) for a in test.assertions]

    # Add scoring if present
    if test.scoring:
        scoring_dict = _scoring_to_dict(test.scoring)
        if scoring_dict:
            result["scoring"] = scoring_dict

    return result


def _task_to_dict(task: Any) -> dict[str, Any]:
    """Convert TaskDefinition to dictionary.

    Args:
        task: TaskDefinition instance.

    Returns:
        Dictionary representation.
    """
    result: dict[str, Any] = {"description": task.description}

    if task.input_data:
        result["input_data"] = task.input_data
    if task.expected_artifacts:
        result["expected_artifacts"] = list(task.expected_artifacts)

    return result


def _assertion_to_dict(assertion: Any) -> dict[str, Any]:
    """Convert Assertion to dictionary.

    Args:
        assertion: Assertion instance.

    Returns:
        Dictionary representation.
    """
    result: dict[str, Any] = {"type": assertion.type}

    if assertion.config:
        result["config"] = assertion.config

    return result


def _test_suite_to_dict(suite: TestSuite) -> dict[str, Any]:
    """Convert TestSuite model to dictionary for YAML serialization.

    Args:
        suite: TestSuite instance.

    Returns:
        Dictionary representation.
    """
    data: dict[str, Any] = {
        "test_suite": suite.test_suite,
        "version": suite.version,
    }

    if suite.description:
        data["description"] = suite.description

    # Add defaults if not using default values
    defaults_dict = _defaults_to_dict(suite.defaults)
    if defaults_dict:
        data["defaults"] = defaults_dict

    # Add agents if present
    if suite.agents:
        data["agents"] = [_agent_to_dict(a) for a in suite.agents]

    # Add tests
    data["tests"] = [_test_to_dict(t) for t in suite.tests]

    return data


class YAMLWriter:
    """YAML writer for ATP test suites.

    Converts TestSuiteData or TestSuite to properly formatted YAML.

    Example:
        >>> writer = YAMLWriter()
        >>> yaml_content = writer.to_yaml(suite_data)
        >>> writer.save(suite_data, "output.yaml")
    """

    def __init__(self) -> None:
        """Initialize the YAML writer."""
        self._yaml = _create_yaml_instance()

    def to_yaml(self, suite: TestSuiteData | TestSuite) -> str:
        """Convert a test suite to YAML string.

        Args:
            suite: TestSuiteData or TestSuite to convert.

        Returns:
            YAML formatted string.
        """
        from atp.generator.core import TestSuiteData
        from atp.loader.models import TestSuite

        if isinstance(suite, TestSuiteData):
            data = _suite_data_to_dict(suite)
        elif isinstance(suite, TestSuite):
            data = _test_suite_to_dict(suite)
        else:
            raise TypeError(
                f"Expected TestSuiteData or TestSuite, got {type(suite).__name__}"
            )

        # Convert to CommentedMap for proper formatting
        commented_data = _to_commented_map(data)

        # Write to string
        stream = StringIO()
        self._yaml.dump(commented_data, stream)
        return stream.getvalue()

    def save(self, suite: TestSuiteData | TestSuite, file_path: str | Path) -> None:
        """Save a test suite to a YAML file.

        Args:
            suite: TestSuiteData or TestSuite to save.
            file_path: Path to output file.
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        from atp.generator.core import TestSuiteData
        from atp.loader.models import TestSuite

        if isinstance(suite, TestSuiteData):
            data = _suite_data_to_dict(suite)
        elif isinstance(suite, TestSuite):
            data = _test_suite_to_dict(suite)
        else:
            raise TypeError(
                f"Expected TestSuiteData or TestSuite, got {type(suite).__name__}"
            )

        # Convert to CommentedMap for proper formatting
        commented_data = _to_commented_map(data)

        # Write to file
        with file_path.open("w", encoding="utf-8") as f:
            self._yaml.dump(commented_data, f)
