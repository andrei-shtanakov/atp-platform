"""Game suite YAML loader with validation and resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from atp.adapters.base import AgentAdapter
from atp.core.exceptions import ValidationError
from atp.loader.parser import VariableSubstitution, YAMLParser
from pydantic import ValidationError as PydanticValidationError

from atp_games.models import GameRunConfig
from atp_games.suites.models import (
    GameAgentConfig,
    GameSuiteConfig,
)
from atp_games.suites.schema import validate_game_suite_schema


class GameSuiteLoader:
    """Load and validate game suite YAML files.

    Parses YAML, validates against schema, resolves game types
    via GameRegistry, creates agent adapters, and instantiates
    evaluators.
    """

    def __init__(self, env: dict[str, str] | None = None) -> None:
        """Initialize game suite loader.

        Args:
            env: Custom environment for variable substitution,
                defaults to os.environ.
        """
        self.parser = YAMLParser()
        self.substitution = VariableSubstitution(env=env)

    def load_file(self, file_path: str | Path) -> GameSuiteConfig:
        """Load game suite from YAML file.

        Args:
            file_path: Path to game suite YAML file.

        Returns:
            Validated GameSuiteConfig.

        Raises:
            ParseError: If YAML parsing fails.
            ValidationError: If validation fails.
        """
        file_path = Path(file_path)
        data = self.parser.parse_file(file_path)

        # Handle inheritance
        if "extends" in data:
            base_path = file_path.parent / data.pop("extends")
            data = self._merge_with_base(base_path, data)

        return self._process_data(data, str(file_path))

    def load_string(
        self,
        content: str,
        base_dir: Path | None = None,
    ) -> GameSuiteConfig:
        """Load game suite from YAML string.

        Args:
            content: YAML content as string.
            base_dir: Base directory for resolving extends paths.

        Returns:
            Validated GameSuiteConfig.

        Raises:
            ParseError: If YAML parsing fails.
            ValidationError: If validation fails.
        """
        data = self.parser.parse_string(content)

        if "extends" in data and base_dir is not None:
            base_path = base_dir / data.pop("extends")
            data = self._merge_with_base(base_path, data)
        elif "extends" in data:
            data.pop("extends")

        return self._process_data(data, None)

    def _merge_with_base(
        self,
        base_path: Path,
        child_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Load base suite and deep-merge child on top.

        Args:
            base_path: Path to the base YAML file.
            child_data: Child suite data to merge on top.

        Returns:
            Merged data dict.
        """
        base_data = self.parser.parse_file(base_path)

        # Recursively handle base inheritance
        if "extends" in base_data:
            grandparent_path = base_path.parent / base_data.pop("extends")
            base_data = self._merge_with_base(grandparent_path, base_data)

        return _deep_merge(base_data, child_data)

    def _process_data(
        self,
        data: dict[str, Any],
        file_path: str | None,
    ) -> GameSuiteConfig:
        """Process parsed data: substitute, validate, build model.

        Args:
            data: Parsed YAML data.
            file_path: Optional file path for error messages.

        Returns:
            Validated GameSuiteConfig.

        Raises:
            ValidationError: If validation fails.
        """
        # Substitute variables
        try:
            data = self.substitution.substitute(data)
        except ValidationError as e:
            if file_path and not e.file_path:
                raise ValidationError(
                    e.message,
                    line=e.line,
                    column=e.column,
                    file_path=file_path,
                ) from e
            raise

        # Validate against JSON Schema
        schema_errors = validate_game_suite_schema(data)
        if schema_errors:
            error_msg = "Schema validation failed:\n  " + "\n  ".join(schema_errors)
            raise ValidationError(error_msg, file_path=file_path)

        # Semantic validation
        self._validate_semantics(data, file_path)

        # Build pydantic model
        try:
            suite = GameSuiteConfig(**data)
        except PydanticValidationError as e:
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            error_msg = "Model validation failed:\n  " + "\n  ".join(errors)
            raise ValidationError(error_msg, file_path=file_path) from e

        return suite

    def _validate_semantics(
        self,
        data: dict[str, Any],
        file_path: str | None,
    ) -> None:
        """Perform semantic validation on game suite data.

        Args:
            data: Game suite data.
            file_path: Optional file path for error messages.

        Raises:
            ValidationError: If semantic validation fails.
        """
        errors: list[str] = []

        # Check for duplicate agent names
        agent_names: set[str] = set()
        agents = data.get("agents", [])
        if isinstance(agents, list):
            for i, agent in enumerate(agents):
                if not isinstance(agent, dict):
                    continue
                name = agent.get("name")
                if name:
                    if name in agent_names:
                        errors.append(f"Duplicate agent name '{name}' at agents[{i}]")
                    agent_names.add(name)

        # Validate builtin agents have strategy
        if isinstance(agents, list):
            for i, agent in enumerate(agents):
                if not isinstance(agent, dict):
                    continue
                if agent.get("adapter") == "builtin":
                    if not agent.get("strategy"):
                        errors.append(
                            f"agents[{i}]: builtin adapter requires 'strategy'"
                        )
                elif agent.get("adapter") in ("http", "docker", "cli"):
                    if not agent.get("endpoint"):
                        errors.append(
                            f"agents[{i}]: {agent.get('adapter')}"
                            " adapter requires 'endpoint'"
                        )

        # Validate game type is registered
        game = data.get("game", {})
        if isinstance(game, dict):
            game_type = game.get("type")
            if game_type:
                try:
                    from game_envs import GameRegistry

                    GameRegistry.get(game_type)
                except KeyError as e:
                    errors.append(str(e).strip("'\""))

        if errors:
            error_msg = "Semantic validation failed:\n  " + "\n  ".join(errors)
            raise ValidationError(error_msg, file_path=file_path)

    def resolve_game(
        self,
        suite: GameSuiteConfig,
    ) -> Any:
        """Resolve game instance from suite config.

        Args:
            suite: Parsed game suite configuration.

        Returns:
            A Game instance from the registry.
        """
        from game_envs import GameRegistry

        config = dict(suite.game.config)
        if suite.game.variant == "repeated" and "num_rounds" not in config:
            config["num_rounds"] = 100

        return GameRegistry.create(suite.game.type, config)

    def resolve_agents(
        self,
        suite: GameSuiteConfig,
    ) -> dict[str, AgentAdapter]:
        """Resolve agent adapters from suite config.

        Args:
            suite: Parsed game suite configuration.

        Returns:
            Dict mapping agent name to adapter instance.
        """
        agents: dict[str, AgentAdapter] = {}
        for agent_cfg in suite.agents:
            agents[agent_cfg.name] = _create_agent_adapter(agent_cfg)
        return agents

    def resolve_run_config(
        self,
        suite: GameSuiteConfig,
    ) -> GameRunConfig:
        """Create GameRunConfig from evaluation settings.

        Args:
            suite: Parsed game suite configuration.

        Returns:
            GameRunConfig with episodes from evaluation config.
        """
        return GameRunConfig(
            episodes=suite.evaluation.episodes,
        )


def _create_agent_adapter(
    agent_cfg: GameAgentConfig,
) -> AgentAdapter:
    """Create an agent adapter from config.

    Args:
        agent_cfg: Agent configuration.

    Returns:
        An AgentAdapter instance.

    Raises:
        ValidationError: If agent config is invalid.
    """
    if agent_cfg.adapter == "builtin":
        from game_envs import StrategyRegistry

        from atp_games.runner.builtin_adapter import BuiltinAdapter

        if not agent_cfg.strategy:
            raise ValidationError(
                f"Agent '{agent_cfg.name}': builtin adapter requires 'strategy'"
            )
        strategy = StrategyRegistry.create(agent_cfg.strategy)
        return BuiltinAdapter(strategy=strategy)

    from atp.adapters import create_adapter

    config = dict(agent_cfg.config)
    if agent_cfg.endpoint:
        config["endpoint"] = agent_cfg.endpoint

    return create_adapter(agent_cfg.adapter, config)


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge override dict on top of base dict.

    Lists are replaced, not concatenated. Dicts are merged
    recursively.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        Merged dictionary (new dict, inputs unchanged).
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
