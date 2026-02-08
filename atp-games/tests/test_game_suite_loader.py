"""Tests for game suite YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from atp.core.exceptions import ParseError, ValidationError

from atp_games.suites.game_suite_loader import GameSuiteLoader, _deep_merge
from atp_games.suites.models import GameSuiteConfig
from atp_games.suites.schema import validate_game_suite_schema

# ── Fixtures ──────────────────────────────────────────────


MINIMAL_SUITE_YAML = """\
type: game_suite
name: Test Suite
game:
  type: prisoners_dilemma
agents:
  - name: agent_a
    adapter: builtin
    strategy: always_cooperate
  - name: agent_b
    adapter: builtin
    strategy: always_defect
"""

FULL_SUITE_YAML = """\
type: game_suite
name: Full PD Suite
version: "2.0"
game:
  type: prisoners_dilemma
  variant: repeated
  config:
    num_rounds: 50
    noise: 0.1
agents:
  - name: tft
    adapter: builtin
    strategy: tit_for_tat
  - name: allc
    adapter: builtin
    strategy: always_cooperate
evaluation:
  episodes: 20
  metrics:
    - type: average_payoff
      weight: 1.0
    - type: exploitability
      weight: 0.5
      config:
        epsilon: 0.2
  thresholds:
    average_payoff:
      min: 1.0
reporting:
  include_strategy_profile: true
  include_payoff_matrix: false
"""

VARIABLE_SUITE_YAML = """\
type: game_suite
name: CI Suite
game:
  type: prisoners_dilemma
agents:
  - name: agent_under_test
    adapter: http
    endpoint: ${AGENT_ENDPOINT}
  - name: baseline
    adapter: builtin
    strategy: tit_for_tat
"""

VARIABLE_DEFAULT_YAML = """\
type: game_suite
name: CI Suite with Default
game:
  type: prisoners_dilemma
agents:
  - name: agent_under_test
    adapter: http
    endpoint: ${AGENT_ENDPOINT:http://localhost:8000}
  - name: baseline
    adapter: builtin
    strategy: tit_for_tat
"""


# ── Schema Validation ────────────────────────────────────


class TestGameSuiteSchema:
    def test_valid_minimal(self) -> None:
        """Minimal valid suite passes schema validation."""
        data: dict[str, Any] = {
            "type": "game_suite",
            "name": "test",
            "game": {"type": "prisoners_dilemma"},
            "agents": [
                {
                    "name": "a",
                    "adapter": "builtin",
                    "strategy": "x",
                },
                {
                    "name": "b",
                    "adapter": "builtin",
                    "strategy": "y",
                },
            ],
        }
        errors = validate_game_suite_schema(data)
        assert errors == []

    def test_missing_required_fields(self) -> None:
        """Missing required fields produce errors."""
        errors = validate_game_suite_schema({})
        assert len(errors) > 0
        assert any("type" in e for e in errors)

    def test_wrong_type_value(self) -> None:
        """Wrong type value produces error."""
        data: dict[str, Any] = {
            "type": "test_suite",
            "name": "test",
            "game": {"type": "pd"},
            "agents": [
                {"name": "a", "adapter": "builtin"},
                {"name": "b", "adapter": "builtin"},
            ],
        }
        errors = validate_game_suite_schema(data)
        assert len(errors) > 0

    def test_too_few_agents(self) -> None:
        """Less than 2 agents produces error."""
        data: dict[str, Any] = {
            "type": "game_suite",
            "name": "test",
            "game": {"type": "pd"},
            "agents": [{"name": "a", "adapter": "builtin"}],
        }
        errors = validate_game_suite_schema(data)
        assert len(errors) > 0

    def test_invalid_adapter_type(self) -> None:
        """Unknown adapter type produces error."""
        data: dict[str, Any] = {
            "type": "game_suite",
            "name": "test",
            "game": {"type": "pd"},
            "agents": [
                {"name": "a", "adapter": "ftp"},
                {"name": "b", "adapter": "builtin"},
            ],
        }
        errors = validate_game_suite_schema(data)
        assert len(errors) > 0

    def test_additional_properties_rejected(self) -> None:
        """Unknown top-level keys are rejected."""
        data: dict[str, Any] = {
            "type": "game_suite",
            "name": "test",
            "game": {"type": "pd"},
            "agents": [
                {"name": "a", "adapter": "builtin"},
                {"name": "b", "adapter": "builtin"},
            ],
            "unknown_key": "value",
        }
        errors = validate_game_suite_schema(data)
        assert len(errors) > 0

    def test_evaluation_config(self) -> None:
        """Evaluation section validates correctly."""
        data: dict[str, Any] = {
            "type": "game_suite",
            "name": "test",
            "game": {"type": "pd"},
            "agents": [
                {"name": "a", "adapter": "builtin"},
                {"name": "b", "adapter": "builtin"},
            ],
            "evaluation": {
                "episodes": 100,
                "metrics": [
                    {"type": "average_payoff", "weight": 1.0},
                ],
            },
        }
        errors = validate_game_suite_schema(data)
        assert errors == []

    def test_evaluation_invalid_episodes(self) -> None:
        """Episodes < 1 produces error."""
        data: dict[str, Any] = {
            "type": "game_suite",
            "name": "test",
            "game": {"type": "pd"},
            "agents": [
                {"name": "a", "adapter": "builtin"},
                {"name": "b", "adapter": "builtin"},
            ],
            "evaluation": {"episodes": 0},
        }
        errors = validate_game_suite_schema(data)
        assert len(errors) > 0


# ── YAML Loading ─────────────────────────────────────────


class TestGameSuiteLoader:
    def test_load_minimal_yaml(self) -> None:
        """Load minimal valid YAML string."""
        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)

        assert isinstance(suite, GameSuiteConfig)
        assert suite.name == "Test Suite"
        assert suite.type == "game_suite"
        assert suite.game.type == "prisoners_dilemma"
        assert len(suite.agents) == 2

    def test_load_full_yaml(self) -> None:
        """Load full YAML with all sections."""
        loader = GameSuiteLoader()
        suite = loader.load_string(FULL_SUITE_YAML)

        assert suite.name == "Full PD Suite"
        assert suite.version == "2.0"
        assert suite.game.variant == "repeated"
        assert suite.game.config["num_rounds"] == 50
        assert suite.game.config["noise"] == 0.1
        assert suite.evaluation.episodes == 20
        assert len(suite.evaluation.metrics) == 2
        assert suite.evaluation.metrics[0].type == "average_payoff"
        assert suite.evaluation.metrics[1].weight == 0.5
        assert suite.evaluation.thresholds["average_payoff"]["min"] == 1.0
        assert suite.reporting.include_payoff_matrix is False

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Load from a file path."""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(MINIMAL_SUITE_YAML)

        loader = GameSuiteLoader()
        suite = loader.load_file(yaml_file)
        assert suite.name == "Test Suite"

    def test_load_missing_file(self) -> None:
        """Loading a non-existent file raises ParseError."""
        loader = GameSuiteLoader()
        with pytest.raises(ParseError):
            loader.load_file("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self) -> None:
        """Loading invalid YAML raises ParseError."""
        loader = GameSuiteLoader()
        with pytest.raises(ParseError):
            loader.load_string("{ invalid: yaml: content:")

    def test_schema_validation_error(self) -> None:
        """Invalid schema data raises ValidationError."""
        loader = GameSuiteLoader()
        with pytest.raises(ValidationError, match="Schema validation"):
            loader.load_string("type: wrong_type\nname: test\n")

    def test_default_evaluation_config(self) -> None:
        """Default evaluation config is applied."""
        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)

        assert suite.evaluation.episodes == 50
        assert suite.evaluation.metrics == []

    def test_default_reporting_config(self) -> None:
        """Default reporting config is applied."""
        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)

        assert suite.reporting.include_strategy_profile is True
        assert suite.reporting.include_payoff_matrix is True
        assert suite.reporting.include_round_by_round is False

    def test_agent_fields_parsed(self) -> None:
        """Agent configuration fields are parsed correctly."""
        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)

        agent_a = suite.agents[0]
        assert agent_a.name == "agent_a"
        assert agent_a.adapter == "builtin"
        assert agent_a.strategy == "always_cooperate"

    def test_http_agent_requires_endpoint(self) -> None:
        """HTTP agent without endpoint raises ValidationError."""
        yaml = """\
type: game_suite
name: test
game:
  type: prisoners_dilemma
agents:
  - name: agent_a
    adapter: http
  - name: agent_b
    adapter: builtin
    strategy: always_cooperate
"""
        loader = GameSuiteLoader()
        with pytest.raises(ValidationError, match="requires 'endpoint'"):
            loader.load_string(yaml)

    def test_builtin_agent_requires_strategy(self) -> None:
        """Builtin agent without strategy raises ValidationError."""
        yaml = """\
type: game_suite
name: test
game:
  type: prisoners_dilemma
agents:
  - name: agent_a
    adapter: builtin
  - name: agent_b
    adapter: builtin
    strategy: always_cooperate
"""
        loader = GameSuiteLoader()
        with pytest.raises(ValidationError, match="requires 'strategy'"):
            loader.load_string(yaml)


# ── Semantic Validation ──────────────────────────────────


class TestSemanticValidation:
    def test_duplicate_agent_names(self) -> None:
        """Duplicate agent names produce validation error."""
        yaml = """\
type: game_suite
name: test
game:
  type: prisoners_dilemma
agents:
  - name: same_name
    adapter: builtin
    strategy: always_cooperate
  - name: same_name
    adapter: builtin
    strategy: always_defect
"""
        loader = GameSuiteLoader()
        with pytest.raises(ValidationError, match="Duplicate agent name"):
            loader.load_string(yaml)

    def test_unknown_game_type(self) -> None:
        """Unknown game type produces validation error."""
        yaml = """\
type: game_suite
name: test
game:
  type: nonexistent_game
agents:
  - name: a
    adapter: builtin
    strategy: x
  - name: b
    adapter: builtin
    strategy: y
"""
        loader = GameSuiteLoader()
        with pytest.raises(ValidationError, match="Unknown game"):
            loader.load_string(yaml)


# ── Variable Substitution ────────────────────────────────


class TestVariableSubstitution:
    def test_env_variable_substitution(self) -> None:
        """Environment variables are substituted."""
        env = {"AGENT_ENDPOINT": "http://test:9000"}
        loader = GameSuiteLoader(env=env)
        suite = loader.load_string(VARIABLE_SUITE_YAML)

        http_agent = suite.agents[0]
        assert http_agent.endpoint == "http://test:9000"

    def test_default_variable_substitution(self) -> None:
        """Variables with defaults use default when env unset."""
        loader = GameSuiteLoader(env={})
        suite = loader.load_string(VARIABLE_DEFAULT_YAML)

        http_agent = suite.agents[0]
        assert http_agent.endpoint == "http://localhost:8000"

    def test_missing_variable_raises(self) -> None:
        """Missing required variable raises ValidationError."""
        loader = GameSuiteLoader(env={})
        with pytest.raises(ValidationError, match="AGENT_ENDPOINT"):
            loader.load_string(VARIABLE_SUITE_YAML)


# ── Suite Inheritance ─────────────────────────────────────


class TestSuiteInheritance:
    def test_extends_base_suite(self, tmp_path: Path) -> None:
        """Child suite inherits from base."""
        base_yaml = """\
type: game_suite
name: Base Suite
game:
  type: prisoners_dilemma
  variant: repeated
  config:
    num_rounds: 100
agents:
  - name: agent_a
    adapter: builtin
    strategy: always_cooperate
  - name: agent_b
    adapter: builtin
    strategy: always_defect
evaluation:
  episodes: 50
"""
        child_yaml = """\
extends: base.yaml
type: game_suite
name: Child Suite
evaluation:
  episodes: 10
"""
        (tmp_path / "base.yaml").write_text(base_yaml)
        child_file = tmp_path / "child.yaml"
        child_file.write_text(child_yaml)

        loader = GameSuiteLoader()
        suite = loader.load_file(child_file)

        # Name overridden
        assert suite.name == "Child Suite"
        # Game inherited
        assert suite.game.type == "prisoners_dilemma"
        assert suite.game.variant == "repeated"
        assert suite.game.config["num_rounds"] == 100
        # Evaluation overridden
        assert suite.evaluation.episodes == 10
        # Agents inherited
        assert len(suite.agents) == 2

    def test_extends_overrides_agents(self, tmp_path: Path) -> None:
        """Child suite can override agents list."""
        base_yaml = """\
type: game_suite
name: Base
game:
  type: prisoners_dilemma
agents:
  - name: a
    adapter: builtin
    strategy: always_cooperate
  - name: b
    adapter: builtin
    strategy: always_defect
"""
        child_yaml = """\
extends: base.yaml
type: game_suite
name: Child
agents:
  - name: tft
    adapter: builtin
    strategy: tit_for_tat
  - name: grim
    adapter: builtin
    strategy: grim_trigger
"""
        (tmp_path / "base.yaml").write_text(base_yaml)
        child_file = tmp_path / "child.yaml"
        child_file.write_text(child_yaml)

        loader = GameSuiteLoader()
        suite = loader.load_file(child_file)

        assert len(suite.agents) == 2
        assert suite.agents[0].name == "tft"
        assert suite.agents[1].name == "grim"

    def test_chained_inheritance(self, tmp_path: Path) -> None:
        """Three-level inheritance chain."""
        grandparent = """\
type: game_suite
name: Grandparent
game:
  type: prisoners_dilemma
  config:
    num_rounds: 200
agents:
  - name: a
    adapter: builtin
    strategy: always_cooperate
  - name: b
    adapter: builtin
    strategy: always_defect
"""
        parent = """\
extends: grandparent.yaml
type: game_suite
name: Parent
game:
  type: prisoners_dilemma
  variant: repeated
"""
        child = """\
extends: parent.yaml
type: game_suite
name: Child
evaluation:
  episodes: 5
"""
        (tmp_path / "grandparent.yaml").write_text(grandparent)
        (tmp_path / "parent.yaml").write_text(parent)
        child_file = tmp_path / "child.yaml"
        child_file.write_text(child)

        loader = GameSuiteLoader()
        suite = loader.load_file(child_file)

        assert suite.name == "Child"
        assert suite.game.variant == "repeated"
        assert suite.game.config["num_rounds"] == 200
        assert suite.evaluation.episodes == 5

    def test_extends_ignored_in_load_string_without_base_dir(
        self,
    ) -> None:
        """load_string ignores extends when no base_dir provided."""
        yaml = """\
extends: base.yaml
type: game_suite
name: Test
game:
  type: prisoners_dilemma
agents:
  - name: a
    adapter: builtin
    strategy: always_cooperate
  - name: b
    adapter: builtin
    strategy: always_defect
"""
        loader = GameSuiteLoader()
        suite = loader.load_string(yaml)
        assert suite.name == "Test"


# ── Deep Merge ────────────────────────────────────────────


class TestDeepMerge:
    def test_simple_merge(self) -> None:
        """Simple key override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Nested dicts are merged recursively."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_list_replacement(self) -> None:
        """Lists are replaced, not concatenated."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_original_unchanged(self) -> None:
        """Original dicts are not mutated."""
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        result = _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}
        assert result == {"a": {"x": 1, "y": 2}}


# ── Resolution ────────────────────────────────────────────


class TestResolution:
    def test_resolve_game(self) -> None:
        """resolve_game creates a Game instance."""
        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)
        game = loader.resolve_game(suite)

        assert game.name == "Prisoner's Dilemma"

    def test_resolve_game_repeated(self) -> None:
        """resolve_game sets num_rounds for repeated variant."""
        loader = GameSuiteLoader()
        suite = loader.load_string(FULL_SUITE_YAML)
        game = loader.resolve_game(suite)

        assert "Prisoner's Dilemma" in game.name

    def test_resolve_agents(self) -> None:
        """resolve_agents creates BuiltinAdapter instances."""
        from atp_games.runner.builtin_adapter import BuiltinAdapter

        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)
        agents = loader.resolve_agents(suite)

        assert len(agents) == 2
        assert "agent_a" in agents
        assert "agent_b" in agents
        assert isinstance(agents["agent_a"], BuiltinAdapter)
        assert isinstance(agents["agent_b"], BuiltinAdapter)

    def test_resolve_run_config(self) -> None:
        """resolve_run_config creates GameRunConfig from evaluation."""
        from atp_games.models import GameRunConfig

        loader = GameSuiteLoader()
        suite = loader.load_string(FULL_SUITE_YAML)
        config = loader.resolve_run_config(suite)

        assert isinstance(config, GameRunConfig)
        assert config.episodes == 20

    def test_resolve_run_config_default(self) -> None:
        """Default run config uses evaluation defaults."""
        loader = GameSuiteLoader()
        suite = loader.load_string(MINIMAL_SUITE_YAML)
        config = loader.resolve_run_config(suite)

        assert config.episodes == 50

    def test_resolve_agents_unknown_strategy(self) -> None:
        """Unknown strategy in resolve_agents raises error."""
        yaml = """\
type: game_suite
name: test
game:
  type: prisoners_dilemma
agents:
  - name: agent_a
    adapter: builtin
    strategy: nonexistent_strategy
  - name: agent_b
    adapter: builtin
    strategy: always_cooperate
"""
        loader = GameSuiteLoader()
        suite = loader.load_string(yaml)
        with pytest.raises(KeyError, match="nonexistent_strategy"):
            loader.resolve_agents(suite)


# ── Builtin YAML Files ───────────────────────────────────


class TestBuiltinYAMLs:
    """Test that builtin YAML files load correctly."""

    @pytest.fixture
    def builtin_dir(self) -> Path:
        return Path(__file__).parent.parent / "atp_games" / "suites" / "builtin"

    def test_prisoners_dilemma_yaml(self, builtin_dir: Path) -> None:
        """prisoners_dilemma.yaml loads successfully."""
        loader = GameSuiteLoader()
        suite = loader.load_file(builtin_dir / "prisoners_dilemma.yaml")

        assert suite.game.type == "prisoners_dilemma"
        assert suite.game.variant == "repeated"
        assert len(suite.agents) == 2
        assert suite.evaluation.episodes == 50

    def test_auction_battery_yaml(self, builtin_dir: Path) -> None:
        """auction_battery.yaml loads successfully."""
        loader = GameSuiteLoader()
        suite = loader.load_file(builtin_dir / "auction_battery.yaml")

        assert suite.game.type == "auction"
        assert len(suite.agents) == 2
        assert suite.evaluation.episodes == 100

    def test_public_goods_yaml(self, builtin_dir: Path) -> None:
        """public_goods.yaml loads successfully."""
        loader = GameSuiteLoader()
        suite = loader.load_file(builtin_dir / "public_goods.yaml")

        assert suite.game.type == "public_goods"
        assert suite.game.variant == "repeated"
        assert len(suite.agents) == 4
        assert suite.evaluation.episodes == 30


# ── Models ────────────────────────────────────────────────


class TestGameSuiteModels:
    def test_game_suite_config_defaults(self) -> None:
        """GameSuiteConfig has correct defaults."""
        from atp_games.suites.models import (
            GameConfig,
            GameSuiteConfig,
        )

        suite = GameSuiteConfig(
            name="test",
            game=GameConfig(type="pd"),
            agents=[
                {"name": "a", "adapter": "builtin"},
                {"name": "b", "adapter": "builtin"},
            ],
        )
        assert suite.type == "game_suite"
        assert suite.version == "1.0"
        assert suite.evaluation.episodes == 50

    def test_game_metric_config(self) -> None:
        """GameMetricConfig parses correctly."""
        from atp_games.suites.models import GameMetricConfig

        metric = GameMetricConfig(
            type="average_payoff",
            weight=0.8,
            config={"threshold": 1.0},
        )
        assert metric.type == "average_payoff"
        assert metric.weight == 0.8
        assert metric.config == {"threshold": 1.0}

    def test_game_agent_config(self) -> None:
        """GameAgentConfig parses correctly."""
        from atp_games.suites.models import GameAgentConfig

        agent = GameAgentConfig(
            name="test",
            adapter="http",
            endpoint="http://localhost:8000",
        )
        assert agent.name == "test"
        assert agent.adapter == "http"
        assert agent.endpoint == "http://localhost:8000"
        assert agent.strategy is None
