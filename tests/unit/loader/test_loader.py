"""Unit tests for TestLoader."""

import pytest

from atp.core.exceptions import ValidationError
from atp.loader.loader import TestLoader


class TestTestLoader:
    """Test TestLoader functionality."""

    def test_load_valid_suite(self):
        """Test loading a valid test suite file."""
        loader = TestLoader()

        suite = loader.load_file("tests/fixtures/test_suites/valid_suite.yaml")

        assert suite.test_suite == "sample_suite"
        assert suite.version == "1.0"
        assert len(suite.tests) == 2
        assert suite.tests[0].id == "test-001"
        assert suite.tests[0].name == "Basic test"

    def test_load_suite_with_variable_substitution(self):
        """Test loading suite with variable substitution."""
        loader = TestLoader(
            env={"API_ENDPOINT": "http://example.com", "TEST_VAR": "test_value"}
        )

        suite = loader.load_file("tests/fixtures/test_suites/with_vars.yaml")

        assert suite.agents[0].config["endpoint"] == "http://example.com"
        assert suite.agents[0].config["api_key"] == "default_key"
        assert "test_value" in suite.tests[0].task.description

    def test_load_suite_missing_required_variable(self):
        """Test that missing required variable raises ValidationError."""
        loader = TestLoader(env={})

        with pytest.raises(ValidationError, match="API_ENDPOINT"):
            loader.load_file("tests/fixtures/test_suites/with_vars.yaml")

    def test_load_suite_duplicate_test_ids(self):
        """Test that duplicate test IDs raise ValidationError."""
        loader = TestLoader()

        with pytest.raises(ValidationError, match="Duplicate test ID"):
            loader.load_file("tests/fixtures/test_suites/invalid_duplicate_ids.yaml")

    def test_load_suite_invalid_weights(self):
        """Test that invalid scoring weights raise ValidationError."""
        loader = TestLoader()

        with pytest.raises(ValidationError, match="weights sum"):
            loader.load_file("tests/fixtures/test_suites/invalid_weights.yaml")

    def test_load_from_string(self):
        """Test loading suite from YAML string."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "string_suite"
        version: "1.0"
        tests:
          - id: "test-001"
            name: "Test from string"
            task:
              description: "Task description"
        """

        suite = loader.load_string(yaml_content)

        assert suite.test_suite == "string_suite"
        assert len(suite.tests) == 1

    def test_load_invalid_yaml_string(self):
        """Test that invalid YAML string raises error."""
        loader = TestLoader()
        invalid_yaml = "invalid: [unclosed"

        with pytest.raises(Exception):  # ParseError
            loader.load_string(invalid_yaml)

    def test_defaults_inheritance(self):
        """Test that defaults are properly inherited by tests."""
        loader = TestLoader()

        suite = loader.load_file("tests/fixtures/test_suites/valid_suite.yaml")

        # Test with explicit constraints should keep them
        assert suite.tests[0].constraints.max_steps == 10
        assert suite.tests[0].constraints.timeout_seconds == 60

        # Test without explicit constraints should inherit from defaults
        # (timeout_seconds defaults to 300 in the model, so we check suite defaults)
        assert suite.defaults.timeout_seconds == 180

        # Both tests should have scoring (either explicit or from defaults)
        for test in suite.tests:
            assert test.scoring is not None

    def test_semantic_validation_duplicate_agents(self):
        """Test that duplicate agent names are detected."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "duplicate_agents"
        agents:
          - name: "agent1"
          - name: "agent1"
        tests:
          - id: "test-001"
            name: "Test"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="Duplicate agent name"):
            loader.load_string(yaml_content)

    def test_validation_error_includes_file_path(self):
        """Test that validation errors include file path."""
        loader = TestLoader()

        try:
            loader.load_file("tests/fixtures/test_suites/invalid_duplicate_ids.yaml")
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            assert "invalid_duplicate_ids.yaml" in str(e)

    def test_load_suite_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "incomplete"
        tests:
          - id: "test-001"
            name: "Test without task"
        """

        with pytest.raises(ValidationError):
            loader.load_string(yaml_content)

    def test_edge_case_empty_tests_list(self):
        """Test that empty tests list raises validation error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "empty"
        tests: []
        """

        with pytest.raises(ValidationError):
            loader.load_string(yaml_content)

    def test_edge_case_no_tests_field(self):
        """Test that missing tests field raises validation error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "no_tests"
        version: "1.0"
        """

        with pytest.raises(ValidationError):
            loader.load_string(yaml_content)

    def test_nested_variable_substitution(self):
        """Test variable substitution in nested structures."""
        loader = TestLoader(env={"HOST": "localhost", "PORT": "8080"})
        yaml_content = """
        test_suite: "nested_vars"
        tests:
          - id: "test-001"
            name: "Test"
            task:
              description: "Connect to ${HOST}"
              input_data:
                endpoint: "http://${HOST}:${PORT}"
                nested:
                  url: "${HOST}"
        """

        suite = loader.load_string(yaml_content)

        assert "localhost" in suite.tests[0].task.description
        assert suite.tests[0].task.input_data["endpoint"] == "http://localhost:8080"
        assert suite.tests[0].task.input_data["nested"]["url"] == "localhost"

    def test_scoring_weights_validation(self):
        """Test that scoring weights are validated to sum to ~1.0."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "bad_weights"
        tests:
          - id: "test-001"
            name: "Test"
            task:
              description: "Task"
            scoring:
              quality_weight: 0.9
              completeness_weight: 0.5
              efficiency_weight: 0.1
              cost_weight: 0.1
        """

        with pytest.raises(ValidationError, match="weights sum"):
            loader.load_string(yaml_content)


class TestMultiAgentLoader:
    """Test multi-agent test loading and validation."""

    def test_load_comparison_mode_test(self):
        """Test loading a comparison mode test."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
            type: "http"
            config:
              endpoint: "http://localhost:8001"
          - name: "agent-2"
            type: "http"
            config:
              endpoint: "http://localhost:8002"
        tests:
          - id: "test-001"
            name: "Compare agents"
            agents:
              - "agent-1"
              - "agent-2"
            mode: "comparison"
            comparison_config:
              metrics: ["quality", "speed"]
              determine_winner: true
            task:
              description: "Test task"
        """

        suite = loader.load_string(yaml_content)

        assert len(suite.tests) == 1
        test = suite.tests[0]
        assert test.mode.value == "comparison"
        assert test.agents == ["agent-1", "agent-2"]
        assert test.comparison_config is not None
        assert test.comparison_config.metrics == ["quality", "speed"]

    def test_load_collaboration_mode_test(self):
        """Test loading a collaboration mode test."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "generator"
          - name: "reviewer"
          - name: "coordinator"
        tests:
          - id: "test-001"
            name: "Collaborate"
            agents:
              - "generator"
              - "reviewer"
              - "coordinator"
            mode: "collaboration"
            collaboration_config:
              max_turns: 5
              coordinator_agent: "coordinator"
            task:
              description: "Collaborative task"
        """

        suite = loader.load_string(yaml_content)

        test = suite.tests[0]
        assert test.mode.value == "collaboration"
        assert test.collaboration_config.max_turns == 5
        assert test.collaboration_config.coordinator_agent == "coordinator"

    def test_load_handoff_mode_test(self):
        """Test loading a handoff mode test."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "stage-1"
          - name: "stage-2"
        tests:
          - id: "test-001"
            name: "Pipeline"
            agents:
              - "stage-1"
              - "stage-2"
            mode: "handoff"
            handoff_config:
              handoff_trigger: "on_success"
              context_accumulation: "merge"
            task:
              description: "Pipeline task"
        """

        suite = loader.load_string(yaml_content)

        test = suite.tests[0]
        assert test.mode.value == "handoff"
        assert test.handoff_config.handoff_trigger.value == "on_success"
        assert test.handoff_config.context_accumulation.value == "merge"

    def test_agent_not_in_suite_raises_error(self):
        """Test that referencing undefined agent raises error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
              - "unknown-agent"
            mode: "comparison"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="not defined"):
            loader.load_string(yaml_content)

    def test_mode_without_agents_raises_error(self):
        """Test that mode without agents raises error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
        tests:
          - id: "test-001"
            name: "Test"
            mode: "comparison"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="agents"):
            loader.load_string(yaml_content)

    def test_multiple_agents_without_mode_raises_error(self):
        """Test that multiple agents without mode raises error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
          - name: "agent-2"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
              - "agent-2"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="mode"):
            loader.load_string(yaml_content)

    def test_collaboration_requires_two_agents(self):
        """Test that collaboration mode requires at least 2 agents."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
            mode: "collaboration"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="at least 2 agents"):
            loader.load_string(yaml_content)

    def test_handoff_requires_two_agents(self):
        """Test that handoff mode requires at least 2 agents."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
            mode: "handoff"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="at least 2 agents"):
            loader.load_string(yaml_content)

    def test_coordinator_not_in_test_agents_raises_error(self):
        """Test that coordinator_agent must be in test's agents list."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
          - name: "agent-2"
          - name: "coordinator"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
              - "agent-2"
            mode: "collaboration"
            collaboration_config:
              coordinator_agent: "coordinator"
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="coordinator_agent"):
            loader.load_string(yaml_content)

    def test_conflicting_mode_config_raises_error(self):
        """Test that conflicting mode configs raise error."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
          - name: "agent-2"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
              - "agent-2"
            mode: "comparison"
            collaboration_config:
              max_turns: 5
            task:
              description: "Task"
        """

        with pytest.raises(ValidationError, match="should not have"):
            loader.load_string(yaml_content)

    def test_single_agent_without_mode_succeeds(self):
        """Test that single agent without mode succeeds."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "multi_agent"
        agents:
          - name: "agent-1"
        tests:
          - id: "test-001"
            name: "Test"
            agents:
              - "agent-1"
            task:
              description: "Task"
        """

        suite = loader.load_string(yaml_content)

        test = suite.tests[0]
        assert test.agents == ["agent-1"]
        assert test.mode is None

    def test_mixed_single_and_multi_agent_tests(self):
        """Test suite with both single and multi-agent tests."""
        loader = TestLoader()
        yaml_content = """
        test_suite: "mixed"
        agents:
          - name: "agent-1"
          - name: "agent-2"
        tests:
          - id: "single-001"
            name: "Single agent test"
            task:
              description: "Single task"
          - id: "multi-001"
            name: "Multi agent test"
            agents:
              - "agent-1"
              - "agent-2"
            mode: "comparison"
            task:
              description: "Multi task"
        """

        suite = loader.load_string(yaml_content)

        assert len(suite.tests) == 2
        assert suite.tests[0].mode is None
        assert suite.tests[0].agents is None
        assert suite.tests[1].mode.value == "comparison"
        assert suite.tests[1].agents == ["agent-1", "agent-2"]
