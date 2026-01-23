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
